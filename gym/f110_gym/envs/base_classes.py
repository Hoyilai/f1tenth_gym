# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Prototype of base classes
Replacement of the old RaceCar, Simulator classes in C++
Author: Hongrui Zheng
"""
from enum import Enum
import warnings

import numpy as np
from numba import njit

from math import atan


from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple

from .dynamic_models import vehicle_dynamics_st_4w, vehicle_dynamics_ks_4w, accl_constraints, steering_constraint


class Integrator(Enum):
    RK4 = 1
    Euler = 2

    @staticmethod
    def step(delta, delta_rear, v, x, y, theta, wb):
        # This function will now need to use the dynamics function for 4-wheel steering.
        # Use the `vehicle_dynamics_st_4w` or `vehicle_dynamics_ks_4w` based on the model you want.
        # Assuming vehicle_dynamics_st_4w:
        x_dot, y_dot, theta_dot = vehicle_dynamics_st_4w(v, delta, delta_rear, theta, wb)
        x += x_dot * Integrator.dt
        y += y_dot * Integrator.dt
        theta += theta_dot * Integrator.dt
        return x, y, theta



class RaceCar(object):
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        params (dict): vehicle parameters dictionary
        is_ego (bool): ego identifier
        time_step (float): physics timestep
        num_beams (int): number of beams in laser
        fov (float): field of view of laser
        state (np.ndarray (7, )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
        odom (np.ndarray(13, )): odometry vector [x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        accel (float): current acceleration input
        steer_angle_vel (float): current steering velocity input
        in_collision (bool): collision indicator

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(self, params, seed, is_ego=False, time_step=0.01, num_beams=1080, fov=4.7, integrator=Integrator.Euler):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser

        Returns:
            None
        """

        # initialization
        self.scan = None
        self.scan = np.zeros(num_beams)

        self.params = params
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.integrator = integrator
        if self.integrator is Integrator.RK4:
            warnings.warn(f"Chosen integrator is RK4. This is different from previous versions of the gym.")

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7, ))

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer = np.empty((0, ))
        self.steer_buffer_size = 2

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if RaceCar.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((num_beams, ))
            RaceCar.scan_angles = np.zeros((num_beams, ))
            RaceCar.side_distances = np.zeros((num_beams, ))

            dist_sides = params['width']/2.
            dist_fr = (params['lf']+params['lr'])/2.

            for i in range(num_beams):
                angle = -fov/2. + i*scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi/2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi/2.)
                        to_fr = dist_fr / np.sin(angle - np.pi/2.)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi/2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi/2)
                        to_fr = dist_fr / np.sin(-angle - np.pi/2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.params = params
    
    def set_map(self, map_path, map_ext):
        """
        Sets the map for the scan simulator.

        Args:
            map_path (str): absolute path to the map yaml file.
            map_ext (str): extension of the map image file.
        """
        # Initialize the scan simulator if it hasn't been already
        if RaceCar.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(self.num_beams, self.fov)
        # Set the map for the scan simulator
        RaceCar.scan_simulator.set_map(map_path, map_ext)

    def ray_cast_agents(self, scan):
        """
        Ray cast onto other agents in the env, modify original scan

        Args:
            scan (np.ndarray, (n, )): original scan range array

        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan
        
        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(opp_pose, self.params['length'], self.params['width'])

            new_scan = ray_cast(np.append(self.state[0:2], self.state[4]), new_scan, self.scan_angles, opp_vertices)

        return new_scan

    def check_ttc(self, current_scan):
        """
        Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
from dynamic_models import accl_constraints, steering_constraint
        Note that this does NOT check collision with other agents.

        state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            current_scan

        Returns:
            None
        """
        
        if current_scan is None:
            raise ValueError("current_scan is None!")
        in_collision = check_ttc_jit(current_scan, self.state[3], self.scan_angles, self.cosines, self.side_distances, self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, front_steer, rear_steer, vel):
        """
        Steps the vehicle's physical simulation

        Args:
            front_steer (float): desired front steering angle
            rear_steer (float): desired rear steering angle
            vel (float): desired longitudinal velocity

        Returns:
            None
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
        steer = 0.
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.
            self.steer_buffer = np.append(front_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(front_steer, self.steer_buffer)

        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(vel, front_steer, self.state[3], self.state[2], self.params['sv_max'], self.params['a_max'], self.params['v_max'], self.params['v_min'])

        # Apply constraints to acceleration and steering velocity
        accl = accl_constraints(self.state[3], accl, self.params['v_switch'], self.params['a_max'], self.params['v_min'], self.params['v_max'])
        sv = steering_constraint(self.state[2], sv, self.params['s_min'], self.params['s_max'], self.params['sv_min'], self.params['sv_max'])

        if self.integrator is Integrator.RK4:
            # RK4 integration with vehicle_dynamics_st_4w
            u = np.array([sv, accl, rear_steer])  # Control inputs including rear steer
            dt = self.time_step

            k1 = vehicle_dynamics_st_4w(self.state, u, self.params['mu'], self.params['C_Sf'], self.params['C_Sr'], self.params['lf'], self.params['lr'], self.params['h'], self.params['m'], self.params['I'], self.params['s_min'], self.params['s_max'], self.params['sv_min'], self.params['sv_max'], self.params['v_switch'], self.params['a_max'], self.params['v_min'], self.params['v_max'])
            k2 = vehicle_dynamics_st_4w(self.state + dt/2 * k1, u, self.params['mu'], self.params['C_Sf'], self.params['C_Sr'], self.params['lf'], self.params['lr'], self.params['h'], self.params['m'], self.params['I'], self.params['s_min'], self.params['s_max'], self.params['sv_min'], self.params['sv_max'], self.params['v_switch'], self.params['a_max'], self.params['v_min'], self.params['v_max'])
            k3 = vehicle_dynamics_st_4w(self.state + dt/2 * k2, u, self.params['mu'], self.params['C_Sf'], self.params['C_Sr'], self.params['lf'], self.params['lr'], self.params['h'], self.params['m'], self.params['I'], self.params['s_min'], self.params['s_max'], self.params['sv_min'], self.params['sv_max'], self.params['v_switch'], self.params['a_max'], self.params['v_min'], self.params['v_max'])
            k4 = vehicle_dynamics_st_4w(self.state + dt * k3, u, self.params['mu'], self.params['C_Sf'], self.params['C_Sr'], self.params['lf'], self.params['lr'], self.params['h'], self.params['m'], self.params['I'], self.params['s_min'], self.params['s_max'], self.params['sv_min'], self.params['sv_max'], self.params['v_switch'], self.params['a_max'], self.params['v_min'], self.params['v_max'])

            # Update state
            self.state = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        elif self.integrator is Integrator.Euler:
            # Euler integration with vehicle_dynamics_st_4w
            u = np.array([sv, accl, rear_steer])  # Control inputs including rear steer
            dt = self.time_step

            # Use the vehicle_dynamics_st_4w function for Euler integration
            f = vehicle_dynamics_st_4w(self.state, u, self.params['mu'], self.params['C_Sf'], self.params['C_Sr'], self.params['lf'], self.params['lr'], self.params['h'], self.params['m'], self.params['I'], self.params['s_min'], self.params['s_max'], self.params['sv_min'], self.params['sv_max'], self.params['v_switch'], self.params['a_max'], self.params['v_min'], self.params['v_max'])

            # Update state with Euler method
            self.state = self.state + dt * f

        else:
            raise SyntaxError(f"Invalid Integrator Specified. Provided {self.integrator.name}. Please choose RK4 or Euler")

            # Bound yaw angle for wrapping around 2*pi
        if self.state[4] > 2*np.pi:
            self.state[4] = self.state[4] - 2*np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2*np.pi

    # The function vehicle_dynamics_st_4w is expected to return the derivatives of the state
    # It is important to ensure that the vehicle_dynamics_st_4w function is properly defined to return the correct derivatives



    

        # update scan
        
        #front_steer = control_inputs[i, 0]
        #rear_steer = control_inputs[i, 1]
        #speed = control_inputs[i, 2]

        #current_scan = agent.update_pose(front_steer, rear_steer, speed)
        


        #return current_scan

    def update_opp_poses(self, opp_poses):
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses


    def update_scan(self, agent_scans, agent_index):
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """
        if agent_scans is None:
            raise ValueError("agent_scans is None in update_scan!")
        if agent_index >= len(agent_scans) or agent_scans[agent_index] is None:
            print(f"Length of agent_scans: {len(agent_scans)}, agent_index: {agent_index}, agent_scans[{agent_index}]: {agent_scans[agent_index]}")
            raise ValueError(f"Invalid scan for index {agent_index} in agent_scans!")
        self.scan = agent_scans[agent_index]

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)
            # Check if new_scan is None
        if new_scan is None:
            new_scan = self.scan  # Keep the old scan if new_scan is None

        agent_scans[agent_index] = new_scan

    def reset(self, pose):
        """
        Resets the vehicle to a pose

        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to. Should be a 1D array with 3 elements (x, y, theta).

        Returns:
            None
        """
        # Ensure that pose is a NumPy array with the correct shape
        pose = np.asarray(pose)
        if pose.ndim != 1 or pose.size != 3:
            raise ValueError("pose must be a 1D array with three elements")

        # Reset the state of the vehicle
        self.state = np.zeros((7, ))
        self.state[0] = pose[0]  # x position
        self.state[1] = pose[1]  # y position
        self.state[4] = pose[2]  # orientation theta

        # Clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # Clear collision indicator
        self.in_collision = False

        # Clear steering buffer
        self.steer_buffer = np.empty((0, ))

        # Reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)



class Simulator(object):
    """
    Simulator class, handles the interaction and update of all vehicles in the environment

    Data Members:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """
    
    def set_map(self, map_path, map_ext):
        """
        Sets the map for each agent's scan simulator.
        
        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file
        """
        # This will call set_map on each RaceCar instance, which should have the method defined.
        for agent in self.agents:
            agent.set_map(map_path, map_ext)

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): ego vehicle's index in list of agents

        Returns:
            None
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))


        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True, time_step=self.time_step, integrator=integrator)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed, is_ego=False, time_step=self.time_step, integrator=integrator)
                self.agents.append(agent)


    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2],self.agents[i].state[4]), self.params['length'], self.params['width'])
        self.collisions, self.collision_idx = collision_multiple(all_vertices)


    def step(self, control_inputs):
        """
        Steps the simulation environment.

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents, current laser scan of each agent, collision indicators, etc.
        """
        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # Update each agent's pose
            agent.update_pose(control_inputs[i, 0], control_inputs[i, 1], control_inputs[i, 2])
            
            # Assuming each agent has a scan attribute that stores the current scan.
            current_scan = agent.scan
            agent_scans.append(current_scan)

            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i+1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            if not agent_scans:
                raise ValueError("agent_scans is empty in step!")
            
            agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.

        # fill in observations
        observations = {
            'ego_idx': self.ego_idx,
            'scans': agent_scans,  # directly use agent_scans list
            'poses_x': [agent.state[0] for agent in self.agents],
            'poses_y': [agent.state[1] for agent in self.agents],
            'poses_theta': [agent.state[4] for agent in self.agents],
            'linear_vels_x': [agent.state[3] for agent in self.agents],
            'linear_vels_y': [0. for agent in self.agents],
            'ang_vels_z': [agent.state[5] for agent in self.agents],
            'collisions': self.collisions
        }

        return observations


    def reset(self, poses):
        """
        Resets the simulator and all agents to specified poses.

        Args:
            poses (list of np.ndarray): List of poses to reset each agent to. Each pose should be a 1D array with three elements (x, y, theta).

        Returns:
            None
        """
        if len(poses) != self.num_agents:
            raise ValueError("Number of poses must match the number of agents in the simulator.")

        # Reset each agent with its corresponding pose
        for i, pose in enumerate(poses):
            if not (isinstance(pose, np.ndarray) and pose.ndim == 1 and pose.size == 3):
                raise ValueError(f"Pose for agent {i} must be a 1D array with three elements")
            self.agents[i].reset(pose)

        # Reset any other environment-specific properties here
        # Example: self.some_environment_property = initial_value

        # Reset collisions
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))




