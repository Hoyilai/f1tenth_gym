import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
import pyglet
pyglet.options['shadow_window'] = False

import logging


import sys
sys.path.append('/home/larry1129/f1tenth_gym/gym/f110_gym/envs')
from rendering import CAR_LENGTH, CAR_WIDTH


from numba import njit

from pyglet.gl import GL_POINTS

initial_front_steer = 0.0  # No steering input at the start
initial_rear_steer = 0.0   # No steering input at the start
initial_speed = 0.0        # Car starts from a standstill


# Configure logging
logging.basicConfig(filename='/home/larry1129/f1tenth_gym/logfile.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')





def wrap_angle(angle):
    """Wraps the angle to the range of -pi to pi."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def scale_steering(steering_angle, max_steering_angle):
    """Scales the steering angle to be within the maximum limits."""
    return np.clip(steering_angle, -max_steering_angle, max_steering_angle)



def proportional_speed_control(front_steering_angle, rear_steering_angle, max_speed, min_speed, max_steering_angle):
    """Adjusts the speed of the vehicle based on the steering angles."""
    steering_magnitude = np.hypot(front_steering_angle, rear_steering_angle)
    steering_scale = min(steering_magnitude / max_steering_angle, 1)
    speed = max_speed * (1 - steering_scale**2)  # Squaring scale for more sensitivity
    speed = np.clip(speed, min_speed, max_speed)
    return speed

def rate_limit_steering(current_steering, proposed_steering, rate_limit):
    """Limits the rate of change of the steering angle."""
    steering_change = proposed_steering - current_steering
    steering_change = np.clip(steering_change, -rate_limit, rate_limit)
    return current_steering + steering_change



def compute_steering_angles(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    # Compute the desired heading
    delta_y = lookahead_point[1] - position[1]
    delta_x = lookahead_point[0] - position[0]
    theta_desired = np.arctan2(delta_y, delta_x)
    
    # Wrap pose_theta and theta_desired to ensure they are within the correct range
    pose_theta = wrap_angle(pose_theta)
    theta_desired = wrap_angle(theta_desired)
    
    # Compute the error in heading
    theta_e = wrap_angle(theta_desired - pose_theta)
    
    # Compute front steering angle based on heading error
    front_steering_angle = np.arctan(2 * np.sin(theta_e) * wheelbase / lookahead_distance)
    
    # Assuming rear steering angle is some function of the front steering angle
    rear_steering_angle = 0.5 * front_steering_angle  # This is just an example and might not be accurate

    print(f"Desired heading: {theta_desired}, Heading error: {theta_e}, Steering angles: front={front_steering_angle}, rear={rear_steering_angle}")
    logging.debug(f"Steering angles: front={front_steering_angle}, rear={rear_steering_angle}")

    return front_steering_angle, rear_steering_angle




"""
Planner Helpers
"""
    
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        # Clear existing waypoints from the batch
        for wp in self.drawn_waypoints:
            e.batch.remove(wp)
        self.drawn_waypoints.clear()

        # Draw new waypoints as lines
        for i in range(len(self.waypoints) - 1):
            line_start = self.waypoints[i]
            line_end = self.waypoints[i + 1]
            self.drawn_waypoints.append(e.batch.add(2, pyglet.gl.GL_LINES, None, 
                ('v2f', [line_start[0], line_start[1], line_end[0], line_end[1]]),
                ('c3B', [255, 255, 255, 255, 255, 255])))  # White color for visibility


    def render_waypoints(self, env_renderer):
        """
        Update waypoints being drawn by EnvRenderer
        """
        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        # Scale the points for rendering
        scaled_points = 50. * points

        # Check if we have previously drawn waypoints, and if so, update them
        if not self.drawn_waypoints:
            for i in range(points.shape[0] - 1):
                line_start = scaled_points[i]
                line_end = scaled_points[i + 1]
                self.drawn_waypoints.append(env_renderer.batch.add(2, pyglet.gl.GL_LINES, None, 
                            ('v2f/stream', [line_start[0], line_start[1], line_end[0], line_end[1]]),
                            ('c3B/stream', [183, 193, 222, 183, 193, 222])))
        else:
            for i, line in enumerate(self.drawn_waypoints):
                if i < points.shape[0] - 1:
                    line.vertices = [scaled_points[i][0], scaled_points[i][1], scaled_points[i + 1][0], scaled_points[i + 1][1]]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None
    '''
    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle
    '''
    #new plan function for four wheel steering
    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0, 0.0

        front_steering_angle, rear_steering_angle = compute_steering_angles(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)  # You need to define this function based on your dynamics or you can use a simple model to compute the rear steering based on the front steering angle.
        speed = vgain * lookahead_point[2]

        return speed, front_steering_angle, rear_steering_angle



class FlippyPlanner:
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """
    def __init__(self, speed=1, flip_every=1, steer=2):
        self.speed = speed
        self.flip_every = flip_every
        self.counter = 0
        self.steer = steer
    
    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        if self.counter%self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


def generate_circle_vertices(center, radius, segments=20):
    """Generate vertices for a circle using its center and radius."""
    vertices = []
    for i in range(segments):
        theta = 2.0 * 3.1415926 * float(i) / float(segments)  # get the current angle
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)
        vertices.extend((center[0] + dx, center[1] + dy))
    return vertices


def draw_wheels_using_pyglet(env_renderer, car_pos, car_angle, CAR_LENGTH, CAR_WIDTH):
    WHEEL_RADIUS = CAR_LENGTH / 8
    # Define the scale factor for rendering based on the renderer's settings
    scale_factor = 50

    # Calculate relative wheel positions based on car position and orientation
    HALF_CAR_LEN = CAR_LENGTH / 2
    HALF_CAR_WIDTH = CAR_WIDTH / 2
    wheel_positions = [
        (car_pos[0] + HALF_CAR_LEN * np.cos(car_angle) - HALF_CAR_WIDTH * np.sin(car_angle),
         car_pos[1] + HALF_CAR_LEN * np.sin(car_angle) + HALF_CAR_WIDTH * np.cos(car_angle)),
        (car_pos[0] + HALF_CAR_LEN * np.cos(car_angle) + HALF_CAR_WIDTH * np.sin(car_angle),
         car_pos[1] + HALF_CAR_LEN * np.sin(car_angle) - HALF_CAR_WIDTH * np.cos(car_angle)),
        (car_pos[0] - HALF_CAR_LEN * np.cos(car_angle) - HALF_CAR_WIDTH * np.sin(car_angle),
         car_pos[1] - HALF_CAR_LEN * np.sin(car_angle) + HALF_CAR_WIDTH * np.cos(car_angle)),
        (car_pos[0] - HALF_CAR_LEN * np.cos(car_angle) + HALF_CAR_WIDTH * np.sin(car_angle),
         car_pos[1] - HALF_CAR_LEN * np.sin(car_angle) - HALF_CAR_WIDTH * np.cos(car_angle))
    ]

    # Clear previous wheel drawings
    if hasattr(env_renderer, 'wheel_drawings'):
        for drawing in env_renderer.wheel_drawings:
            drawing.delete()
    env_renderer.wheel_drawings = []

    # Draw the wheels at the new positions
    for wheel_pos in wheel_positions:
        scaled_wheel_pos = (scale_factor * wheel_pos[0], scale_factor * wheel_pos[1])
        wheel_vertices = generate_circle_vertices(scaled_wheel_pos, scale_factor * WHEEL_RADIUS)
        wheel_color = (100, 100, 100) * (len(wheel_vertices) // 2)  # Gray color for wheels
        wheel_drawing = env_renderer.batch.add(len(wheel_vertices) // 2, pyglet.gl.GL_TRIANGLE_FAN, None,
                                               ('v2f/static', wheel_vertices),
                                               ('c3B/static', wheel_color))
        env_renderer.wheel_drawings.append(wheel_drawing)




def main():
    """
    main entry point
    """

    # Parameters for speed control
    MAX_SPEED = 10.0  # Maximum speed (m/s)
    MIN_SPEED = 2.0   # Minimum speed (m/s)
    MAX_STEERING_ANGLE = 0.35  # Maximum steering angle (rad) for both front and rear


    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}#0.90338203837889}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)


    # Define initial values before the while loop
    initial_front_steering_angle = 0.0  # Initialize with the starting front steering angle
    initial_rear_steering_angle = 0.0   # Initialize with the starting rear steering angle
    MAX_STEERING_RATE = 100  # Maximum change per timestep, adjust as needed

    prev_front_steering_angle = initial_front_steering_angle
    prev_rear_steering_angle = initial_rear_steering_angle

    # Define the rate limit function
    def rate_limit_steering(prev_angle, new_angle, max_rate):
        # Calculate the difference between the new angle and the previous angle
        angle_diff = new_angle - prev_angle
        
        # If the difference is greater than the max rate, limit the change
        if angle_diff > max_rate:
            return prev_angle + max_rate
        elif angle_diff < -max_rate:
            return prev_angle - max_rate
        else:
            return new_angle



    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        car_pos = [obs['poses_x'][0], obs['poses_y'][0]]
        car_angle = obs['poses_theta'][0]
        draw_wheels_using_pyglet(env_renderer, car_pos, car_angle, CAR_LENGTH, CAR_WIDTH)  # Define this function

        planner.render_waypoints(env_renderer)


    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    env.add_render_callback(render_callback)
    

    initial_action = np.array([[0.0, 0.0, 1.0]])  # This is just a sample, yours might be different
    print(initial_action.shape)  # <-- Add this line here
    obs, step_reward, done, info = env.reset(initial_action)
    env.render()

    laptime = 0.0
    start = time.time()

    # Main loop
    while not done:
    # Wrap the current pose to [-pi, pi]
        obs['poses_theta'][0] = wrap_angle(obs['poses_theta'][0])

        # Plan the path and get control commands
        speed, front_steering_angle, rear_steering_angle = planner.plan(
            obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain']
        )

        # Apply rate limiting to the steering commands (if implemented)
        front_steering_angle = rate_limit_steering(
            prev_front_steering_angle, front_steering_angle, MAX_STEERING_RATE
        )
        rear_steering_angle = rate_limit_steering(
            prev_rear_steering_angle, rear_steering_angle, MAX_STEERING_RATE
        )

        # Proportional speed control based on steering
        speed = proportional_speed_control(
            front_steering_angle, rear_steering_angle, MAX_SPEED, MIN_SPEED, MAX_STEERING_ANGLE
        )

        # Ensure steering commands are within vehicle limits
        front_steering_angle = scale_steering(front_steering_angle, MAX_STEERING_ANGLE)
        rear_steering_angle = scale_steering(rear_steering_angle, MAX_STEERING_ANGLE)

        # Execute simulation step
        obs, step_reward, done, info = env.step(np.array([[front_steering_angle, rear_steering_angle, speed]]))

        # Update previous steering angles for next iteration's rate limiting
        prev_front_steering_angle = front_steering_angle
        prev_rear_steering_angle = rear_steering_angle

        # Render the environment
        env.render()




        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()

