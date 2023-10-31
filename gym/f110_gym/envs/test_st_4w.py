import numpy as np
from dynamic_models import vehicle_dynamics_st_4w

# Define vehicle parameters
mu = 1.0489         # Friction coefficient between the tire and the road.
C_Sf = 1.1043       # Cornering stiffness of the front tires.
C_Sr = 1.1584       # Cornering stiffness of the rear tires.
lf = 0.125          # Distance from the center of gravity of the vehicle to the front axle.
lr = 0.125          # Distance from the center of gravity of the vehicle to the rear axle.
h = 0.02            # Height of the vehicle's center of gravity above the ground.
m = 3.74            # Mass of the vehicle.
I = 0.016           # Moment of inertia of the vehicle about the vertical axis.
s_min = -0.01       # Minimum slip angle.
s_max = 0.01        # Maximum slip angle.
sv_min = -0.01      # Minimum slip velocity.
sv_max = 0.01       # Maximum slip velocity.
v_switch = 0.01     # Velocity threshold below which the kinematic model is used.
a_max = 0.01        # Maximum allowed acceleration.
v_min = -2.0        # Minimum allowable velocity (negative for reverse).
v_max = 2.0         # Maximum allowable velocity.


# Initial state [x, y, psi, v_x, v_y, dot_psi]
x0 = np.array([0, 0, 0, 1, 0, 0])  # Start with a forward velocity of 1 m/s

# Control inputs: [acceleration, front steering angle, rear steering angle]
u_init = np.array([0, np.radians(10), np.radians(-5)])  # Front steering of 10 degrees and rear steering of -5 degrees

# Time (Not used in this static model, but can be any value)
t = 0

# Call the dynamics function
x_dot = vehicle_dynamics_st_4w(x0, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)


print(x_dot)
