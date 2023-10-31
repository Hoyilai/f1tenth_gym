import numpy as np
from dynamic_models import vehicle_dynamics_ks_4w

# Define initial state
# x = [s_x, s_y, v_x, v_y, ψ, r, δ, v_delta]
x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Define control inputs for 4-wheel steering
# u = [delta_dot_fl, delta_dot_fr, delta_dot_rl, delta_dot_rr, a]
u_init = np.array([0.1, 0.1, 0.1, 0.1, 0.0])

# Define vehicle parameters
parameters = {
    "mu": 1.0489,
    "C_Sf": 1.1043,
    "C_Sr": 1.1584,
    "lf": 0.125,
    "lr": 0.125,
    "h": 0.02,
    "m": 3.74,
    "I": 0.016,
    "s_min": -0.01,
    "s_max": 0.01,
    "sv_min": -0.01,
    "sv_max": 0.01,
    "v_switch": 0.01,
    "a_max": 0.01,
    "v_min": -2.0,
    "v_max": 2.0,
    "delta_max": 0.5,
    "delta_min" : -0.5
}


'''

x_dot: Change in x position (0.0) - The vehicle is not moving in the x-direction.
y_dot: Change in y position (0.0) - The vehicle is not moving in the y-direction.
delta_fl_dot: Change in steering angle of front left wheel (0.1) - The front left wheel is turning at a rate of 0.1 rad/s.
delta_fr_dot: Change in steering angle of front right wheel (0.1) - The front right wheel is turning at a rate of 0.1 rad/s.
delta_rl_dot: Change in steering angle of rear left wheel (0.1) - The rear left wheel is turning at a rate of 0.1 rad/s.
delta_rr_dot: Change in steering angle of rear right wheel (0.1) - The rear right wheel is turning at a rate of 0.1 rad/s.
v_dot: Change in velocity (0.0) - The vehicle's speed is not changing.
psi_dot: Change in orientation (0.0) - The vehicle's orientation is not changing.

'''



# Time (not really used in the model but required by its definition)
t = 0

# Call the vehicle dynamics function
x_dot = vehicle_dynamics_ks_4w(t, x0, u_init, parameters)

# Print the result
print(x_dot)
