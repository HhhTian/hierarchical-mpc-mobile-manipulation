# controller/planner.py

import numpy as np

def generate_base_square_wave(t, amplitude=2.0, period=8.0):
    """
    Generate square wave trajectory of base:
    - 0 <= t < period: to amplitude (peak)
    - period <= t < 2*period: back to 0 (valley)
    """
    if t % (2 * period) < period:
        return amplitude
    else:
        return 0.0

def reference_trajectory(t, horizon=10, dt=0.1):
    """
    Generate base and EE reference trajectories for the next N steps (for MPC)
    base: follow square wave
    EE: fixed at (2.0, 0.0, z)
    return：
        r_base_seq: N x 2
        r_ee_seq:   N x 3
    """
    r_base_seq = []
    r_ee_seq = []

    for i in range(horizon):
        t_future = t + i * dt
        base_x = 2.0
        # base_x = generate_base_square_wave(t_future)
        base_y = 0.0
        r_base_seq.append([base_x, base_y])
        # z = 1.2 + 0.1 * np.sin(0.2 * t_future)
        z = 1.2
        r_ee_seq.append([2.0, 0.0, z])  # EE target fixed

    return np.array(r_base_seq), np.array(r_ee_seq)