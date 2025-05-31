from controller.model import HTMPCModel
from controller.planner import *
from controller.mpc import STMPC_EE_Tracking
import numpy as np

model = HTMPCModel(dt=0.1)
x0 = np.zeros(8)
u = np.array([1.0, 0.0, 0.5, 1.0])  # x acc + x arm ext

x1 = model.step(x0, u)

print("x1 =", x1)
print("EE position:", model.ee_position(x1))

# current time
t = 7.6  

# generate reference trajectory
r_base_seq, r_ee_seq = reference_trajectory(t, horizon=10, dt=0.1)

print("Sequence of base reference trajectories:\n", r_base_seq)
print("End-effector reference trajectory sequence:\n", r_ee_seq)


model = HTMPCModel(dt=0.1)
mpc = STMPC_EE_Tracking(model)

x0 = np.zeros(8)
t = 0.0
_, r_ee_seq = reference_trajectory(t, horizon=10, dt=0.1)

u0 = mpc.solve(x0, r_ee_seq)
print("Optimal control u0:", u0)