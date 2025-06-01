# simulation/simulate.py

import numpy as np
import matplotlib.pyplot as plt
from controller.model import HTMPCModel
from controller.mpc import STMPC_EE_Tracking
from controller.planner import reference_trajectory
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# initialize
dt = 0.1
horizon = 100
steps = 400  # simu time = steps * dt

# system model
model = HTMPCModel(dt=dt)
controller = STMPC_EE_Tracking(model, horizon=horizon, dt=dt)

# initial state：x_b, y_b, q_ext, q_lift, vx, vy, v_ext, v_lift
x = np.zeros(model.n + model.m)
x[0] = 0.0  # base_x
x[1] = 0.0  # base_y
x[2] = 1.0  # arm extension
x[3] = 1.2  # arm lift = EE 

x_history = [x.copy()]
ee_history = [model.ee_position(x[:model.n])]
ref_history = []

r_base_seq, r_ee_seq = reference_trajectory(0.0, horizon=horizon, dt=dt)
ref_history.append(r_ee_seq[0]) 

v_history = [x[model.n:].copy()]  # init v，model.n = 4

# simulation loop
for step in range(steps):
    t = step * dt

    # Generate EE target sequence
    r_base_seq, r_ee_seq = reference_trajectory(t, horizon=horizon, dt=dt)

    # call MPC for u0
    u0 = controller.solve(x, r_ee_seq)

    # state update 1 step
    x = model.step(x, u0)

    # record
    x_history.append(x.copy())
    ee_history.append(model.ee_position(x[:model.n]))
    ref_history.append(r_ee_seq[0])  # current ee target
    v_history.append(x[model.n:].copy())

# to np.array
x_history = np.array(x_history)
ee_history = np.array(ee_history)
ref_history = np.array(ref_history)
v_history = np.array(v_history)

# debug
print("EE pos:", ee_history[-1])
print("EE target:", ref_history[-1])
print("r_ee_seq shape:", r_ee_seq.shape)
print("ref_history shape:", ref_history.shape)
print("ref x:", ref_history[:, 0])
print("ref y:", ref_history[:, 1])
print("ref z:", ref_history[:, 2])

# Plot velocity over time
plt.figure(figsize=(10, 6))
plt.plot(v_history)
plt.title("Velocity History")
plt.xlabel("Timestep")
plt.ylabel("Velocity (m/s or rad/s)")
plt.legend(["vx", "vy", "v_ext", "v_lift"])
plt.grid()
plt.tight_layout()
plt.savefig("results/velocity_history.png")

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ee_history[:, 0], ee_history[:, 1], ee_history[:, 2], label='EE actual')
ax.plot(ref_history[:, 0], ref_history[:, 1], ref_history[:, 2], '--', label='EE target')
ax.plot(x_history[:, 0], x_history[:, 1], np.zeros_like(x_history[:, 0]), label='Base path')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.title("3D Trajectory of EE and Base")
plt.tight_layout()
plt.savefig("results/trajectory_3d.png")  # save image

# error plot
assert ee_history.shape == ref_history.shape, "Shape mismatch between EE and target histories"
time = np.arange(len(ee_history)) * dt
errors = ee_history - ref_history

plt.figure(figsize=(10, 6))
plt.plot(time, errors[:, 0], label='EE x error')
plt.plot(time, errors[:, 1], label='EE y error')
plt.plot(time, errors[:, 2], label='EE z error')
plt.xlabel("Time [s]")
plt.ylabel("Tracking Error [m]")
plt.title("EE Tracking Error Over Time")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("results/ee_error.png")

# animation
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
line1, = ax.plot([], [], [], label='EE actual')
line2, = ax.plot([], [], [], label='EE target', linestyle='--')
line3, = ax.plot([], [], [], label='Base path', color='green')
ax.set_xlim(0, 4)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 2)

def update(frame):
    line1.set_data(ee_history[:frame, 0], ee_history[:frame, 1])
    line1.set_3d_properties(ee_history[:frame, 2])
    line2.set_data(ref_history[:frame, 0], ref_history[:frame, 1])
    line2.set_3d_properties(ref_history[:frame, 2])
    line3.set_data(x_history[:frame, 0], x_history[:frame, 1])
    line3.set_3d_properties(np.zeros_like(x_history[:frame, 0]))  # base z=0
    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, frames=len(ee_history), blit=True)
ani.save("results/ee_animation.gif", fps=10)

print("✅ image saved as: results/velocity_history.png, trajectory_3d.png, ee_error.png, ee_animation.gif")