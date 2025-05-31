# controller/model.py

import numpy as np
import casadi as ca

class HTMPCModel:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.n = 4  # q: x_b, y_b, q_ext, q_lift
        self.m = 4  # v: vx, vy, v_ext, v_lift

    def A(self):
        G = np.eye(self.n)  # Simplify G(q) = I for holonomic system with omni-wheel
        return np.block([
            [np.zeros((self.n, self.n)), G],
            [np.zeros((self.m, self.n)), np.zeros((self.m, self.m))]
        ])

    def B(self):
        return np.block([
            [np.zeros((self.n, self.m))],
            [np.eye(self.m)]
        ])

    def step(self, x, u):
        # x = [q, v], u = [a]
        A = self.A()
        B = self.B()
        dx = A @ x + B @ u
        return x + self.dt * dx

    def ee_position(self, x):
        """
        Extract the position of EE (x, y, z) from state vector 
        X = [x_b, y_b, EE_horizontal_extension, EE_vertical_lift]
        Assuming that the arm can extend forward + lift vertically
        """
        x_b, y_b, q_ext, q_lift = x[0], x[1], x[2], x[3]
        x_ee = x_b + q_ext
        y_ee = y_b
        z_ee = q_lift
        return np.array([x_ee, y_ee, z_ee])

    
    def ee_position_casadi(self, q):
        """
        for calculating EE positions in MPC controllers
        q: casadi MX (n,)
        Forward kinematics using DH parameters or analytical formulas
        """
        x_b = q[0]
        y_b = q[1]
        q_ext = q[2]
        q_lift = q[3]

        x_ee = x_b + q_ext
        y_ee = y_b
        z_ee = q_lift

        return ca.vertcat(x_ee, y_ee, z_ee)
    
    """ for Differential Wheeled Mobile:
    def G(self, q):
        # take θ from base
        theta = q[2]
        return np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])
    """