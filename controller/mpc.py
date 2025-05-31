# controller/mpc.py

import casadi as ca
import numpy as np

class STMPC_EE_Tracking:
    def __init__(self, model, horizon=10, dt=0.1):
        self.model = model
        self.horizon = horizon
        self.dt = dt
        
        self.nx = model.n + model.m  # state dimension：q + v
        self.nu = model.m            # control dimension: acceleration
        
        self._build_controller()
        
        self.u_prev = np.zeros((self.nu, self.N))

    def _build_controller(self):
        N = self.horizon
        nx = self.nx
        nu = self.nu

        # CasADi variable
        x = ca.MX.sym('x', nx)  # initial state
        U = ca.MX.sym('U', nu, N)  # input control sequence
        r_seq = ca.MX.sym('r_seq', 3, N)  # EE reference trajectory

        X = [x]
        cost = 0
        g = []
        lbg = []
        ubg = []
        x_seq = [x]  # collect intermediate states
        g_x = []
        lbx_x = []
        ubx_x = []
        
        A = self.model.A()
        B = self.model.B()

        for k in range(N):
            u_k = U[:, k]
            x_k = X[-1]

            # state update：x_{k+1} = x_k + dt * (A x + B u)
            x_next = x_k + self.dt * (A @ x_k + B @ u_k)
            X.append(x_next)

            # EE tracking error
            ee_pos = self.model.ee_position_casadi(x_k[:self.model.n])  # dh/forward kinematics
            e_k = ee_pos - r_seq[:, k]
            cost += ca.dot(e_k, e_k)  # tracking cost
            Q = ca.diag(ca.DM([1.0, 20.0, 1.0]))
            cost += ca.mtimes([e_k.T, Q, e_k])

            
            v_k = x_k[self.model.n:]  #  vx, vy, v_ext, v_lift
            cost += 1 * ca.dot(v_k, v_k)  # L2 penalty on speed  #============
            
            # Add input constrains: -1 <= u_k <= 1
            g.append(u_k)
            lbg += [-1.0] * nu
            ubg += [ 1.0] * nu
            
            # speed constrain
            g.append(v_k)
            v_max = 1.0  #===============# 
            lbg += [-v_max] * self.model.m
            ubg += [ v_max] * self.model.m
            
            # # state range constrain [-5, 5]
            # x_next_pos = x_next[:self.model.n]  # only position part
            # g_x.append(x_next_pos)
            # lbx_x += [-5.0] * self.model.n
            # ubx_x += [5.0] * self.model.n

        # build nonlinear programming & solve
        opt_vars = ca.vec(U)    # Flatten decision variables
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*g, *g_x),
            'p': ca.vertcat(x, ca.vec(r_seq))
        }

        solver_opts = {'print_time': 0, 'ipopt.print_level': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)

        # Store for solve()
        self.lbg = lbg
        self.ubg = ubg
        self.nx = nx
        self.nu = nu
        self.N = N
        # self.lbg = lbg + lbx_x
        # self.ubg = ubg + ubx_x

    def solve(self, x0, ee_ref_seq):
        """
        Solve STMPC problem.
        
        Input:
          x0: current state (nx,)
          ee_ref_seq: EE reference trajectory N x 3

        Output:
          optimal first control term u0: (nu,)
        """
        p_val = np.concatenate([x0, ee_ref_seq.flatten()])
        u_init = np.hstack([self.u_prev[:, 1:], self.u_prev[:, -1:]]).flatten()
        # u_init = np.zeros((self.nu, self.N)).flatten()

        sol = self.solver(x0=u_init, p=p_val, lbg=self.lbg, ubg=self.ubg)
        u_opt = sol['x'].full().reshape(self.nu, self.N)
        self.u_prev = u_opt

        return u_opt[:, 0]  # only u_0





"""
建议优化点
1.动态更新 A/B: 如果你的系统是非线性的，而 A/B 是线性化结果，应该在 _build_controller() 外部以当前状态为中心动态更新 A 和 B（否则可能不收敛或性能较差）。

2.引入速度/加速度惩罚项（正则项）：

cost += ca.dot(u_k, u_k) * weight
否则可能导致控制发散或震荡。

3.加约束（可选）： 如：

控制输入范围 u_min <= u <= u_max
状态约束（避免撞墙等）

4.CasADi 表达式缓存或优化： 如果 model.ee_position_casadi() 内部每次都重新构建表达式，效率会比较低。考虑缓存表达式图。

5.代码可视化调试： 添加 debug flag，输出每步 EE 的预测轨迹和误差，便于可视化对齐问题。
"""


"""
你可以在 STMPC_EE_Tracking 中添加缓存，记录上一次的 
U, 并在下一次调用 solve() 时用它作为 x0 的初始值：

class STMPC_EE_Tracking:
    def __init__(...):
        ...
        self.last_u = np.zeros((self.nu, self.horizon))

    def solve(self, x0, ee_ref_seq):
        p_val = np.concatenate([x0, ee_ref_seq.flatten()])
        sol = self.solver(x0=self.last_u.flatten(), p=p_val)
        u_opt = sol['x'].full().reshape(self.nu, self.horizon)
        self.last_u = u_opt  # store for next round
        return u_opt[:, 0]
这样可以实现 warm start, 有助于加速收敛, 也更真实地模拟工业控制场景。
"""