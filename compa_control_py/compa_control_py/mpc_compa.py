# mpc_compa.py
import numpy as np
from pydrake.solvers import (
    MathematicalProgram, 
    OsqpSolver
)

class CompaMPC:
    def __init__(self, r_wheel, b_wheel, a_wheel, 
                 N=20, dt=0.01, nx: int = 5, nu: int = 5):
        """
        N : horizon length (timesteps)
        dt: sampling period (s) = 1/hz
        """
        self.r_wheel, self.b_wheel, self.a_wheel = r_wheel, b_wheel, a_wheel
        self.N = N
        self.dt = dt

        # State: [x, y, roll, pitch, yaw_turret, yaw_base_w]; 
        # Input: [vx_base, vy_base, roll_dot_gimbal, pitch_dot_gimbal, yaw_dot_turret]
        self.nx = nx
        self.nu = nu

        # Cost weights
        self.Q = np.diag([5.0, 5.0, 3.0, 2.0, 2.0])    # tracking       5x5
        self.R = np.diag([0.1, 0.1, 0.2, 0.5, 0.5])    # control effort 5x5 
        self.P = np.diag([8.0, 8.0, 10.0, 10.0, 10.0]) # terminal       5x5

        # System matrices (simple integrator model)
        self.A = np.eye(self.nx)                    # 5x5
        self.B = self.dt * np.eye(self.nx, self.nu) # 5x5

        # - - - - - - - - - - - - L      R      Roll  Pitch Yaw  #
        self.qdot_min = np.array([-8.0, -8.0, -1.0, -1.0, -2.0])
        self.qdot_max = np.array([+8.0, +8.0, +1.0, +1.0, +2.0])

        # Keep last u delta_u cost later
        self.u_last = np.zeros(self.nu)
        self.solver = OsqpSolver()

    def solve(self, x0, x_ref_seq):
        """
        x0: (6,) current state [x, y, roll, pitch, yaw, (yaw_base_w: used for Jacobian)]
        x_ref_seq: (N+1, 5) reference over horizon (including terminal ref)
        Returns u0 (2,) : first control to apply.
        """
        assert x_ref_seq.shape == (self.N + 1, self.nx)

        prog = MathematicalProgram()

        M = self.build_J(x0[-1], self.r_wheel, self.b_wheel, self.a_wheel)

        # Decision variables
        X = prog.NewContinuousVariables(self.nx, self.N + 1, "x")
        U = prog.NewContinuousVariables(self.nu, self.N, "u")

        # Dynamics: x_{k+1} = A*x_k + B*u_k
        for k in range(self.N):
            xk  = X[:, k]
            xk1 = X[:, k + 1]
            uk  = U[:, k]
            expr = xk1 - (self.A @ xk + self.B @ uk)
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.nx))

            # max_roll  = np.deg2rad(20.0)
            # max_pitch = np.deg2rad(20.0)
            # prog.AddBoundingBoxConstraint(
            #     [-np.inf, -np.inf, -np.inf, -max_roll, -max_pitch],
            #     [ np.inf,  np.inf,  np.inf,  max_roll,  max_pitch],
            #     X[:, k]
            # )
            prog.AddLinearConstraint(M, self.qdot_min, self.qdot_max, uk)

        # Initial condition: x_0 = x0
        prog.AddLinearEqualityConstraint(X[:, 0] - x0[:5], np.zeros(self.nx))

        # Stage costs
        for k in range(self.N):
            xk = X[:, k]
            uk = U[:, k]
            x_ref_k = x_ref_seq[k, :]

            # Tracking: (xk - x_ref_k)^T Q (xk - x_ref_k)
            prog.AddQuadraticErrorCost(self.Q, x_ref_k, xk)

            # Control effort: u_k^T R u_k
            prog.AddQuadraticCost(uk @ self.R @ uk)

            # TODO: later add delta_u cost + terrain/traversability cost

        # Terminal cost
        xN = X[:, self.N]
        x_ref_N = x_ref_seq[self.N, :]
        prog.AddQuadraticErrorCost(self.P, x_ref_N, xN)

        result = self.solver.Solve(prog)
        if not result.is_success():
            return np.zeros(self.nu)

        U_opt = result.GetSolution(U) # shape (nu, N)
        u0 = U_opt[:, 0]
        self.u_last = u0.copy()
        return u0
    
    def build_J(self, yaw, r_wheel, b_wheel, a_wheel):
        ''' Derived Jacobian based on dynamics - returns angular velocities for:
                1. right_wheel
                2. left_wheel
                3. turret 
                4. roll
                5. pitch
        '''
        r_w, b, a = r_wheel, b_wheel, a_wheel
        c, s = np.cos(yaw), np.sin(yaw)

        J = np.array([
            [r_w/2 * (c - s*b/a), r_w/2 * (c + s*b/a), 0, 0, 0],
            [r_w/2 * (s + c*b/a), r_w/2 * (s - c*b/a), 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [r_w/(2*a), -r_w/(2*a), 0, 0, 1],
        ])

        return np.linalg.inv(J)
