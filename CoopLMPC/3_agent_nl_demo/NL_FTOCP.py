import casadi as ca
import numpy as np
import numpy.linalg as la

import pdb

class NL_FTOCP(object):
    def __init__(self, N, agent):
        self.agent = agent

        self.n_x = self.agent.n_x
        self.n_u = self.agent.n_u

        self.l_r = self.agent.l_r
        self.l_f = self.agent.l_f

        self.dt = self.agent.dt

        self.N = N

        self.F, self.b = self.agent.get_state_constraints()
        self.H, self.g = self.agent.get_input_constraints()

        self.x_preds = np.zeros((self.n_x, self.N+1, 1))
        self.u_preds = np.zeros((self.n_u, self.N, 1))

        self.cost = []

    def solve(self, x_0, abs_t, x_f, x_guess=None, u_guess=None, expl_constraints=None, verbose=False):
        opti = ca.Opti()

        x = opti.variable(self.n_x, self.N+1)
        u = opti.variable(self.n_u, self.N)
        slack = opti.variable(self.n_x)

        if x_guess is not None:
            opti.set_initial(x, x_guess)
        if u_guess is not None:
            opti.set_initial(u, u_guess)
        opti.set_initial(slack, np.zeros(self.n_x))

        opti.subject_to(x[:,0] == np.squeeze(x_0))
        opti.subject_to(opti.bounded(-0.5*self.dt, u[0,0]-self.u_preds[0,0,-1], 0.5*self.dt))
        opti.subject_to(opti.bounded(-3.0*self.dt, u[1,0]-self.u_preds[0,0,-1], 3.0*self.dt))

        stage_cost = 0
        for i in range(self.N):
            stage_cost = stage_cost + 1

            beta = ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)
            opti.subject_to(x[0,i+1] == x[0,i] + self.dt*x[3,i]*ca.cos(x[2,i] + beta))
            opti.subject_to(x[1,i+1] == x[1,i] + self.dt*x[3,i]*ca.sin(x[2,i] + beta))
            opti.subject_to(x[2,i+1] == x[2,i] + self.dt*x[3,i]*ca.sin(beta))
            opti.subject_to(x[3,i+1] == x[3,i] + self.dt*u[1,i])

            if self.F is not None:
                opti.subject_to(ca.mtimes(self.F, x[:,i]) <= self.b)
            if self.H is not None:
                opti.subject_to(ca.mtimes(self.H, u[:,i]) <= self.g)

            if expl_constraints is not None:
                V = expl_constraints[i][0]
                w = expl_constraints[i][1]
                opti.subject_to(ca.mtimes(V, x[:2,i]) <= w)

            if i < self.N-1:
                opti.subject_to(opti.bounded(-0.5*self.dt, u[0,i+1]-u[0,i], 0.5*self.dt))
                opti.subject_to(opti.bounded(-3.0*self.dt, u[1,i+1]-u[1,i], 3.0*self.dt))

        opti.subject_to(x[:,self.N] - x_f == slack)
        # opti.subject_to(x[:,self.N] == x_f)
        # opti.subject_to(slack >= 0)

        slack_cost = 1e6*ca.sumsqr(slack)
        total_cost = stage_cost + slack_cost
        opti.minimize(total_cost)

        solver_opts = {
            "mu_strategy" : "adaptive",
            "mu_init" : 1e-5,
            "mu_min" : 1e-15,
            "barrier_tol_factor" : 1
            }
        # solver_opts = {}
        plugin_opts = {"verbose" : False, "print_time" : False, "print_out" : False}
        opti.solver('ipopt', plugin_opts, solver_opts)

        sol = opti.solve()

        slack_val = sol.value(slack)
        if la.norm(slack_val) > 1e-8:
            print('Warning! Solved, but with slack norm of %g is greater than 1e-8!' % la.norm(slack_val))

        if sol.stats()['success'] and la.norm(slack_val) <= 1e-8:
            feasible = True

            x_pred = sol.value(x)
            u_pred = sol.value(u)
            sol_cost = sol.value(stage_cost)
        else:
            print(sol.stats()['return_status'])
            # print(opti.debug.show_infeasibilities())
            feasible = False
            x_pred = None
            u_pred = None
            sol_cost = None

            # pdb.set_trace()

        return x_pred, u_pred, sol_cost

    def update_predictions(self, x_pred, u_pred):
        self.x_preds = np.append(self.x_preds, np.expand_dims(x_pred, axis=-1), axis=-1)
        self.u_preds = np.append(self.u_preds, np.expand_dims(u_pred, axis=-1), axis=-1)

    def update_horizon_length(self, N):
        self.N = N
