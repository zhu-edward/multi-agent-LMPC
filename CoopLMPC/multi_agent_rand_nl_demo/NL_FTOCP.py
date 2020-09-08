import casadi as ca
import numpy as np
import numpy.linalg as la

import pdb

class NL_FTOCP(object):
    def __init__(self, agent):
        self.agent = agent

        self.n_x = self.agent.n_x
        self.n_u = self.agent.n_u

        self.l_r = self.agent.l_r
        self.l_f = self.agent.l_f

        self.dt = self.agent.dt

        self.F, self.b = self.agent.get_state_constraints()
        self.H, self.g = self.agent.get_input_constraints()

        self.x_lim = self.agent.x_lim
        self.y_lim = self.agent.y_lim
        self.psi_lim = self.agent.psi_lim
        self.v_lim = self.agent.v_lim

        self.a_lim = self.agent.a_lim
        self.df_lim = self.agent.df_lim
        self.da_lim = self.agent.da_lim
        self.ddf_lim = self.agent.ddf_lim

        self.state_lb = []
        self.state_ub = []
        if self.x_lim is not None:
            self.state_lb += [self.x_lim[0]]
            self.state_ub += [self.x_lim[1]]
        else:
            self.state_lb += [-1000.0]
            self.state_ub += [1000.0]
        if self.y_lim is not None:
            self.state_lb += [self.y_lim[0]]
            self.state_ub += [self.y_lim[1]]
        else:
            self.state_lb += [-1000.0]
            self.state_ub += [1000.0]
        if self.psi_lim is not None:
            self.state_lb += [self.psi_lim[0]]
            self.state_ub += [self.psi_lim[1]]
        else:
            self.state_lb += [-1000.0]
            self.state_ub += [1000.0]
        if self.v_lim is not None:
            self.state_lb += [self.v_lim[0]]
            self.state_ub += [self.v_lim[1]]
        else:
            self.state_lb += [-1000.0]
            self.state_ub += [1000.0]

        self.input_lb = []
        self.input_ub = []
        if self.df_lim is not None:
            self.input_lb += [self.df_lim[0]]
            self.input_ub += [self.df_lim[1]]
        else:
            self.input_lb += [-1000.0]
            self.input_ub += [1000.0]
        if self.a_lim is not None:
            self.input_lb += [self.a_lim[0]]
            self.input_ub += [self.a_lim[1]]
        else:
            self.input_lb += [-1000.0]
            self.input_ub += [1000.0]

        self.input_rate_lb = []
        self.input_rate_ub = []
        if self.ddf_lim is not None:
            self.input_rate_lb += [self.ddf_lim[0]]
            self.input_rate_ub += [self.ddf_lim[1]]
        else:
            self.input_rate_lb += [-1000.0]
            self.input_rate_ub += [1000.0]
        if self.da_lim is not None:
            self.input_rate_lb += [self.da_lim[0]]
            self.input_rate_ub += [self.da_lim[1]]
        else:
            self.input_rate_lb += [-1000.0]
            self.input_rate_ub += [1000.0]

        self.cost = None
        self.opti = None
        self.x_s = None
        self.x_f = None
        self.x = None
        self.u = None

        self.opti0 = None
        self.cost0 = None
        self.x_s0 = None
        self.x_f0 = None
        self.x0 = None
        self.u0 = None

    def solve_opti(self, abs_t, x_0, x_ss, N, last_u, x_guess=None, u_guess=None, expl_constraints=None, verbose=False):
        opti = ca.Opti()

        x = opti.variable(self.n_x, N+1)
        u = opti.variable(self.n_u, N)
        slack = opti.variable(self.n_x)

        if x_guess is not None:
            opti.set_initial(x, x_guess)
        if u_guess is not None:
            opti.set_initial(u, u_guess)
        opti.set_initial(slack, np.zeros(self.n_x))

        da_lim = self.agent.da_lim
        ddf_lim = self.agent.ddf_lim

        opti.subject_to(x[:,0] == np.squeeze(x_0))
        opti.subject_to(opti.bounded(ddf_lim[0]*self.dt, u[0,0]-last_u[0], ddf_lim[1]*self.dt))
        opti.subject_to(opti.bounded(da_lim[0]*self.dt, u[1,0]-last_u[1], da_lim[1]*self.dt))

        stage_cost = 0
        for i in range(N):
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
                opti.subject_to(ca.mtimes(V, x[:2,i]) + w <= np.zeros(len(w)))

            if i < N-1:
                opti.subject_to(opti.bounded(ddf_lim[0]*self.dt, u[0,i+1]-u[0,i], ddf_lim[1]*self.dt))
                opti.subject_to(opti.bounded(da_lim[0]*self.dt, u[1,i+1]-u[1,i], da_lim[1]*self.dt))

        opti.subject_to(x[:,N] - x_ss == slack)

        slack_cost = 1e6*ca.sumsqr(slack)
        total_cost = stage_cost + slack_cost
        opti.minimize(total_cost)

        solver_opts = {
            "mu_strategy" : "adaptive",
            "mu_init" : 1e-5,
            "mu_min" : 1e-15,
            "barrier_tol_factor" : 1,
            "print_level" : 0,
            "linear_solver" : "ma27"
            }
        # solver_opts = {}
        plugin_opts = {"verbose" : False, "print_time" : False, "print_out" : False}
        opti.solver('ipopt', plugin_opts, solver_opts)

        sol = opti.solve()

        slack_val = sol.value(slack)
        if la.norm(slack_val) > 1e-6:
            print('Warning! Solved, but with slack norm of %g is greater than 1e-8!' % la.norm(slack_val))

        if sol.stats()['success'] and la.norm(slack_val) <= 1e-6:
            print('Solve success, with slack norm of %g!' % la.norm(slack_val))
            feasible = True

            x_pred = sol.value(x)
            u_pred = sol.value(u)
            sol_cost = sol.value(stage_cost)
        else:
            # print(sol.stats()['return_status'])
            # print(opti.debug.show_infeasibilities())
            feasible = False
            x_pred = None
            u_pred = None
            sol_cost = None

            # pdb.set_trace()

        return x_pred, u_pred, sol_cost

    def solve_opti0(self, abs_t, x_0, x_ss, N, x_guess=None, u_guess=None, expl_constraints=None, verbose=False):
        opti = ca.Opti()

        x = opti.variable(self.n_x, N+1)
        u = opti.variable(self.n_u, N)
        slack = opti.variable(self.n_x)

        if x_guess is not None:
            opti.set_initial(x, x_guess)
        if u_guess is not None:
            opti.set_initial(u, u_guess)
        opti.set_initial(slack, np.zeros(self.n_x))

        da_lim = self.agent.da_lim
        ddf_lim = self.agent.ddf_lim

        opti.subject_to(x[:,0] == np.squeeze(x_0))

        stage_cost = 0
        for i in range(N):
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
                opti.subject_to(ca.mtimes(V, x[:2,i]) + w <= np.zeros(len(w)))

            if i < N-1:
                opti.subject_to(opti.bounded(ddf_lim[0]*self.dt, u[0,i+1]-u[0,i], ddf_lim[1]*self.dt))
                opti.subject_to(opti.bounded(da_lim[0]*self.dt, u[1,i+1]-u[1,i], da_lim[1]*self.dt))

        opti.subject_to(x[:,N] - x_ss == slack)

        slack_cost = 1e6*ca.sumsqr(slack)
        total_cost = stage_cost + slack_cost
        opti.minimize(total_cost)

        solver_opts = {
            "mu_strategy" : "adaptive",
            "mu_init" : 1e-5,
            "mu_min" : 1e-15,
            "barrier_tol_factor" : 1,
            "print_level" : 0,
            "linear_solver" : "ma27"
            }
        # solver_opts = {}
        plugin_opts = {"verbose" : False, "print_time" : False, "print_out" : False}
        opti.solver('ipopt', plugin_opts, solver_opts)

        sol = opti.solve()

        slack_val = sol.value(slack)
        if la.norm(slack_val) > 1e-6:
            print('Warning! Solved, but with slack norm of %g is greater than 1e-8!' % la.norm(slack_val))

        if sol.stats()['success'] and la.norm(slack_val) <= 1e-6:
            print('Solve success, with slack norm of %g!' % la.norm(slack_val))
            feasible = True

            x_pred = sol.value(x)
            u_pred = sol.value(u)
            sol_cost = sol.value(stage_cost)
        else:
            # print(sol.stats()['return_status'])
            # print(opti.debug.show_infeasibilities())
            feasible = False
            x_pred = None
            u_pred = None
            sol_cost = None

            # pdb.set_trace()

        return x_pred, u_pred, sol_cost
