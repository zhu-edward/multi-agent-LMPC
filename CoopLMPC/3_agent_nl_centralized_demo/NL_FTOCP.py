import casadi as ca
import numpy as np
import numpy.linalg as la

import pdb

class NL_FTOCP(object):
    def __init__(self, agent):
        self.agent = agent

        self.n_x = self.agent.n_x
        self.n_u = self.agent.n_u
        self.n_a = self.agent.n_a

        self.l_r = self.agent.l_r
        self.l_f = self.agent.l_f

        self.dt = self.agent.dt

        self.F, self.b = self.agent.get_state_constraints()
        self.H, self.g = self.agent.get_input_constraints()
        self.r = self.agent.r

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

        self.cost = []

    def solve(self, abs_t, x_0, x_ss, N, last_u, x_guess=None, u_guess=None, expl_constraints=None, verbose=False):
        x = ca.SX.sym('x', self.n_x*(N+1))
        u = ca.SX.sym('y', self.n_u*N)
        slack = ca.SX.sym('slack', self.n_x)

        z = ca.vertcat(x, u, slack)

        # Flatten candidate solutions into 1-D array (- x_1 -, - x_2 -, ..., - x_N+1 -)
        if x_guess is not None:
            x_guess_flat = x_guess.flatten(order='F')
        else:
            x_guess_flat = np.zeros(self.n_x*(N+1))

        if u_guess is not None:
            u_guess_flat = u_guess.flatten(order='F')
        else:
            u_guess_flat = np.zeros(self.n_u*N)
        slack_guess = np.zeros(self.n_x)

        z_guess = np.concatenate((x_guess_flat, u_guess_flat, slack_guess))

        lb_slack = [-1000.0]*self.n_x
        ub_slack = [1000.0]*self.n_x

        # Box constraints on decision variables
        lb_x = x_0.tolist() + self.state_lb*(N) + self.input_lb*(N) + lb_slack
        ub_x = x_0.tolist() + self.state_ub*(N) + self.input_ub*(N) + ub_slack

        # Constraints on functions of decision variables
        lb_g = []
        ub_g = []

        stage_cost = 0
        constraint = []
        for i in range(N):
            stage_cost += 1

            # Formulate dynamics equality constraints as inequalities
            beta = ca.atan2(self.l_r*ca.tan(u[self.n_u*i+0]), self.l_f+self.l_r)
            constraint = ca.vertcat(constraint, x[self.n_x*(i+1)+0] - (x[self.n_x*i+0] + self.dt*x[self.n_x*i+3]*ca.cos(x[self.n_x*i+2] + beta)))
            constraint = ca.vertcat(constraint, x[self.n_x*(i+1)+1] - (x[self.n_x*i+1] + self.dt*x[self.n_x*i+3]*ca.sin(x[self.n_x*i+2] + beta)))
            constraint = ca.vertcat(constraint, x[self.n_x*(i+1)+2] - (x[self.n_x*i+2] + self.dt*x[self.n_x*i+3]*ca.sin(beta)))
            constraint = ca.vertcat(constraint, x[self.n_x*(i+1)+3] - (x[self.n_x*i+3] + self.dt*u[self.n_u*i+1]))

            lb_g += [0.0]*self.n_x
            ub_g += [0.0]*self.n_x

            # Steering rate constraints
            if self.ddf_lim is not None:
                if i == 0:
                    # Constrain steering rate with respect to previously applied input
                    constraint = ca.vertcat(constraint, u[self.n_u*(i)+0]-last_u[0])
                if i < N-1:
                    # Constrain steering rate along horizon
                    constraint = ca.vertcat(constraint, u[self.n_u*(i+1)+0]-u[self.n_u*(i)+0])
                lb_g += [self.ddf_lim[0]*self.dt]
                ub_g += [self.ddf_lim[1]*self.dt]

            # Throttle rate constraints
            if self.da_lim is not None:
                if i == 0:
                    # Constrain throttle rate
                    constraint = ca.vertcat(constraint, u[self.n_u*i+1]-last_u[1])
                if i < N-1:
                    constraint = ca.vertcat(constraint, u[self.n_u*(i+1)+1]-u[self.n_u*(i)+1])
                lb_g += [self.da_lim[0]*self.dt]
                ub_g += [self.da_lim[1]*self.dt]

            # Exploration constraints on predicted positions of agent
            if expl_constraints is not None:
                V = expl_constraints[i][0]
                w = expl_constraints[i][1]

                for j in range(V.shape[0]):
                    constraint = ca.vertcat(constraint, V[j,0]*x[self.n_x*i+0] + V[j,1]*x[self.n_x*i+1] + w[j])
                    lb_g += [-1e9]
                    ub_g += [0]

        # Formulate terminal soft equality constraint as inequalities
        constraint = ca.vertcat(constraint, x[self.n_x*N:] - x_ss + slack)
        lb_g += [0.0]*self.n_x
        ub_g += [0.0]*self.n_x

        slack_cost = 1e6*ca.sumsqr(slack)
        total_cost = stage_cost + slack_cost

        opts = {'verbose' : False, 'print_time' : 0,
            'ipopt.print_level' : 5,
            'ipopt.mu_strategy' : 'adaptive',
            'ipopt.mu_init' : 1e-5,
            'ipopt.mu_min' : 1e-15,
            'ipopt.barrier_tol_factor' : 1,
            'ipopt.linear_solver': 'ma27'}
        nlp = {'x' : z, 'f' : total_cost, 'g' : constraint}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        sol = solver(lbx=lb_x, ubx=ub_x, lbg=lb_g, ubg=ub_g, x0=z_guess)

        pdb.set_trace()

        if solver.stats()['success']:
            if la.norm(slack_val) <= 1e-8:
                print('Solve success for safe set point', x_ss)
                feasible = True

                z_sol = np.array(sol['x'])
                x_pred = z_sol[:self.n_x*(N+1)].reshape((N+1,self.n_x)).T
                u_pred = z_sol[self.n_x*(N+1):self.n_x*(N+1)+self.n_u*N].reshape((N,self.n_u)).T
                slack_val = z_sol[self.n_x*(N+1)+self.n_u*N:]
                cost_val = sol['f']
            else:
                print('Warning! Solved, but with slack norm of %g is greater than 1e-8!' % la.norm(slack_val))
                feasible = False
                x_pred = None
                u_pred = None
                cost_val = None
        else:
            print(solver.stats()['return_status'])
            feasible = False
            x_pred = None
            u_pred = None
            cost_val = None

            # pdb.set_trace()

        # pdb.set_trace()

        return x_pred, u_pred, cost_val

    def solve_opti(self, abs_t, x_0, x_ss, N, last_u, x_guess=None, u_guess=None, expl_constraints=None, verbose=False):
        opti = ca.Opti()

        x = opti.variable(self.n_x*self.n_a, N+1)
        u = opti.variable(self.n_u*self.n_a, N)
        slack = opti.variable(self.n_x*self.n_a)

        if x_guess is not None:
            opti.set_initial(x, x_guess)
        if u_guess is not None:
            opti.set_initial(u, u_guess)
        opti.set_initial(slack, np.zeros(self.n_x*self.n_a))

        da_lim = self.agent.da_lim
        ddf_lim = self.agent.ddf_lim

        pairs = list(itertools.combinations(range(self.n_a), 2))

        opti.subject_to(x[:,0] == np.squeeze(x_0))
        for j in range(self.n_a):
            opti.subject_to(opti.bounded(ddf_lim[0]*self.dt, u[j*self.n_u+0,0]-last_u[j*self.n_u+0], ddf_lim[1]*self.dt))
            opti.subject_to(opti.bounded(da_lim[0]*self.dt, u[j*self.n_u+1,0]-last_u[j*self.n_u+1], da_lim[1]*self.dt))

        stage_cost = 0
        for i in range(N):
            stage_cost = stage_cost + 1

            for j in range(self.n_a):
                beta = ca.atan2(self.l_r*ca.tan(u[j*self.n_u+0,i]), self.l_f+self.l_r)
                opti.subject_to(x[j*self.n_x+0,i+1] == x[j*self.n_x+0,i] + self.dt*x[j*self.n_x+3,i]*ca.cos(x[j*self.n_x+2,i] + beta))
                opti.subject_to(x[j*self.n_x+1,i+1] == x[j*self.n_x+1,i] + self.dt*x[j*self.n_x+3,i]*ca.sin(x[j*self.n_x+2,i] + beta))
                opti.subject_to(x[j*self.n_x+2,i+1] == x[j*self.n_x+2,i] + self.dt*x[j*self.n_x+3,i]*ca.sin(beta))
                opti.subject_to(x[j*self.n_x+3,i+1] == x[j*self.n_x+3,i] + self.dt*u[j*self.n_u+1,i])

            if self.F is not None:
                opti.subject_to(ca.mtimes(self.F, x[:,i]) <= self.b)
            if self.H is not None:
                opti.subject_to(ca.mtimes(self.H, u[:,i]) <= self.g)

            if expl_constraints is not None:
                V = expl_constraints[i][0]
                w = expl_constraints[i][1]
                opti.subject_to(ca.mtimes(V, x[:2,i]) + w <= np.zeros(len(w)))

            for p in pairs:
                opti.subject_to(ca.norm_2(x[p[0]*self.n_x:p[0]*self.n_x+1,i] - x[p[1]*self.n_x:p[1]*self.n_x+1,i]) >= self.r)

            if i < N-1:
                opti.subject_to(opti.bounded(ddf_lim[0]*self.dt, u[j*self.n_u+0,i+1]-u[j*self.n_u+0,i], ddf_lim[1]*self.dt))
                opti.subject_to(opti.bounded(da_lim[0]*self.dt, u[j*self.n_u+1,i+1]-u[j*self.n_u+1,i], da_lim[1]*self.dt))

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
        if la.norm(slack_val) > 2e-8:
            print('Warning! Solved, but with slack norm of %g is greater than 1e-8!' % la.norm(slack_val))

        if sol.stats()['success'] and la.norm(slack_val) <= 1e-8:
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
