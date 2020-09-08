import casadi as ca
import numpy as np
import numpy.linalg as la

import pdb

class init_FTOCP(object):
    def __init__(self, Q, P, R, Rd, N, agent, x_refs=None):
        self.Q = Q
        self.P = P
        self.R = R
        self.N = N
        self.Rd = Rd

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

        self.x_refs_idx = 0
        if x_refs is not None:
            self.x_refs = x_refs
        else:
            self.x_refs = [np.zeros(self.n_x)]

        self.cost = None
        self.opti = None
        self.agt_x = []
        self.x_s = None
        self.x_f = None
        self.x = None
        self.u = None

        self.opti0 = None
        self.cost0 = None
        self.agt_x0 = []
        self.x_s0 = None
        self.x_f0 = None
        self.x0 = None
        self.u0 = None

    def build_opti_solver(self, agt_idx):
        self.opti = ca.Opti()

        self.x = self.opti.variable(self.n_x, self.N+1)
        self.u = self.opti.variable(self.n_u, self.N)
        self.x_s = self.opti.parameter(self.n_x)
        self.x_f = self.opti.parameter(self.n_x)
        self.last_u = self.opti.parameter(self.n_u)

        self.agt_x = []
        for i in range(agt_idx):
            self.agt_x.append(self.opti.parameter(self.n_x, self.N+1))

        self.opti.set_value(self.x_f, self.x_refs[self.x_refs_idx])

        da_lim = self.agent.da_lim
        ddf_lim = self.agent.ddf_lim
        r = self.agent.r

        self.opti.subject_to(self.x[:,0] == self.x_s)
        self.opti.subject_to(self.opti.bounded(ddf_lim[0]*self.dt, self.u[0,0]-self.last_u[0], ddf_lim[1]*self.dt))
        self.opti.subject_to(self.opti.bounded(da_lim[0]*self.dt, self.u[1,0]-self.last_u[1], da_lim[1]*self.dt))

        stage_cost = 0
        for i in range(self.N):
            stage_cost += ca.bilin(self.Q, self.x[:,i+1]-self.x_f, self.x[:,i+1]-self.x_f) + ca.bilin(self.R, self.u[:,i], self.u[:,i])

            beta = ca.atan2(self.l_r*ca.tan(self.u[0,i]), self.l_f+self.l_r)
            self.opti.subject_to(self.x[0,i+1] == self.x[0,i] + self.dt*self.x[3,i]*ca.cos(self.x[2,i] + beta))
            self.opti.subject_to(self.x[1,i+1] == self.x[1,i] + self.dt*self.x[3,i]*ca.sin(self.x[2,i] + beta))
            self.opti.subject_to(self.x[2,i+1] == self.x[2,i] + self.dt*self.x[3,i]*ca.sin(beta))
            self.opti.subject_to(self.x[3,i+1] == self.x[3,i] + self.dt*self.u[1,i])

            if self.F is not None:
                self.opti.subject_to(ca.mtimes(self.F, self.x[:,i+1]) <= self.b)
            if self.H is not None:
                self.opti.subject_to(ca.mtimes(self.H, self.u[:,i]) <= self.g)

            # Treat init and final states of agents before as obstacles
            for j in range(agt_idx):
                self.opti.subject_to(ca.bilin(np.eye(2), self.x[:2,i+1]-self.agt_x[j][:2,i+1], self.x[:2,i+1]-self.agt_x[j][:2,i+1]) >= (1.5*2*r)**2)

            if i < self.N-1:
                stage_cost += ca.bilin(self.Rd, self.u[:,i+1]-self.u[:,i], self.u[:,i+1]-self.u[:,i])
                self.opti.subject_to(self.opti.bounded(ddf_lim[0]*self.dt, self.u[0,i+1]-self.u[0,i], ddf_lim[1]*self.dt))
                self.opti.subject_to(self.opti.bounded(da_lim[0]*self.dt, self.u[1,i+1]-self.u[1,i], da_lim[1]*self.dt))

        self.cost = stage_cost
        self.opti.minimize(self.cost)

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
        self.opti.solver('ipopt', plugin_opts, solver_opts)

    def build_opti0_solver(self, agt_idx):
        self.opti0 = ca.Opti()

        self.x0 = self.opti0.variable(self.n_x, self.N+1)
        self.u0 = self.opti0.variable(self.n_u, self.N)
        self.x_s0 = self.opti0.parameter(self.n_x)
        self.x_f0 = self.opti0.parameter(self.n_x)

        self.agt_x0 = []
        for i in range(agt_idx):
            self.agt_x0.append(self.opti0.parameter(self.n_x, self.N+1))

        self.opti0.set_value(self.x_f0, self.x_refs[self.x_refs_idx])

        da_lim = self.agent.da_lim
        ddf_lim = self.agent.ddf_lim
        r = self.agent.r

        self.opti0.subject_to(self.x0[0,0] == self.x_s0[0])
        self.opti0.subject_to(self.x0[1,0] == self.x_s0[1])
        self.opti0.subject_to(self.opti0.bounded(0, self.x0[2,0], 2*np.pi))
        self.opti0.subject_to(self.x0[3,0] == self.x_s0[3])

        stage_cost = 0
        for i in range(self.N):
            stage_cost += ca.bilin(self.Q, self.x0[:,i+1]-self.x_f0, self.x0[:,i+1]-self.x_f0) + ca.bilin(self.R, self.u0[:,i], self.u0[:,i])

            beta = ca.atan2(self.l_r*ca.tan(self.u0[0,i]), self.l_f+self.l_r)
            self.opti0.subject_to(self.x0[0,i+1] == self.x0[0,i] + self.dt*self.x0[3,i]*ca.cos(self.x0[2,i] + beta))
            self.opti0.subject_to(self.x0[1,i+1] == self.x0[1,i] + self.dt*self.x0[3,i]*ca.sin(self.x0[2,i] + beta))
            self.opti0.subject_to(self.x0[2,i+1] == self.x0[2,i] + self.dt*self.x0[3,i]*ca.sin(beta))
            self.opti0.subject_to(self.x0[3,i+1] == self.x0[3,i] + self.dt*self.u0[1,i])

            if self.F is not None:
                self.opti0.subject_to(ca.mtimes(self.F, self.x0[:,i+1]) <= self.b)
            if self.H is not None:
                self.opti0.subject_to(ca.mtimes(self.H, self.u0[:,i]) <= self.g)

            # Treat init and final states of agents before as obstacles
            for j in range(agt_idx):
                self.opti0.subject_to(ca.bilin(np.eye(2), self.x0[:2,i+1]-self.agt_x0[j][:2,i+1], self.x0[:2,i+1]-self.agt_x0[j][:2,i+1]) >= (1.5*2*r)**2)

            if i < self.N-1:
                stage_cost += ca.bilin(self.Rd, self.u0[:,i+1]-self.u0[:,i], self.u0[:,i+1]-self.u0[:,i])
                self.opti0.subject_to(self.opti0.bounded(ddf_lim[0]*self.dt, self.u0[0,i+1]-self.u0[0,i], ddf_lim[1]*self.dt))
                self.opti0.subject_to(self.opti0.bounded(da_lim[0]*self.dt, self.u0[1,i+1]-self.u0[1,i], da_lim[1]*self.dt))

        self.cost0 = stage_cost
        self.opti0.minimize(self.cost0)

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
        self.opti0.solver('ipopt', plugin_opts, solver_opts)

    def solve_opti(self, k, x_0, last_u, agt_trajs, x_guess=None, u_guess=None, verbose=False):
        if x_guess is not None:
            self.opti.set_initial(self.x, x_guess)
        if u_guess is not None:
            self.opti.set_initial(self.u, u_guess)

        self.opti.set_value(self.last_u, last_u)
        self.opti.set_value(self.x_s, x_0)

        for i in range(len(self.agt_x)):
            if k < agt_trajs[i].shape[1]-1:
                traj = agt_trajs[i][:,k:min(k+self.N+1,agt_trajs[i].shape[1])]
                if traj.shape[1] < self.N+1:
                    traj = np.hstack((traj, np.tile(traj[:,-1].reshape((-1,1)), (self.N+1-traj.shape[1]))))
            else:
                traj = np.tile(agt_trajs[i][:,-1].reshape((-1,1)), (self.N+1))
            self.opti.set_value(self.agt_x[i], traj)

        try:
            sol = self.opti.solve()
            feasible = True
            x_pred = sol.value(self.x)
            u_pred = sol.value(self.u)
            return x_pred, u_pred, feasible
        except:
            print(self.opti.stats()['return_status'])
            return None, None, False

        return x_pred, u_pred, feasible

    def solve_opti0(self, k, x_0, agt_trajs, x_guess=None, u_guess=None, verbose=False):
        if x_guess is not None:
            self.opti0.set_initial(self.x0, x_guess)
        if u_guess is not None:
            self.opti0.set_initial(self.u0, u_guess)

        self.opti0.set_value(self.x_s0, x_0)

        for i in range(len(self.agt_x0)):
            if k < agt_trajs[i].shape[1]-1:
                traj = agt_trajs[i][:,k:min(k+self.N+1,agt_trajs[i].shape[1])]
                if traj.shape[1] < self.N+1:
                    traj = np.hstack((traj, np.tile(agt_trajs[i][:,-1].reshape((-1,1)), (self.N+1-traj.shape[1]))))
            else:
                traj = np.tile(agt_trajs[i][:,-1].reshape((-1,1)), (self.N+1))
            self.opti0.set_value(self.agt_x0[i], traj)

        try:
            sol = self.opti0.solve()
            feasible = True
            x_pred = sol.value(self.x0)
            u_pred = sol.value(self.u0)
            return x_pred, u_pred, feasible
        except:
            print(self.opti0.stats()['return_status'])
            return None, None, False

        return x_pred, u_pred, feasible

    def update_x_refs(self, x_refs):
        self.x_refs = x_refs
        self.opti.set_value(self.x_f, self.x_refs[self.x_refs_idx])

    def get_x_refs(self):
        return self.x_refs

    def advance_reference_idx(self):
        self.x_refs_idx += 1
        self.opti.set_value(self.x_f, self.x_refs[self.x_refs_idx])

    def reset_reference_idx(self):
        self.x_refs_idx = 0
        self.opti.set_value(self.x_f, self.x_refs[self.x_refs_idx])

    def get_reference_idx(self):
        return self.x_refs_idx
