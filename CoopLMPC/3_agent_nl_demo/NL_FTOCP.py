import casadi as ca
import numpy as np

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

    def solve(self, x_0, abs_t, SS, expl_constraints, verbose=False):
        opti = ca.Opti()

        if expl_constraints is not None:
            V = expl_constraints[0]
            w = expl_constraints[1]

        x = opti.variable(self.n_x, self.N+1)
        u = opti.variable(self.n_u, self.N)
        gamma = opti.variable()

        opti.subject_to(x[:,0] == np.squeeze(x_0))
        opti.subject_to(opti.bounded(-0.5*self.dt, u[0,0]-self.u_preds[0,0,-1], 0.5*self.dt))
        opti.subject_to(opti.bounded(-3.0*self.dt, u[1,0]-self.u_preds[0,0,-1], 3.0*self.dt))

        cost = 0
        for i in range(self.N):
            cost = cost + 1

            opti.subject_to(x[0,i+1] == x[0,i] + self.dt*x[3,i]*ca.cos(x[2,i] + ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)))
            opti.subject_to(x[1,i+1] == x[1,i] + self.dt*x[3,i]*ca.sin(x[2,i] + ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)))
            opti.subject_to(x[2,i+1] == x[2,i] + self.dt*x[3,i]*ca.sin(ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)))
            opti.subject_to(x[3,i+1] == x[3,i] + self.dt*u[1,i])

            if self.F is not None:
                opti.subject_to(ca.mtimes(self.F, x[:,i]) <= self.b)
            if self.H is not None:
                opti.subject_to(ca.mtimes(self.H, u[:,i]) <= self.g)

            if expl_constraints is not None:
                opti.subject_to(ca.mtimes(V[i], x[:2,i]) <= w[i])

            if i < self.N-1:
                opti.subject_to(opti.bounded(-0.5*self.dt, u[0,i+1]-u[0,i], 0.5*self.dt))
                opti.subject_to(opti.bounded(-3.0*self.dt, u[1,i+1]-u[1,i], 3.0*self.dt))

        
