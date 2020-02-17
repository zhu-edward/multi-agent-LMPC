import casadi as ca
import numpy as np

class NL_FTOCP(object):
    def __init__(self, N, agent):
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

    def solve(self, x_0, abs_t, verbose=False):
        opti = ca.Opti()

        x = opti.variable(self.n_x, self.N+1)
        y = opti.variable(self.n_u, self.N)

        opti.subject_to(x[:,0] == np.squeeze(x_0))
        opti.subject_to(opti.bounded(-0.3*self.dt, u[0,i]-self.u_preds[0,0,-1], 0.3*self.dt))
        opti.subject_to(opti.bounded(-3.0*self.dt, u[1,i]-self.u_preds[0,0,-1], 3.0*self.dt))

        for i in range(self.N):
            opti.subject_to(x[0,i+1] == x[0,i] + self.dt*x[3,i]*ca.cos(x[2,i] + ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)))
            opti.subject_to(x[1,i+1] == x[1,i] + self.dt*x[3,i]*ca.sin(x[2,i] + ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)))
            opti.subject_to(x[2,i+1] == x[2,i] + self.dt*x[3,i]*ca.sin(ca.atan2(self.l_r*ca.tan(u[0,i]), self.l_f+self.l_r)))
            opti.subject_to(x[3,i+1] == x[3,i] + self.dt*u[1,i])

            if self.F is not None:
                opti.subject_to(self.F*x[:,i] <= self.b)
            if self.H is not None:
                opti.subject_to(self.H*u[:,i] <= self.g)

            if i < self.N-1:
                opti.subject_to(opti.bounded(-0.3*self.dt, u[0,i+1]-u[0,i], 0.3*self.dt))
                opti.subject_to(opti.bounded(-3.0*self.dt, u[1,i+1]-u[1,i], 3.0*self.dt))
