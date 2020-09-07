import numpy as np

class CT_Kin_Bike_Model(object):

    def __init__(self, l_r, l_f):

        self.l_r = l_r
        self.l_f = l_f

        self.n_x = 4
        self.n_u = 2

    def sim(self, x, u):
        beta = np.arctan2(self.l_r*np.tan(u[0]), self.l_f + self.l_r)
        x_dot = np.zeros(4)
        x_dot[0] = x[3]*np.cos(x[2] + beta)
        x_dot[1] = x[3]*np.sin(x[2] + beta)
        x_dot[2] = x[3]*np.sin(beta)
        x_dot[3] = u[1]

        return x_dot

    def get_numerical_jacs(self, x, u, eps):
        A = np.zeros((self.n_x, self.n_x))
        B = np.zeros((self.n_x, self.n_u))
        c = np.zeros(self.n_x)

        for i in range(self.n_x):
            e = np.zeros(self.n_x)
            e[i] = eps

            x_u = x + e
            x_l = x - e

            A[:,i] = (self.sim(x_u, u) - self.sim(x_l, u))/(2*eps)

        for i in range(self.n_u):
            e = np.zeros(self.n_u)
            e[i] = eps

            u_u = u + e
            u_l = u - e

            B[:,i] = (self.sim(x, u_u) - self.sim(x, u_l))/(2*eps)

        c = self.sim(x, u)

        return A, B, c

class DT_Kin_Bike_Model(object):

    def __init__(self, l_r, l_f, dt):

        self.l_r = l_r
        self.l_f = l_f
        self.dt = dt

        self.n_x = 4
        self.n_u = 2

    def sim(self, x_k, u_k):
        beta = np.arctan2(self.l_r*np.tan(u_k[0]), self.l_f + self.l_r)
        x_kp1 = np.zeros(4)
        x_kp1[0] = x_k[0] + self.dt*x_k[3]*np.cos(x_k[2] + beta)
        x_kp1[1] = x_k[1] + self.dt*x_k[3]*np.sin(x_k[2] + beta)
        x_kp1[2] = x_k[2] + self.dt*x_k[3]*np.sin(beta)
        x_kp1[3] = x_k[3] + self.dt*u_k[1]

        return x_kp1

    def sim_ct(self, x, u):
        beta = np.arctan2(self.l_r*np.tan(u[0]), self.l_f + self.l_r)
        x_dot = np.zeros(4)
        x_dot[0] = x[3]*np.cos(x[2] + beta)
        x_dot[1] = x[3]*np.sin(x[2] + beta)
        x_dot[2] = x[3]*np.sin(beta)
        x_dot[3] = u[1]

        return x_dot

    def get_jacs(self, x, u):
        beta = np.arctan2(self.l_r*np.tan(u[0]), self.l_f + self.l_r)
        dbeta_ddf = lambda df : self.l_r/(np.cos(u[0])**2*(self.l_f+self.l_r)*(1+(self.l_r*np.tan(u[0])/(self.l_f+self.l_r))**2))

        A_c = np.zeros((self.n_x, self.n_x))
        B_c = np.zeros((self.n_x, self.n_u))
        c_c = np.zeros(self.n_x)

        A_c[0,2] = -x[3]*np.sin(x[2]+beta)
        A_c[0,3] = np.cos(x[2]+beta)
        A_c[1,2] = x[3]*np.cos(x[2]+beta)
        A_c[1,3] = np.sin(x[2]+beta)
        A_c[2,3] = np.sin(beta)/self.l_r

        B_c[0,0] = -x[3]*np.sin(x[2]+beta)*dbeta_ddf(u[0])
        B_c[1,0] = x[3]*np.cos(x[2]+beta)*dbeta_ddf(u[0])
        B_c[2,0] = x[3]*np.cos(beta)*dbeta_ddf(u[0])/self.l_r
        B_c[3,1] = 1

        c_c = self.sim_ct(x, u)

        A_d = self.dt*A_c
        B_d = self.dt*B_c
        c_d = self.dt*c_c

        return A_d, B_d, c_d

    def get_numerical_jacs(self, x, u, eps):
        A_c = np.zeros((self.n_x, self.n_x))
        B_c = np.zeros((self.n_x, self.n_u))
        c_c = np.zeros(self.n_x)

        for i in range(self.n_x):
            e = np.zeros(self.n_x)
            e[i] = eps

            x_u = x + e
            x_l = x - e

            A_c[:,i] = (self.sim_ct(x_u, u) - self.sim_ct(x_l, u))/(2*eps)

        for i in range(self.n_u):
            e = np.zeros(self.n_u)
            e[i] = eps

            u_u = u + e
            u_l = u - e

            B_c[:,i] = (self.sim_ct(x, u_u) - self.sim_ct(x, u_l))/(2*eps)

        c_c = self.sim_ct(x, u)

        A_d = np.eye(self.n_x) + self.dt*A_c
        B_d = self.dt*B_c
        c_d = self.dt*c_c

        return A_d, B_d, c_d
