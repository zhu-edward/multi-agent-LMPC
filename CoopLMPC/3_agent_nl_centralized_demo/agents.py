import numpy as np
from scipy import linalg as sla
import pdb

from dynamics_models import Centralized_DT_Kin_Bike_Model, DT_Kin_Bike_Model

class DT_Kin_Bike_Agent(DT_Kin_Bike_Model):
	def __init__(self, l_r, l_f, w, dt, col_buf=None,
		a_lim=[-3.0, 3.0], df_lim=[-0.5, 0.5], x_lim=[-10.0, 10.0],
		y_lim=[-10.0, 10.0], psi_lim=None, v_lim=[-10.0, 10.0],
		da_lim=[-7.0, 7.0], ddf_lim=[-0.7, 0.7]):
		super(DT_Kin_Bike_Agent, self).__init__(l_r, l_f, dt)

		self.w = w
		self.l = l_r + l_f
		if col_buf is None:
			self.col_buf = 0
		else:
			self.col_buf = col_buf
		self.r = np.sqrt((self.w/2.0)**2 + (self.l/2.0)**2) + self.col_buf
		self.dt = dt

		self.a_lim = a_lim
		self.df_lim = df_lim

		self.x_lim = x_lim
		self.y_lim = y_lim
		self.psi_lim = psi_lim
		self.v_lim = v_lim

		self.da_lim = da_lim
		self.ddf_lim = ddf_lim

		# Build the matrices for the input constraint
		F = []
		b = []
		if self.x_lim is not None:
			F.append(np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]]))
			b.append(np.array([[x_lim[1]], [-x_lim[0]]]))

		if self.y_lim is not None:
			F.append(np.array([[0., 1., 0., 0.], [0., -1., 0., 0.]]))
			b.append(np.array([[y_lim[1]], [-y_lim[0]]]))

		if self.psi_lim is not None:
			F.append(np.array([[0., 0., 1., 0.], [0., 0., -1., 0.]]))
			b.append(np.array([[psi_lim[1]], [-psi_lim[0]]]))

		if self.v_lim is not None:
			F.append(np.array([[0., 0., 0., 1.], [0., 0., 0., -1.]]))
			b.append(np.array([[v_lim[1]], [-v_lim[0]]]))

		if len(F) > 0:
			self.F = np.vstack(F)
			self.b = np.squeeze(np.vstack(b))
		else:
			self.F = None
			self.b = None

		H = []
		g = []
		if self.df_lim is not None:
			H.append(np.array([[1., 0.], [-1., 0.]]))
			g.append(np.array([[df_lim[1]], [-df_lim[0]]]))

		if self.a_lim is not None:
			H.append(np.array([[0., 1.], [0., -1.]]))
			g.append(np.array([[a_lim[1]], [-a_lim[0]]]))

		if len(H) > 0:
			self.H = np.vstack(H)
			self.g = np.squeeze(np.vstack(g))
		else:
			self.H = None
			self.b = None

	def get_jacobians(self, x, u, eps):
	   # A, B, c = self.get_numerical_jacs(x, u, eps)
	   A, B, c = self.get_jacs(x, u)

	   return A, B, c

	def update_state_input(self, x, u):
		self.state_his.append(x)
		self.input_his.append(u)

	def get_state_constraints(self):
		return self.F, self.b

	def get_input_constraints(self):
		return self.H, self.g

	def get_collision_buff_r(self):
		return self.r

	def get_state_input_his(self):
		return self.state_his, self.input_his

class Centralized_DT_Kin_Bike_Agent(Centralized_DT_Kin_Bike_Model):
	def __init__(self, l_r, l_f, w, dt, n_a, col_buf=None,
		a_lim=[-3.0, 3.0], df_lim=[-0.5, 0.5], x_lim=[-10.0, 10.0],
		y_lim=[-10.0, 10.0], psi_lim=None, v_lim=[-10.0, 10.0],
		da_lim=[-7.0, 7.0], ddf_lim=[-0.7, 0.7]):
		super(Centralized_DT_Kin_Bike_Agent, self).__init__(l_r, l_f, dt, n_a)

		self.w = w
		self.l = l_r + l_f
		if col_buf is None:
			self.col_buf = 0
		else:
			self.col_buf = col_buf
		self.r = np.sqrt((self.w/2.0)**2 + (self.l/2.0)**2) + self.col_buf
		self.dt = dt
		self.n_a = n_a

		self.a_lim = a_lim
		self.df_lim = df_lim

		self.x_lim = x_lim
		self.y_lim = y_lim
		self.psi_lim = psi_lim
		self.v_lim = v_lim

		self.da_lim = da_lim
		self.ddf_lim = ddf_lim

		# Build the matrices for the input constraint
		F = []
		b = []
		if self.x_lim is not None:
			F.append(np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]]))
			b.append(np.array([[x_lim[1]], [-x_lim[0]]]))

		if self.y_lim is not None:
			F.append(np.array([[0., 1., 0., 0.], [0., -1., 0., 0.]]))
			b.append(np.array([[y_lim[1]], [-y_lim[0]]]))

		if self.psi_lim is not None:
			F.append(np.array([[0., 0., 1., 0.], [0., 0., -1., 0.]]))
			b.append(np.array([[psi_lim[1]], [-psi_lim[0]]]))

		if self.v_lim is not None:
			F.append(np.array([[0., 0., 0., 1.], [0., 0., 0., -1.]]))
			b.append(np.array([[v_lim[1]], [-v_lim[0]]]))

		if len(F) > 0:
			F = np.vstack(F)
			b = np.squeeze(np.vstack(b))
			F_cent = [F for _ in range(n_a)]
			self.F = sla.block_diag(*F_cent)
			self.b = np.tile(b, n_a)
		else:
			self.F = None
			self.b = None

		H = []
		g = []
		if self.df_lim is not None:
			H.append(np.array([[1., 0.], [-1., 0.]]))
			g.append(np.array([[df_lim[1]], [-df_lim[0]]]))

		if self.a_lim is not None:
			H.append(np.array([[0., 1.], [0., -1.]]))
			g.append(np.array([[a_lim[1]], [-a_lim[0]]]))

		if len(H) > 0:
			H = np.vstack(H)
			g = np.squeeze(np.vstack(g))
			H_cent = [H for _ in range(n_a)]
			self.H = sla.block_diag(*H_cent)
			self.g = np.tile(g, n_a)
		else:
			self.H = None
			self.b = None

	def get_jacobians(self, x, u, eps):
	   # A, B, c = self.get_numerical_jacs(x, u, eps)
	   A, B, c = self.get_jacs(x, u)

	   return A, B, c

	def update_state_input(self, x, u):
		self.state_his.append(x)
		self.input_his.append(u)

	def get_state_constraints(self):
		return self.F, self.b

	def get_input_constraints(self):
		return self.H, self.g

	def get_collision_buff_r(self):
		return self.r

	def get_state_input_his(self):
		return self.state_his, self.input_his
