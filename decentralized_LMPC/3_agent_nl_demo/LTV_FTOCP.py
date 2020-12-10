from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy as sp
import cvxpy as cp
import itertools, sys, pdb

class LTV_FTOCP(object):

	def __init__(self, Q, P, R, Rd, N, agent, x_refs=None, obstacles=[]):
		self.Q = Q
		self.P = P
		self.R = R
		self.N = N
		self.Rd = Rd

		self.agent = agent
		self.obstacles = obstacles
		self.dt = agent.dt

		self.n_x = self.agent.n_x
		self.n_u = self.agent.n_u

		if x_refs is not None:
			self.x_refs = x_refs
		else:
			self.x_refs = [np.zeros(self.n_x)]

		self.x_refs_idx = 0

		self.F, self.b = self.agent.get_state_constraints()
		self.H, self.g = self.agent.get_input_constraints()
		self.r = self.agent.get_collision_buff_r()

		self.cost = []

		self.x_preds = np.zeros((self.n_x, self.N+1, 1))
		self.u_preds = np.zeros((self.n_u, self.N, 1))

	def solve(self, x_0, abs_t, verbose=False):
		x = cp.Variable((self.n_x, self.N+1))
		u = cp.Variable((self.n_u, self.N))

		if len(self.obstacles) > 0:
			M = 1000

		da_lim = self.agent.da_lim[1]
		ddf_lim = self.agent.ddf_lim[1]

		# Initial condition
		constr = [x[:,0] == np.squeeze(x_0)]
		constr += [cp.abs(u[0,0]-self.u_preds[0,0,-1]) <= ddf_lim*self.dt] # Steering rate
		constr += [cp.abs(u[1,0]-self.u_preds[1,0,-1]) <= da_lim*self.dt] # Throttle rate

		cost = 0

		# Horizon constraints
		for i in range(self.N):
			# Time varying dynamics constraints
			x_lin = self.x_preds[:,i,-1]
			u_lin = self.u_preds[:,i,-1]
			A, B, c = self.agent.get_jacobians(x_lin, u_lin, 0.0001)

			constr += [x[:,i+1] == (np.eye(self.n_x) + A)@x[:,i] - A.dot(x_lin) + B@(u[:,i]-u_lin) + c]

			# State and input constraints
			if self.F is not None:
				constr += [self.F@x[:,i] <= self.b]
			if self.H is not None:
				constr += [self.H@u[:,i] <= self.g]

			# Stage cost
			cost += cp.quad_form(x[:,i]-self.x_refs[self.x_refs_idx], self.Q) + cp.quad_form(u[:,i], self.R)
			if i < self.N-1:
				constr += [cp.abs(u[0,i+1]-u[0,i]) <= ddf_lim*self.dt] # Steering rate
				constr += [cp.abs(u[1,i+1]-u[1,i]) <= da_lim*self.dt] # Throttle rate
				cost += cp.quad_form(u[:,i+1]-u[:,i], self.Rd) # Control rate penalty

		# Terminal state constraints
		if self.F is not None:
			constr += [self.F@x[:,self.N] <= self.b]
		# Terminal cost
		cost += cp.quad_form(x[:,self.N]-self.x_refs[self.x_refs_idx], self.P)

		# Solve the Finite Time Optimal Control Problem
		problem = cp.Problem(cp.Minimize(cost), constr)
		problem.solve(solver=cp.MOSEK, verbose=verbose)

		if problem.status != cp.OPTIMAL:
			if problem.status == cp.INFEASIBLE:
				print('Optimization was infeasible for time %g' % abs_t)
			elif problem.status == cp.UNBOUNDED:
				print('Optimization was unbounded for time %g' % abs_t)
			elif problem.status == cp.INFEASIBLE_INACCURATE:
				print('Optimization was infeasible inaccurate for time %g' % abs_t)
			elif problem.status == cp.UNBOUNDED_INACCURATE:
				print('Optimization was unbounded inaccurate for time %g' % abs_t)
			elif problem.status == cp.OPTIMAL_INACCURATE:
				print('Optimization was optimal inaccurate for time %g' % abs_t)

		self.cost.append(cost.value)

		if x.value is not None and u.value is not None:
			self.x_preds = np.append(self.x_preds, np.expand_dims(x.value, axis=-1), axis=-1)
			self.u_preds = np.append(self.u_preds, np.expand_dims(u.value, axis=-1), axis=-1)
		else:
			print('Optimization variables returned None')
			print(problem.status)
			pdb.set_trace()

		return x.value, u.value

	def update_x_refs(self, x_refs):
		self.x_refs = x_refs

	def get_x_refs(self):
		return self.x_refs

	def advance_reference_idx(self):
		self.x_refs_idx += 1

	def reset_reference_idx(self):
		self.x_refs_idx = 0

	def get_reference_idx(self):
		return self.x_refs_idx
