from __future__ import division

import numpy as np
from numpy import linalg as la
import pdb, copy, sys

import utils.utils

class NL_LMPC(object):
	"""Learning Model Predictive Controller (LMPC)
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""
	def __init__(self, ftocp, N):
		# Initialization
		self.ftocp = ftocp
		self.N = N
		self.ftocp_N = self.N

		self.Qfun  = []
		self.SS_t = []
		self.uSS_t = []
		self.Qfun_t = []

		self.n_x = self.ftocp.n_x
		self.n_u = self.ftocp.n_u

		self.it    = 0

		self.x_cls = []
		self.u_cls = []

		self.ss_idxs = []

		self.last_cost = np.inf

		self.x_preds_best = []
		self.u_preds_best = []
		self.idxs_best = []

		self.x_preds_best_it = []
		self.u_preds_best_it = []
		self.idxs_best_it = []

		self.traj_lens = []

		self.expl_constrs = None

	"""
	Add state trajectory and input sequence to the set of candidate points for safe set creation
	"""
	def addTrajectory(self, x, u, x_f):
		print(x.shape)

		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.x_cls.append(copy.copy(x))
		self.u_cls.append(copy.copy(u))
		self.traj_lens.append(x.shape[1])

		# Compute and store the cost associated with the feasible trajectory
		cost = np.array(self.computeCost(x, u, x_f))
		self.last_cost = cost[0]
		self.Qfun.append(cost)

		# Reset horizon length
		self.ftocp_N = self.N

		# Save best predictions and safe set indicies
		if self.it > 0:
			self.x_preds_best.append(self.x_preds_best_it)
			self.u_preds_best.append(self.u_preds_best_it)
			self.idxs_best.append(self.idxs_best_it)
			self.x_preds_best_it = []
			self.u_preds_best_it = []
			self.idxs_best_it = []

		# Increment iteration counter and print the cost of the trajectories stored in the safe set
		self.it += 1
		print ('Trajectory of length %i added to the Safe Set. Current Iteration: %i' % (x.shape[1], self.it))
		print ('Performance of stored trajectories:')
		print([self.Qfun[i][0] for i in range(self.it)])

		return cost

	"""
	Compute the cost to go for each element in trajectory in a DP like strategy: start from the last point x[len(x)-1] and move backwards
	"""
	def computeCost(self, x, u, x_f):
		l = x.shape[1]
		for t in range(l-1,-1,-1):
			if t == l-1: # Terminal cost
				cost = [0]
			else:
				cost.append(1 + cost[-1])
				# if la.norm(x[:,t] - x_f, 2) <= 1e-5:
				# 	cost.append(0)
				# else:
				# 	cost.append(1 + cost[-1])
		# Finally flip the cost to have correct order
		return np.flip(cost).tolist()

	def solve(self, ts, x_t, x_f, tol, verbose=True):
		# Get the safe set, cost-to-go, and indicies at this time step
		SS = self.SS_t[min(ts+self.ftocp_N, self.traj_lens[-1]-1)]
		Qfun = self.Qfun_t[min(ts+self.ftocp_N, self.traj_lens[-1]-1)]
		idxs = self.idxs_t[min(ts+self.ftocp_N, self.traj_lens[-1]-1)]
		if self.expl_constrs is not None:
			expl_con = self.expl_constrs[ts:ts+self.ftocp_N]
		else:
			expl_con = None

		cost_cands = []
		x_pred_cands = []
		u_pred_cands = []
		idx_cands = []
		x_ss_cands = []

		# Form candidate solution
		if len(self.x_preds_best_it) == 0:
			if len(self.x_preds_best) == 0:
				# Use initial feasible trajectory for first timestep of first iteration
				x_guess = self.x_cls[-1][:,:self.N+1]
				u_guess = self.u_cls[-1][:,:self.N]
			else:
				# On new iteration, use predictions from first time step of last iteration
				x_guess = self.x_preds_best[-1][0]
				u_guess = self.u_preds_best[-1][0]
			last_u = np.zeros(self.n_u)
		else:
			if self.ftocp_N == self.N:
				# Get the safe set point chosen at the last time step,
				# find its successor state and input and append to the prediction at the last time step
				it_idx, ts_idx = self.idxs_best_it[-1]
				ss_kp1 = self.x_cls[it_idx][:,min(ts_idx+1, self.traj_lens[it_idx]-1)]
				uss_k = self.u_cls[it_idx][:,min(ts_idx, self.traj_lens[it_idx]-1)]
				x_guess = np.append(self.x_preds_best_it[-1][:,1:], ss_kp1.reshape((-1,1)), axis=1)
				u_guess = np.append(self.u_preds_best_it[-1][:,1:], uss_k.reshape((-1,1)), axis=1)
			else:
				# We should be able to reach the goal state now so the prediction horizon is shrinking
				# We shrink the candidate solution by one step
				x_guess = self.x_preds_best_it[-1][:,1:]
				u_guess = self.u_preds_best_it[-1][:,1:]
			last_u = self.u_preds_best_it[-1][:,0]

		# pdb.set_trace()

		# Solve for each element in safe set
		for i in range(SS.shape[1]):
			# pdb.set_trace()
			x_ss = SS[:,i]
			term_cost = Qfun[i]
			idx = idxs[i]

			# We only attempt to solve with safe set points which offer the possibility of cost improvement
			if self.ftocp_N > 1:
				if self.ftocp_N + term_cost <= self.last_cost:
				# if True:
					x_pred, u_pred, cost = self.ftocp.solve_opti(ts, x_t, x_ss, self.ftocp_N, last_u,
						x_guess=x_guess,
						u_guess=u_guess,
						expl_constraints=expl_con,
						verbose=verbose)
					# x_pred, u_pred, cost = self.ftocp.solve(ts, x_t, x_ss, self.ftocp_N, last_u,
					# 	x_guess=x_guess,
					# 	u_guess=u_guess,
					# 	expl_constraints=expl_con,
					# 	verbose=verbose)
					if cost is not None:
						x_pred_cands.append(x_pred)
						u_pred_cands.append(u_pred)
						cost_cands.append(cost + term_cost)
						idx_cands.append(idx)
						x_ss_cands.append(x_ss)
				else:
					print('No performance improvement possible, skipping...')
			else:
				u_t = self.u_preds_best_it[-1][:,1]
				x_tp1 = self.ftocp.agent.sim(x_t, u_t)
				# Check for feasibility and store the solution
				if la.norm(x_tp1 - x_f) <= 10**tol:
					cost_cands.append(1 + term_cost)
					x_pred_cands.append(np.hstack((x_t.reshape((-1,1)), x_f.reshape((-1,1)))))
					u_pred_cands.append(u_t.reshape((-1,1)))
					idx_cands.append(None)
					x_ss_cands.append(None)

		if len(cost_cands) > 0:
			min_idx = np.argmin(cost_cands)
			x_pred_best = x_pred_cands[min_idx]
			u_pred_best = u_pred_cands[min_idx]
			cost_best = cost_cands[min_idx]
			idx_best = idx_cands[min_idx]
			x_ss_best = x_ss_cands[min_idx]

			self.x_preds_best_it.append(x_pred_best)
			self.u_preds_best_it.append(u_pred_best)
			self.idxs_best_it.append(idx_best)
		else:
			print('None of the safe set points are feasible terminal conditions')
			x_pred_best = None
			u_pred_best = None
			cost_best = np.inf
			pdb.set_trace()

		if x_pred_best is not None and self.ftocp_N > 1:
			if la.norm(x_pred_best[:,-1] - x_f) <= 10**tol:
				print('Reaching goal state at end of horizon, decreasing horizon from %i to %i' % (self.ftocp_N, self.ftocp_N-1))
				self.ftocp_N -= 1

		if cost_best > self.last_cost:
			print('ERROR: The cost is not decreasing')
			# pdb.set_trace()

		self.last_cost = cost_best

		return x_pred_best, u_pred_best, cost_best, SS, self.ftocp_N

	def get_safe_set_q_func(self):
		return (self.SS_t, self.uSS_t, self.Qfun_t)

	"""
	Update the safe sets given time and iteration indicies for each step
	"""
	def update_safe_sets(self, ss_idxs):
		self.ss_idxs = ss_idxs

		self.idxs_t = []
		self.SS_t = []
		self.uSS_t = []
		self.Qfun_t = []
		for t in range(len(self.ss_idxs)-1):
			ss = np.empty((self.n_x,0))
			uss = np.empty((self.n_u,0))
			qfun = np.empty(0)
			idxs = []
			for (i, j) in enumerate(self.ss_idxs[t]['it_range']):
				# Collect states corresponding to iteration and timestep indicies
				ss = np.append(ss, self.x_cls[j][:,self.ss_idxs[t]['ts_range'][i]], axis=1)
				# Collect inputs corresponding to iteration and timestep indicies
				uss = np.append(uss, self.u_cls[j][:,self.ss_idxs[t]['ts_range'][i]], axis=1)
				# Collect cost-to-gos corresponding to iteration and timestep indicies
				qfun = np.append(qfun, self.Qfun[j][self.ss_idxs[t]['ts_range'][i]])
				# Collect iteration and timestep indicies
				for k in self.ss_idxs[t]['ts_range'][i]:
					idxs.append((j, k))
			self.SS_t.append(ss)
			self.uSS_t.append(uss)
			self.Qfun_t.append(qfun)
			self.idxs_t.append(idxs)

		# pdb.set_trace()

	def update_exploration_constraints(self, expl_constrs):
		self.expl_constrs = expl_constrs
