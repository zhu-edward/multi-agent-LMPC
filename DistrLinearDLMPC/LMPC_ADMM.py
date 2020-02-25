
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools

class LMPC_ADMM(object):
	"""Learning Model Predictive Controller (LMPC)
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""
	def __init__(self, ftocp_ADMM, syslist, M, CVX):
		# Initialization
		#self.ftocp = ftocp
		self.ftocp_ADMM = ftocp_ADMM
		#self.SS    = []
		#self.uSS   = []
		#self.Qfun  = []
		#self.Q = ftocp.Q
		#self.R = ftocp.R
		self.it    = 0
		self.CVX = CVX
		self.syslist = syslist
		self.Qfun = [[] for i in range(M)]
		self.cost = [[] for i in range(M)]
		self.SS = [[] for i in range(M)]
		self.uSS = [[] for i in range(M)]

		for m in range(M):
			self.Qfun[m] = syslist[m][m].Qfun
			self.cost[m] = syslist[m][m].cost
			self.SS[m] = syslist[m][m].SS
			self.uSS[m] = syslist[m][m].uSS



	def addTrajectory(self, ftocp_ADMM_list, syslist, M):

		for m in range(M):

			x = syslist[m][m].xcl
			u = syslist[m][m].ucl
			print("only x0 here and not xcl closed loop x")
			# Add the feasible trajectory x and the associated input sequence u to the safe set
			#
			#

			if syslist[m][m].SS == []:
				syslist[m][m].SS.append(copy.copy(x))
				syslist[m][m].uSS.append(copy.copy(u))
			else:
				syslist[m][m].SS = np.append(np.asarray(syslist[m][m].SS)[0], x)
				syslist[m][m].uSS = np.append(np.asarray(syslist[m][m].uSS)[0], u)
				print("these append here are wrong")

			# Compute and store the cost associated with the feasible trajectory
			#syslist[m][m].cost = self.computeCost(syslist, M)
			#syslist[m][m].Qfun.append(syslist[m][m].cost)
			#syslist[m][m].Qfun = np.append(syslist[m][m].Qfun, syslist[m][m].cost)

			self.computeCost(syslist, M)
			self.Qfun[m].append(self.cost[m])
			#syslist[m][m].Qfun = np.append(syslist[m][m].Qfun, syslist[m][m].cost)

			# Initialize zVector
			#self.zt = np.array(x[ftocp_ADMM_list[m].N])
	#		ftocp_ADMM_list[m].zt = np.array(x[ftocp_ADMM_list[m].N])

			# Augment iteration counter and print the cost of the trajectories stored in the safe set
			#self.it = self.it + 1
			#ftocp_ADMM_list[m].it = ftocp_ADMM_list[m].it+1
	#		print("Trajectory added to the Safe Set. Current Iteration: ", ftocp_ADMM_list[m].it)
			#print("Performance stored trajectories: \n", [np.asarray(self.Qfun[m])[i][0] for i in range(0, ftocp_ADMM_list[m].it)])


			print("Performance stored trajectories: \n", [np.asarray(self.Qfun[m])])
			print("Agent: \n", m)
			print("Check why the performance is empty!!!")

		for m in range(M):
			syslist[m][m].cost = self.cost[m]
			syslist[m][m].Qfun = self.Qfun[m]
			syslist[m][m].SS = self.SS[m]
			syslist[m][m].uSS = self.uSS[m]

		return syslist




	def computeCost(self, syslist, M):

		for m in range(M):

			x = syslist[m][m].xcl
			u = syslist[m][m].ucl

			# Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
			for i in range(0,len(x)):
				idx = len(x)-1 - i
				if i == 0:
					#syslist[m][m].cost = [np.dot(np.dot(x[idx],syslist[m][m].Qi),x[idx])]
					self.cost[m] = [np.dot(np.dot(x[idx], syslist[m][m].Qi), x[idx])]
				else:
					#syslist[m][m].cost.append(np.dot(np.dot(x[idx],syslist[m][m].Qi),x[idx]) + np.dot(np.dot(u[idx],syslist[m][m].Ri),u[idx]) + syslist[m][m].cost[-1])
					self.cost[m].append(np.dot(np.dot(x[idx], syslist[m][m].Qi), x[idx]) + np.dot(np.dot(u[idx], syslist[m][m].Ri), u[idx]) + self.cost[m][-1])

	#halt = 1
			# Finally flip the cost to have correct order
			#return np.flip(syslist[m][m].cost).tolist()
		for m in range(M):
			self.cost[m] = np.flip(self.cost[m]).tolist()

		return self.cost


	def solve(self, ftocp_ADMM_list, syslist, M, N, rho, ADMM_iterations, verbose = False):

		for m in range(M):

			# Build SS and cost matrices used in the ftocp
			# NOTE: it is possible to use a subset of the stored data to reduce computational complexity while having all guarantees on safety and performance improvement
			syslist[m][m].SS_vector = np.squeeze(list(itertools.chain.from_iterable(self.SS[m]))).T # From a 3D list to a 2D array
			syslist[m][m].Qfun_vector = 0
			syslist[m][m].Qfun_vector = np.expand_dims(np.array(list(itertools.chain.from_iterable(self.Qfun[m]))), 0) # From a 2D list to a 1D array

			# pdb.set_trace()

			# Reinitialize lambdas to zero
			n = syslist[m][m].n
			n_ss = syslist[m][m].SS_vector.shape[1]

			syslist[m][m].lambda_x = np.zeros(n * (N+1))
			syslist[m][m].lambda_x_old = np.zeros(n * (N+1))
			syslist[m][m].lambda_a = np.zeros(n_ss)
			syslist[m][m].lambda_a_old = np.zeros(n_ss)
			syslist[m][m].a = np.zeros(n_ss)  # a stands for alpha
			syslist[m][m].a_old = np.zeros(n_ss)

			for t in syslist[m][m].Ni_from:
				syslist[m][t].lambda_x = np.zeros(n * (N+1))
				syslist[m][t].lambda_x_old = np.zeros(n * (N+1))
				syslist[m][t].x = np.zeros(syslist[t][t].n * (N + 1))
				syslist[m][t].x_old = np.zeros(syslist[t][t].n * (N + 1))

		for ti in range(ADMM_iterations):

			# print(["ADMM Iteration:", ti])

			for m in range(M):
				syslist = ftocp_ADMM_list[m].solve_ADMM2(syslist, m, rho)

				# print(["Agent:", m, "x:", syslist[m][m].lambda_x])

			for m in range(M):
				syslist = ftocp_ADMM_list[m].solve_ADMM1(syslist[m][m].xt, syslist, m, N, rho, verbose, syslist[m][m].SS_vector, syslist[m][m].Qfun_vector, self.CVX)

				# print(["Agent:", m, "x:", syslist[m][m].x])
				# for t in syslist[m][m].Ni_from:
				# 	print(["Agent:", m, "neighbor:", t, "xt:", syslist[m][t].x])

			for m in range(M):
				syslist = ftocp_ADMM_list[m].update_ADMM1(syslist, m)

			#diff_1 = la.norm(syslist[0][0].x_old.flatten(order='F') - syslist[2][0].x_old.flatten(order='F'))
			#diff_2 = la.norm(syslist[1][1].x_old.flatten(order='F') - syslist[0][1].x_old.flatten(order='F'))
			#diff_3 = la.norm(syslist[2][2].x_old.flatten(order='F') - syslist[1][2].x_old.flatten(order='F'))
			diff_1 = la.norm(syslist[0][0].x.flatten(order='F') - syslist[2][0].x.flatten(order='F'))
			diff_2 = la.norm(syslist[1][1].x.flatten(order='F') - syslist[0][1].x.flatten(order='F'))
			diff_3 = la.norm(syslist[2][2].x.flatten(order='F') - syslist[1][2].x.flatten(order='F'))
			print('ADMM iteration: %i, diff 1: %g, diff 2: %g, diff 3: %g' % (ti, diff_1, diff_2, diff_3))

			diff_states = syslist[2][2].x.flatten(order='F') - syslist[1][2].x.flatten(order='F')
			print('ADMM iteration: %i, diff states: %g, diff 2: %g, diff 3: %g, diff 4: %g' % (ti, diff_states[0], diff_states[1], diff_states[2], diff_states[3]))


			diff_a1 = la.norm(syslist[0][0].a - syslist[1][1].a)
			diff_a2 = la.norm(syslist[1][1].a - syslist[2][2].a)
			diff_a3 = la.norm(syslist[2][2].a - syslist[0][0].a)
			print('ADMM iteration: %i, diff a: %g, diff 2: %g, diff 3: %g' % (ti, diff_a1, diff_a2, diff_a3))