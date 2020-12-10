
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools

class edgei(object):
	"""subsystem
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		-
		-
		- solve: uses ftocp and the stored data to comptute the predicted trajectory
	Attributes:
		- (dynamic model parameters) A_Ni, B_i, Q_i, R_i,
		- (graph information) neighborlists: Ni_to, Ni_from
		- (convex SS) alpha_i
		- (ADMM variables) dual multipliers lambda_i,
		- (MPC variables) x_i_pred, u_i_pred """
	def __init__(self, lambdaij, lambdaij_old, xij, xij_old):
		# Initialization
		#self.edgelist = edgelist
		self.lambdaij    = lambdaij
		self.xij   = xij
		self.lambdaij_old  = lambdaij_old
		self.xij_old = xij_old


"""
	def ADMM_edge_ini(self, M, syslist):
		# initialize all edges between the M subsystems in sysi list
		edgelist = np.zeros((M,M),dtype=object)
		for im in range(M):
			nx = syslist[im].Bi.shape[0]
			lambdaij = np.zeros(nx * N)
			lambdaij_old = np.zeros(nx * N)
			xij = np.zeros(nx * N)
			xij_old = np.zeros(nx * N)
			jm = syslist[im].Nifrom
			edgelist[im][jm] = edgei(lambdaij,lambdaij_old,xij,xij_old)
			lm = syslist[im].Nito
			edgelist[im][jm] = edgei(lambdaij, lambdaij_old, xij, xij_old)
		return edgelist
 """
