
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools

class sysi(object):
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
    def __init__(self, ANi, Bi, Qi, Ri, Ni_to, Ni_from, a, a_old, lambda_x, lambda_x_old, lambda_a, lambda_a_old, x, x_old, SS, uSS, cost, Qfun, xcl, ucl, xPred, uPred, aPred):
        # Initialization
        self.ANi = ANi
        self.Bi    = Bi
        self.Qi   = Qi
        self.Ri  = Ri
        self.Ni_to = Ni_to
        self.Ni_from = Ni_from
        self.a = a                          # a stands for alpha
        self.a_old = a_old
        self.lambda_a = lambda_a
        self.lambda_a_old = lambda_a_old
        self.lambda_x = lambda_x
        self.lambda_x_old = lambda_x_old
        self.x = x
        self.x_old = x_old
        self.n = ANi.shape[0]
        self.d = Bi.shape[1]
        self.SS = SS
        self.uSS = uSS
        self.cost = cost
        self.Qfun = Qfun

        self.xcl = xcl
        self.ucl = ucl
        self.SS_vector = []
        self.Qfun_vector = []
        self.xt = []

        self.xPred = []
        self.uPred = []
        self.aPred = []

        self.xADMM1 = []
        self.uADMM1 = []
        self.aADMM1 = []
        self.xADMM1_track = []
        self.uADMM1_track = []
        self.aADMM1_track = []



