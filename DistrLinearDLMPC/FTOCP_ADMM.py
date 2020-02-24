import numpy as np
import pdb
import scipy
from cvxpy import *
import pickle

class FTOCP_ADMM(object):
    """ Finite Time Optimal Control Problem (FTOCP)
    Methods:
        - solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
        - model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

    """
    def __init__(self, N, m, syslist):
        # Define variables
        self.N = N # Horizon Length

        # System Dynamics (x_{k+1} = A x_k + Bu_k)
        self.A = syslist[m][m].ANi
        self.B = syslist[m][m].Bi
        self.n = syslist[m][m].ANi.shape[0]
        self.d = syslist[m][m].Bi.shape[1]

        # Cost (h(x,u) = x^TQx +u^TRu)
        self.Q = syslist[m][m].Qi
        self.R = syslist[m][m].Ri

        # Neighboring subsystems
        self.Ni_from = syslist[m][m].Ni_from
        self.Ni_to = syslist[m][m].Ni_to

        self.Ni_from_to = np.union1d(syslist[m][m].Ni_from,syslist[m][m].Ni_to)
        self.SS = syslist[m][m].SS
        self.Qfun = syslist[m][m].Qfun



        # Initialize Predicted Trajectory
        self.x = []
        self.u = []

        self.it = 0
        self.zt = []

    def solve_ADMM1(self, x0, syslist, m, N, rho, verbose = False, SS = None, Qfun = None, CVX = None):
    #    syslist[m][m].xt, syslist, m, N, rho, verbose, syslist[m][m].SS_vector, syslist[m][m].Qfun_vector, self.CVX)


        """This methos solve a FTOCP given:
            - x0: initial condition
            - SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
            - Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
        """
        x0 = syslist[m][m].xt[0:syslist[m][m].n]

        # Initialize Variables
        Ni_from_to = np.union1d(syslist[m][m].Ni_from, syslist[m][m].Ni_to)

        n_Ni_from = 0
        for t in syslist[m][m].Ni_from:
            n_Ni_from += syslist[t][t].n

        xVar = Variable((syslist[m][m].n+n_Ni_from, N+1))
        uVar = Variable((syslist[m][m].d, N))

        # If SS is given initialize lambdaVar multipliers used to encorce terminal constraint
        SS_ = syslist[m][m].SS
        SS = np.asarray(SS_)[0]
        if SS is not None:
            if CVX == True:
                #alphaVar = Variable((SS.shape[1], 1), boolean=False) # Initialize vector of variables
                alphaVar = Variable((SS.shape[0]), boolean=False)  # Initialize vector of variables
            else:
                #alphaVar = Variable((SS.shape[1], 1), boolean=True) # Initialize vector of variables
                alphaVar = Variable((SS.shape[0]), boolean=True)  # Initialize vector of variables

        Qfun = syslist[m][m].Qfun

        # State Constraints
        constr = [xVar[0:syslist[m][m].n,0] == x0[0]]
        #for t in syslist[m][m].Ni_from:
        #    constr += [xVar[syslist[m][m].n:syslist[m][m].n+syslist[t][t].n, 0] == syslist[t][t].xt[0:syslist[t][t].n]]
        for i in range(0, N):
            constr += [xVar[0:syslist[m][m].n,i+1] == syslist[m][m].ANi*xVar[:,i] + syslist[m][m].Bi*uVar[:,i],
                        uVar[:,i] >= -1.0,
                        uVar[:,i] <=  1.0,
                        xVar[:,i] >= -700.0,
                        xVar[:,i] <=  700.0,]

        # Terminal Constraint if SS not empty --> enforce the terminal constraint
        if SS is not None:
            constr += [SS.T * alphaVar[:] == xVar[0:syslist[m][m].n,N], # Terminal state \in ConvHull(SS)
            #constr += [np.dot(SS.T, alphaVar) == xVar[0:syslist[m][m].n, N],
                        np.ones((1, SS.shape[0])) * alphaVar[:] == 1, # Multiplies \lambda sum to 1
                        alphaVar >= 0] # Multiplier are positive definite



        # Cost Function
        cost = 0
        for i in range(0, N):
            # Running cost h(x,u) = x^TQx + u^TRu
            cost += quad_form(xVar[0:syslist[m][m].n, i], syslist[m][m].Qi) + norm(syslist[m][m].Ri**0.5*uVar[:,i])**2

        # Terminal cost if SS not empty

        if SS is not None:
            #cost += syslist[m][m].Qfun[0, :] * alphaVar[:, 0]  # It terminal cost is given by interpolation using \lambda
            cost += sum([x*y for x,y in zip(syslist[m][m].Qfun[0],alphaVar)])
        else:
            cost += quad_form(xVar[0:syslist[m][m].n, N], syslist[m][m].Qi) + norm(syslist[m][m].Ri ** 0.5 * uVar[:, N-1]) ** 2


        # Terminal cost if SS not empty
        #if SS is not None:
        #    cost += Qfun[0, :] * lambVar[:, 0]  # It terminal cost is given by interpolation using \lambda
        #else:
            # cost += norm(self.Q**0.5*x[:,self.N])**2 # If SS is not given terminal cost is quadratic
        #    cost += quad_form(x[:, i], self.Q) + quad_form(u[:, i], self.R)




        # ADMM cost terms
        lambda_x = syslist[m][m].lambda_x
        lambda_a = syslist[m][m].lambda_a
        # for t in syslist[m][m].Ni_from:
        #     lambda_x = np.append(lambda_x , syslist[m][t].lambda_x)

        # xVar_flat = vec(xVar)
        # cost += lambda_x * xVar.flatten()
        # cost += lambda_x * xVar_flat
        cost += lambda_a * alphaVar

        nt_count = syslist[m][m].n
        for t in syslist[m][m].Ni_from:
            # cost += (rho) * norm( xVar[nt_count:nt_count+syslist[t][t].n,:].flatten() - (syslist[m][t].x_old.flatten() + syslist[t][t].x_old.flatten())/2)**2
            cost += (rho) * norm(vec(xVar[nt_count:nt_count+syslist[t][t].n,:]) - (syslist[m][t].x_old.flatten(order='F') + syslist[t][t].x_old.flatten(order='F'))/2)**2
            # cost += syslist[m][t].lambda_x*(vec(xVar[nt_count:nt_count+syslist[t][t].n,:]) - (syslist[m][t].x_old.flatten(order='F') + syslist[t][t].x_old.flatten(order='F')))
            cost += syslist[m][t].lambda_x*vec(xVar[nt_count:nt_count+syslist[t][t].n,:])
        #    for k in syslist[t][t].Ni_to:
        #        cost += (rho) * norm( xVar[nt_count:nt_count+syslist[t][t].n,:].flatten() - (syslist[m][t].x_old.flatten() + syslist[k][t].x_old.flatten())/2)**2
            nt_count = nt_count + syslist[t][t].n

        for t in syslist[m][m].Ni_to:
            # cost += (rho) * norm( xVar[0:syslist[m][m].n,:].flatten() - (syslist[m][m].x_old.flatten() + syslist[t][m].x_old.flatten())/2)**2
            cost += (rho) * norm( vec(xVar[0:syslist[m][m].n,:]) - (syslist[m][m].x_old.flatten(order='F') + syslist[t][m].x_old.flatten(order='F'))/2)**2
            # cost += lambda_x*(vec(xVar[0:syslist[m][m].n,:]) - (syslist[m][m].x_old.flatten(order='F') + syslist[t][m].x_old.flatten(order='F')))

        cost += lambda_x*vec(xVar[0:syslist[m][m].n,:])

        # make one list out of Ni_from and Ni_to without redundant terms!
        for t in Ni_from_to:
            cost += (rho) * norm(alphaVar - (syslist[m][m].a_old + syslist[t][t].a_old)/2)**2




# Solve the Finite Time Optimal Control Problem
        problem = Problem(Minimize(cost), constr)
        if CVX == True:
            try:
                problem.solve(verbose=False, solver=ECOS) # I find that ECOS is better please use it when solving QPs
            except:
                halt = 1
        else:
            problem.solve(verbose=True)


        # Store the open-loop predicted trajectory
        x = xVar.value
        u = uVar.value
        alpha = alphaVar.value

        syslist[m][m].xADMM1 = x
        syslist[m][m].uADMM1 = u
        syslist[m][m].aADMM1 = alpha

        # syslist[m][m].xPred = x[0:syslist[m][m].n,:].flatten(order='F')
        syslist[m][m].xPred = x[0:syslist[m][m].n,:]
        syslist[m][m].uPred = u
        syslist[m][m].aPred = alpha

        nx_count = syslist[m][m].n
        for t in syslist[m][m].Ni_from:
            # syslist[m][t].xPred = x[nx_count:nx_count+syslist[t][t].n, :].flatten(order='F')
            syslist[m][t].xPred = x[nx_count:nx_count+syslist[t][t].n, :]
            nx_count += syslist[t][t].n

  #      try:
  #          syslist[m][m].xADMM1_track = [syslist[m][m].xADMM1_track, x]
  #          syslist[m][m].uADMM1_track = [syslist[m][m].uADMM1_track, u]
  #          syslist[m][m].aADMM1_track = [syslist[m][m].aADMM1_track, alpha]
  #      except:
  #          syslist[m][m].xADMM1_track = [x]
  #          syslist[m][m].uADMM1_track = [u]
  #          syslist[m][m].aADMM1_track = [alpha]

        return syslist




    def update_ADMM1(self, syslist, m):
        # THIS NEEDS TO BE DONE IN A SECOND STEP ONLY ONCE ALL THE SUBSYSTEMS SOLVED THEIR ADMM 1 SUBPROBLEMS
        # HOW TO STORE THE x AND u VALUES FIRST??


        syslist[m][m].x_old = syslist[m][m].x
        syslist[m][m].x = syslist[m][m].xPred

        syslist[m][m].a_old = syslist[m][m].a
        syslist[m][m].a = syslist[m][m].aPred

        for t in syslist[m][m].Ni_from:
            syslist[m][t].x_old = syslist[m][t].x
            syslist[m][t].x = syslist[m][t].xPred



        """
        x = syslist[m][m].xADMM1
        u = syslist[m][m].uADMM1

        syslist[m][m].x_old = syslist[m][m].x
        syslist[m][m].u = u

        try:
            syslist[m][m].x = x[0:syslist[m][m].n,:]
        except:
            halt = 1

        nt_count = syslist[m][m].n
        for t in syslist[m][m].Ni_from:
            syslist[m][t].x_old = syslist[m][t].x
            try:
                syslist[m][t].x = x[nt_count:nt_count + syslist[t][t].n,:]
            except:
                halt = 1
                # Save the lmpc object
                filename = 'ADMM_track_syslist.pkl'
                filehandler = open(filename, 'wb')
                pickle.dump(syslist, filehandler)

            nt_count: nt_count + syslist[t][t].n

        #if SS is not None:
        alpha = syslist[m][m].aADMM1
        syslist[m][m].a_old = syslist[m][m].a
        syslist[m][m].a = alpha

    #    print(alpha.sum()    )
    #    print(x[0:syslist[m][m].n,0] == x0[:])
    #    print(["Error in final state:", np.dot(SS.T, alpha) - x[0:syslist[m][m].n,N]])
    #    print(["Final state:", np.dot(SS.T, alpha)])
    #    print(["Error in initial value:", x[0:syslist[m][m].n,0] - x0[:]])

    #    if np.linalg.norm(np.subtract(x[0:syslist[m][m].n,0], x0[:])) >= 0.0001:
    #        halt = 1

    #    t = syslist[m][m].Ni_from[0]
    #    if np.linalg.norm(np.subtract(x[syslist[m][m].n:syslist[m][m].n+syslist[t][t].n, 0], syslist[t][t].xt[0:syslist[t][t].n])) >= 0.0001:
    #        halt = 1

        return syslist
        """

        return syslist




    def solve_ADMM2(self, syslist, m, rho):
        """This methos solve a FTOCP given:
            - x0: initial condition
            - SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
            - Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
        """


        Ni_from_to = np.union1d(syslist[m][m].Ni_from, syslist[m][m].Ni_to)

        syslist[m][m].lambda_x_old = syslist[m][m].lambda_x
        lambda_x = syslist[m][m].lambda_x
        for t in syslist[m][m].Ni_to:
            # lambda_x +=  rho * (syslist[m][m].x_old.flatten() - syslist[t][m].x_old.flatten())
            lambda_x +=  rho * (syslist[m][m].x_old.flatten(order='F') - syslist[t][m].x_old.flatten(order='F'))
            #lambda_x += rho * (syslist[m][m].x.flatten() - syslist[t][m].x.flatten())
        #syslist[m][m].lambda_x_old = syslist[m][m].lambda_x
        syslist[m][m].lambda_x = lambda_x



        syslist[m][m].lambda_a_old = syslist[m][m].lambda_a
        lambda_a = syslist[m][m].lambda_a
        for t in Ni_from_to:
            lambda_a += rho * (syslist[m][m].a_old - syslist[t][t].a_old)
            #lambda_a += rho * (syslist[m][m].a - syslist[t][t].a)
        #syslist[m][m].lambda_a_old = syslist[m][m].lambda_a
        syslist[m][m].lambda_a = lambda_a


        for t in syslist[m][m].Ni_from:
            syslist[m][t].lambda_x_old = syslist[m][t].lambda_x
            lambda_x_t = syslist[m][t].lambda_x
            # lambda_x_t +=  rho * (syslist[m][t].x_old.flatten() - syslist[t][t].x_old.flatten())
            lambda_x_t +=  rho * (syslist[m][t].x_old.flatten(order='F') - syslist[t][t].x_old.flatten(order='F'))
            #lambda_x_t += rho * (syslist[m][t].x.flatten() - syslist[t][t].x.flatten())
        #    for k in syslist[t][t].Ni_to:
        #        lambda_x_t += rho * (syslist[m][t].x_old.flatten() - syslist[k][t].x_old.flatten())

            #syslist[m][t].lambda_x_old = syslist[m][t].lambda_x
            syslist[m][t].lambda_x = lambda_x_t

        return syslist






    def model(self, sysm, x, u):
        # Compute state evolution
        return (np.dot(sysm.ANi,x) + np.squeeze(np.dot(sysm.Bi,u)))  #.tolist()
