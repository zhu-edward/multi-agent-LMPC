import numpy as np
import control
from FTOCP_ADMM import FTOCP_ADMM
from FTOCP import FTOCP
from LMPC_ADMM import LMPC_ADMM
from sysi import sysi
from edgei import edgei
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.linalg import block_diag

def main():
    # Define system dynamics and cost
    A00 = np.array([[0, 0], [0, 0]])
    A11 = np.array([[1,1],[0,1]])
    A22 = np.array([[1, 1], [0, 1]])
    A33 = np.array([[1, 1], [0, 1]])
    A12 = 0.3*np.array([[1, 0.33], [0, 1]])
    #A12 = A00
    #A12 = np.array([[0, 0], [0, 0]])
    A13 = A00
    A21 = A00
    A23 = 0.3*np.array([[1, 0.33], [0, 1]])
    A31 = 0.3*np.array([[1, 0.33], [0, 1]])
    #A31 = A00
    #A31 = np.array([[0, 0], [0, 0]])
    A32 = A00

    AN1 = np.reshape(np.array([A11,A12]),(2,4))
    AN2 = np.reshape(np.array([A22, A23]),(2,4))
    AN3 = np.reshape(np.array([A33,A31]),(2,4))
    ANi_list = [AN1,AN2,AN3]


    B1 = np.array([[0],[1]])
    B2 = np.array([[0], [1]])
    B3 = np.array([[0], [1]])
    Bi_list = [B1,B2,B3]

    Qi = np.diag([1.0, 1.0]) #np.eye(2)
    Ri = 1.0 #np.array([[1]])

    Qi_list = [Qi,Qi,Qi]
    Ri_list = [Ri,Ri,Ri]

    Ni_to1 = np.array([3]) - 1
    Ni_from1 = np.array([2]) - 1
    Ni_to2 = np.array([1]) - 1
    Ni_from2 = np.array([3]) - 1
    Ni_to3 = np.array([2]) - 1
    Ni_from3 = np.array([1]) - 1
    Ni_to = [Ni_to1,Ni_to2,Ni_to3]
    Ni_from = [Ni_from1,Ni_from2,Ni_from3]

    A = np.block([[A11,A12,A13], [A21,A22,A23], [A31,A32,A33]])
    B = block_diag(B1,B2,B3)
    Q = block_diag(Qi,Qi,Qi) #np.eye(2)
    R = block_diag(Ri,Ri,Ri)

    print("Computing first feasible trajectory")

    # Initial Condition
    x10 = [-5.0, 0.0]
    x20 = [-4.0, 0.0]
    x30 = [-3.0, 0.0]
    x0 = np.concatenate((x10,x20,x30))
    x0_list = np.array([x10,x20,x30])

    # Initialize FTOCP object
    N_feas = 30
    ftocp_for_mpc  = FTOCP(N_feas, A, B, 0.1*Q, R)

    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible = []
    xt           = x0
    time         = 0


    # time Loop (Perform the task until close to the origin)
    while np.dot(xt, xt) > 10**(-15):
        xt = xcl_feasible[time] # Read measurements

        ftocp_for_mpc.solve(xt, verbose = False) # Solve FTOCP

        # Read input and apply it to the system
        ut = ftocp_for_mpc.uPred[:,0]#[:]
        ucl_feasible.append(ut)
        xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
        time += 1

    print(np.round(np.array(xcl_feasible).T, decimals=2))
    print(np.round(np.array(ucl_feasible).T, decimals=2))
    # ====================================================================================





    # ====================================================================================
    # Run LMPC
    # ====================================================================================

    # Initialize LMPC object
    N_LMPC = 3 # horizon length
    N = 3


    # Initialize Distributed system structure
    M = 3 # number of agents

    # Initialize ADMM variables
    rho = 1


    # Initialize the subsystems
    #syslist = np.array(M,dtype=object)
    syslist = np.zeros((3,3),dtype=sysi)
    for m in range(M):
        n = ANi_list[m].shape[0]
        lambda_x = np.zeros(n * (N+1))
        lambda_x_old = np.zeros(n * (N+1))
        lambda_a = np.zeros(1)
        lambda_a_old = np.zeros(1)
        a = np.zeros(1)                     # a stands for alpha
        a_old = np.zeros(1)
        x = np.zeros(n * (N+1))
        x_old = np.zeros(n * (N+1))
        SS = []
        uSS = []
        cost = 0
        Qfun = []
        xcl =[]
        ucl=[]
        syslist[m][m] = sysi(ANi_list[m], Bi_list[m], Qi_list[m], Ri_list[m], Ni_to[m], Ni_from[m], a, a_old, lambda_x, lambda_x_old, lambda_a, lambda_a_old, x, x_old, SS, uSS, cost, Qfun, xcl, ucl, [], [], [])

#    edgelist = ADMM_edge_ini(M, syslist)



 #   def ADMM_edge_ini(M, syslist):
        # initialize all edges between the M subsystems in sysi list
 #   edgelist = np.zeros((M, M, M), dtype=object)
    for m in range(M):
        # subsystem m

        for t in syslist[m][m].Ni_from:
        # subsystem t: others dynamics that influence subsystem m
            nx = syslist[t][t].n
            lambda_x = np.zeros(nx * (N+1))         # lambda are used in m problem
            lambda_x_old = np.zeros(nx * (N+1))
            x = np.zeros(nx * (N+1))                # belief of subsystem m over x_t
            x_old = np.zeros(nx * (N+1))

            SS = None
            uSS = None
            cost = 0
            Qfun = []
            xcl = []
            ucl = []

            syslist[m][t] = sysi(np.zeros((1,1)), np.zeros((1,1)), [0], [0], [0], [0], [0], [0], lambda_x, lambda_x_old, [0], [0], x, x_old, SS, uSS, cost, Qfun, xcl, ucl, [], [], [])



    ftocp_ADMM_list = np.zeros((3, ), dtype=FTOCP_ADMM)
    for m in range(M):
        ftocp_ADMM_list[m] = FTOCP_ADMM(N_LMPC, m, syslist)                # ftocp solved by LMPC

    lmpc_ADMM = LMPC_ADMM(ftocp_ADMM_list, syslist, M, CVX=True)        # Initialize the LMPC (decide if you wanna use the CVX hull)

    n_count = 0
    d_count = 0

    xcl_feasible_array = np.asarray(xcl_feasible)
    ucl_feasible_array = np.asarray(ucl_feasible)


    for m in range(M):
        syslist[m][m].xcl = xcl_feasible_array[:, n_count:n_count+syslist[m][m].n]
        syslist[m][m].ucl = ucl_feasible_array[:, d_count:d_count+syslist[m][m].d]

        n_count += syslist[m][m].n
        d_count += syslist[m][m].d

        syslist[m][m].lambda_a = np.zeros(xcl_feasible_array.shape[0])
        syslist[m][m].lambda_a_old = np.zeros(xcl_feasible_array.shape[0])
        syslist[m][m].a = np.zeros(xcl_feasible_array.shape[0])  # a stands for alpha
        syslist[m][m].a_old = np.zeros(xcl_feasible_array.shape[0])

    lmpc_ADMM.addTrajectory(ftocp_ADMM_list, syslist, M)        # xcl_feasible, ucl_feasible)         # Add feasible trajectory to the safe set





    totalIterations = 10 # Number of iterations to perform
    ADMM_iterations = 200

    # run simulation
    # iteration loop
    print("Starting LMPC")
    for it in range(0,totalIterations):
        # Set initial condition at each iteration

    #    for m in range(M):
            #syslist[m][m].xcl = x0_list[m]
            #syslist[m][m].ucl = []
     #       syslist[m][m].xcl = np.asarray(x0_list[m])
      #      syslist[m][m].ucl = []

        print("check here in line 227 whether xcl or xt needs to be initialized... also x0 is dim 2 and later first element is taken and then it is dim 1... ")

        time = 0
        n_count = 0
        for m in range(M):
            syslist[m][m].xcl = [x0[n_count:n_count + syslist[m][m].n]]
            syslist[m][m].xt = syslist[m][m].xcl
            syslist[m][m].ucl = []
            n_count += syslist[m][m].n

        print(["Iteration", it, "xcl", syslist[0][0].xcl, "time", time])

        # time Loop (Perform the task until close to the origin)
        while np.linalg.norm(syslist[m][m].xt) > 10 ** (-10):

            for m in range(M):
                syslist[m][m].xt = syslist[m][m].xcl[time*syslist[m][m].n:(time+1)*syslist[m][m].n]

            print(["Iteration", it, "xcl", syslist[0][0].xcl, "time", time])

            lmpc_ADMM.solve(ftocp_ADMM_list, syslist, M, N, rho, ADMM_iterations, verbose = False)

            # Read optimal input
            # ut = lmpc.uPred[:,0]#[0]

            #print("does not even enter this routine of appending the closed loop x")
            for m in range(M):
                ut = syslist[m][m].u[:, 0]  # [0]
                # Apply optimal input to the system
                if syslist[m][m].ucl == []:
                    syslist[m][m].ucl = np.append(syslist[m][m].ucl,ut)
                    syslist[m][m].xcl = np.append([syslist[m][m].xcl], [lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], syslist[m][m].xt, ut)])
                else:
                    syslist[m][m].ucl = np.stack((syslist[m][m].ucl, ut))
                    syslist[m][m].xcl = np.stack(([syslist[m][m].xcl], [lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], syslist[m][m].xt, ut)]))
            time += 1


            #except:
                # Save the track object
             #   filename = 'track_object.pkl'
              #  filehandler = open(filename, 'wb')
               # pickle.dump(syslist, filehandler)

              #  break
              #  print('broke')


            print(["Iteration", it, "xcl", syslist[0][0].xcl, "time", time])

            if time >= 1:
                halt = 1
        # Add trajectory to update the safe set and value function
        #lmpc.addTrajectory(xcl, ucl)

  #      for m in range(M):
  #          syslist[m][m].x = syslist[m][m].xcl
  #          syslist[m][m].u = syslist[m][m].ucl

        #if np.dot(syslist[m][m].xcl[time], syslist[m][m].xcl[time]) <= 10 ** (-10):
        lmpc_ADMM.addTrajectory(ftocp_ADMM_list, syslist, M)

        halt = 1


    # =====================================================================================

"""
    # ====================================================================================
    # Compute optimal solution by solving a FTOCP with long horizon
    # ====================================================================================
    N = 100 # Set a very long horizon to fake infinite time optimal control problem
    ftocp_opt = FTOCP(N, A, B, Q, R)
    ftocp_opt.solve(xcl[0])
    xOpt = ftocp_opt.xPred
    uOpt = ftocp_opt.uPred
    costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
    print("Optimal cost is: ", costOpt[0])
    # Store optimal solution in the lmpc object
    lmpc.optCost = costOpt[0]
    lmpc.xOpt    = xOpt
"""
"""
    # Save the lmpc object
filename = 'lmpc_object.pkl'
filehandler = open(filename, 'wb')
pickle.dump(lmpc, filehandler)
"""
if __name__== "__main__":
  main()
