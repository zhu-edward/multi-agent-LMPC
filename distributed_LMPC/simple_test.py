import numpy as np
#import control
from FTOCP import FTOCP
from LMPC import LMPC
from sysi import sysi
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
    A12 = 0.5*np.array([[1, 1], [0, 1]])
    #A12 = A00
    #A12 = np.array([[0, 0], [0, 0]])
    A13 = A00
    A21 = A00
    A23 = 0.5*np.array([[1, 1], [0, 1]])
    A31 = 0.5*np.array([[1, 1], [0, 1]])
    #A31 = A00
    #A31 = np.array([[0, 0], [0, 0]])
    A32 = A00

    AN1 = np.reshape(np.array([A11,A12]),(2,4))
    AN2 = np.reshape(np.array([A22, A23]),(2,4))
    AN3 = np.reshape(np.array([A31,A33]),(2,4))
    ANi = [AN1,AN2,AN3]


    B1 = np.array([[0],[1]])
    B2 = np.array([[0], [1]])
    B3 = np.array([[0], [1]])
    Bi = [B1,B2,B3]

    Qi = np.diag([1.0, 1.0]) #np.eye(2)
    Ri = 1.0#np.array([[1]])

    Q1 = Qi
    Q2 = Qi
    Q3 = Qi
    R1 = Ri
    R2 = Ri
    R3 = Ri

    Ni_to1 = [3]
    Ni_from1 = [2]
    Ni_to2 = [1]
    Ni_from2 = [3]
    Ni_to3 = [2]
    Ni_from3 = [1]
    Ni_to = [Ni_to1,Ni_to2,Ni_to3]
    Ni_from = [Ni_from1,Ni_from2,Ni_from3]
    alphai = 0
    lambdai = 0
    alphai_old = 0
    lambdai_old = 0


    A = np.block([[A11,A12,A13], [A21,A22,A23], [A31,A32,A33]])
    B = block_diag(B1,B2,B3)
    Q = block_diag(Qi,Qi,Qi) #np.eye(2)
    R = block_diag(Ri,Ri,Ri)

    # ctrb_check = control.ctrb(A,B)
    # print(np.linalg.matrix_rank(ctrb_check))

    print("Computing first feasible trajectory")

    # Initial Condition
    x10 = [-5.0, 0.0]
    x20 = [-4.0, 0.0]
    x30 = [-3.0, 0.0]
    x0 = np.concatenate((x10,x20,x30))

    # Initialize FTOCP object
    N_feas = 10
    ftocp_for_mpc  = FTOCP(N_feas, A, B, 0.01*Q, R)

    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible =[]
    xt           = x0
    time         = 0


    # time Loop (Perform the task until close to the origin)
  #  while np.dot(xt, xt) > 10**(-15):
  #      xt = xcl_feasible[time] # Read measurements

#        ftocp_for_mpc.solve(xt, verbose = False) # Solve FTOCP

        # Read input and apply it to the system
 #       ut = ftocp_for_mpc.uPred[:,0]#[:]
  #      ucl_feasible.append(ut)
   #     xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
    #    time += 1

 #   print(np.round(np.array(xcl_feasible).T, decimals=2))
  #  print(np.round(np.array(ucl_feasible).T, decimals=2))
    # ====================================================================================





    # ====================================================================================
    # Run LMPC
    # ====================================================================================

    # Initialize LMPC object
    N_LMPC = 3 # horizon length

    # Initialize Distributed system structure
    M = 3 # number of agents

    # Initialize ADMM variables
    lambdai = 0
    lambdai_old = 0

    alphai = 0
    alphai_old = 0

    rho = 1


    # Initialize the subsystems
    syslist = []
    for im in range(M):
        syslist += [sysi(ANi[im], Bi[im], Qi, Ri, Ni_to[im], Ni_from[im], alphai, lambdai, alphai_old, lambdai_old)]

#    for im in range(M):
#        sysi_(im) = syslist[im]
#    for

 #   for i in range(len(test_cases)):
  #      print
   #     test_cases[i].indent




    ftocp = FTOCP(N_LMPC, A, B, Q, R) # ftocp solved by LMPC
    lmpc = LMPC(ftocp, CVX=True) # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible) # Add feasible trajectory to the safe set

    totalIterations = 20 # Number of iterations to perform

    # run simulation
    # iteration loop
    print("Starting LMPC")
    for it in range(0,totalIterations):
        # Set initial condition at each iteration
        xcl = [x0]
        ucl =[]
        time = 0
        # time Loop (Perform the task until close to the origin)
        while np.dot(xcl[time], xcl[time]) > 10**(-10):

            # Read measurement
            xt = xcl[time]

            # Solve FTOCP
            lmpc.solve(xt, verbose = False)
            # Read optimal input
            ut = lmpc.uPred[:,0]#[0]

            # Apply optimal input to the system
            ucl.append(ut)
            xcl.append(lmpc.ftocp.model(xcl[time], ut))
            time += 1

        # Add trajectory to update the safe set and value function
        lmpc.addTrajectory(xcl, ucl)

    # =====================================================================================


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

    # Save the lmpc object
    filename = 'lmpc_object.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(lmpc, filehandler)

if __name__== "__main__":
  main()
