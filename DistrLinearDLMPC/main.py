import numpy as np
from FTOCP_ADMM import FTOCP_ADMM
from FTOCP import FTOCP
from LMPC_ADMM import LMPC_ADMM
from sysi import sysi
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import copy, pdb
import os, sys, pickle, argparse

os.environ['TZ'] = 'America/Los_Angeles'
#time.tzset()
FILE_DIR =  os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')))
BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-2]))
sys.path.append(BASE_DIR)

from scipy.linalg import block_diag
#from utils.lmpc_visualizer import lmpc_visualizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_traj', action='store_true', help='Use trajectory from file', default=False)
    args = parser.parse_args()

    out_dir = '/'.join((BASE_DIR, 'out'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_dir = '/'.join((BASE_DIR, 'logs'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Flags
    plot_init = False  # Plot initial trajectory
    pause_each_solve = False  # Pause on each FTOCP solution

    plot_lims = [[-10, 10], [-10, 10]]
    tol = -4

    # Initialize LMPC object
    N_LMPC = 3 # horizon length
    N = 3


    # Initialize Distributed system structure
    M = 3 # number of agents

    # Initialize ADMM variables
    rho = 1

    # Define system dynamics and cost
    A00 = np.array([[0, 0], [0, 0]])
    A11 = np.array([[1,1],[0,1]])
    A22 = np.array([[1, 1], [0, 1]])
    A33 = np.array([[1, 1], [0, 1]])
    #A12 = 0.3*np.array([[1, 0.33], [0, 1]])
    A12 = np.array([[0.3, 0.1], [0, 0.3]])
    #A12 = A00
    #A12 = np.array([[0, 0], [0, 0]])
    A13 = A00
    A21 = A00
    #A23 = 0.3*np.array([[1, 0.33], [0, 1]])
    #A31 = 0.3*np.array([[1, 0.33], [0, 1]])
    A23 = np.array([[0.3, 0.1], [0, 0.3]])
    A31 = np.array([[0.3, 0.1], [0, 0.3]])
    #A31 = A00
    #A31 = np.array([[0, 0], [0, 0]])
    A32 = A00

    AN1 = np.concatenate([A11,A12],axis=1) #np.reshape(np.array([A11,A12]),(2,4))
    AN2 = np.concatenate([A22,A23],axis=1) #np.reshape(np.array([A22, A23]),(2,4))
    AN3 = np.concatenate([A33,A31],axis=1) #np.reshape(np.array([A33,A31]),(2,4))
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

    # Initial Condition
    #x10 = [-5.0, 0.0]
    #x20 = [-4.0, 0.0]
    #x30 = [-3.0, 0.0]
    x10 = [-10.0, 1.0]
    x20 = [-9.0, 1.0]
    x30 = [-8.0, 1.0]
    x0 = np.concatenate((x10,x20,x30))
    x0_list = np.array([x10,x20,x30])




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





    if False: #not args.init_traj:

        print("Computing first feasible trajectory")

        # ====================================================================================
        # Run LTV MPC to compute feasible solutions for all agents
        # ====================================================================================
        # Goal conditions (these will be updated once the initial trajectories are found)
        #x_f = [np.nan * np.ones((n_x, 1)) for _ in range(n_a)]
        #x_f[0] = np.array([0.0, -5.0, 3 * np.pi / 2, 0.0])
        #x_f[1] = np.array([5.0, 5.0, np.pi / 4, 0.0])
        #x_f[2] = np.array([-5.0, 5.0, 3 * np.pi / 4, 0.0])
        x1f = [0.0, 0.0]
        x2f = [0.0, 0.0]
        x3f = [0.0, 0.0]
        xf = np.concatenate((x1f, x2f, x3f))
        xf_list = np.array([x1f, x2f, x3f])



        # Initialize FTOCP object
        #N_feas = 30
        N_feas = 15
        ftocp_for_mpc  = FTOCP(N_feas, A, B, 0.01*Q, R)

        #mpc_vis = [lmpc_visualizer(pos_dims=[0, 1], n_state_dims=2, n_act_dims=1, agent_id=m, n_agents=M,
         #                      plot_lims=plot_lims) for m in range(M)]

    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
        xcl_feasible = [x0]
        ucl_feasible = []
        xt           = x0
        time         = 0


        # time Loop (Perform the task until close to the origin)
        while np.dot(xt, xt) > 10**(-4):
            xt = xcl_feasible[time] # Read measurements

            ftocp_for_mpc.solve(xt, verbose = False) # Solve FTOCP

            # Read input and apply it to the system
            ut = ftocp_for_mpc.uPred[:,0]#[:]
            ucl_feasible.append(ut)
            xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
            time += 1
            print(xt)

        print(np.round(np.array(xcl_feasible).T, decimals=2))
        print(np.round(np.array(ucl_feasible).T, decimals=2))
    # ====================================================================================
        n_count = 0
        d_count = 0
        nx = A11.shape[0]
        #nu = B1.shape[1]

        xcl_feasible_array = np.asarray(xcl_feasible)
        ucl_feasible_array = np.asarray(ucl_feasible)

        for m in range(M):
            syslist[m][m].xcl = xcl_feasible_array[:, n_count:n_count + syslist[m][m].n]
            syslist[m][m].ucl = ucl_feasible_array[:, d_count:d_count + syslist[m][m].d]
            n_count += syslist[m][m].n
            d_count += syslist[m][m].d

        for m in range(M):
            xcl_feas = syslist[m][m].xcl
            ucl_feas = syslist[m][m].ucl

            # Save initial trajecotry if file doesn't exist
            if not os.path.exists('/'.join((FILE_DIR, 'init_traj_%i.npz' % m))):
                print('Saving initial trajectory for agent %i' % (m + 1))
                np.savez('/'.join((FILE_DIR, 'init_traj_%i.npz' % m)), x=xcl_feas, u=ucl_feas)

            #mpc_vis[m].close_figure()

        del ftocp_for_mpc #, mpc_vis

    else:
        # Load initial trajectory from file
#        x_f = [np.nan * np.ones((nx, 1)) for _ in range(M)]
        xcl_feas = []
        ucl_feas = []
        for m in range(M):
            init_traj = np.load('/'.join((FILE_DIR, 'init_traj_%i.npz' % m)), allow_pickle=True)
            xcl_feas.append(init_traj['x'])
            ucl_feas.append(init_traj['u'])

            syslist[m][m].xcl = np.asarray(xcl_feas[0])
            syslist[m][m].ucl = np.asarray(ucl_feas[0])









    # ====================================================================================
    # Run LMPC
    # ====================================================================================
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

#    n_count = 0
#    d_count = 0

#    xcl_feasible_array = np.asarray(xcl_feasible)
#    ucl_feasible_array = np.asarray(ucl_feasible)


    for m in range(M):
#        syslist[m][m].xcl = xcl_feasible_array[:, n_count:n_count+syslist[m][m].n]
#        syslist[m][m].ucl = ucl_feasible_array[:, d_count:d_count+syslist[m][m].d]

#        n_count += syslist[m][m].n
#        d_count += syslist[m][m].d
        xcl_feas = syslist[m][m].xcl

        syslist[m][m].lambda_a = np.zeros(xcl_feas.shape[0])
        syslist[m][m].lambda_a_old = np.zeros(xcl_feas.shape[0])
        syslist[m][m].a = np.zeros(xcl_feas.shape[0])  # a stands for alpha
        syslist[m][m].a_old = np.zeros(xcl_feas.shape[0])

    lmpc_ADMM.addTrajectory(ftocp_ADMM_list, syslist, M)        # xcl_feasible, ucl_feasible)         # Add feasible trajectory to the safe set

    # pdb.set_trace()

    totalIterations = 10 # Number of iterations to perform
    ADMM_iterations = 500
    # ADMM_iterations = 10

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


        time = 0
        n_count = 0
        for m in range(M):
            syslist[m][m].xcl = x0[n_count:n_count + syslist[m][m].n]
            syslist[m][m].xt = syslist[m][m].xcl
            syslist[m][m].ucl = []
            n_count += syslist[m][m].n

        # print(["Iteration", it, "xcl", syslist[0][0].xcl, "time", time])

        # time Loop (Perform the task until close to the origin)
        while np.linalg.norm(syslist[m][m].xt) > 10 ** (-4):

            for m in range(M):
                syslist[m][m].xt = syslist[m][m].xcl[time*syslist[m][m].n:(time+1)*syslist[m][m].n]

            print(["Iteration", it, "time", time, "xt", syslist[0][0].xt])

            print('Saving tracking')
            np.savez('/'.join((FILE_DIR, 'tracksyslist.npz')), syslist=syslist)

 #           #FOR PLOTTING
 #           track = np.load('/'.join((FILE_DIR, 'tracksyslist.npz')), allow_pickle=True)
 #           tracksyslist = track['syslist']



            lmpc_ADMM.solve(ftocp_ADMM_list, syslist, M, N, rho, ADMM_iterations, verbose = False)

            for m in range(M):
                ut = syslist[m][m].uPred[:,0]  # [0]

                xt = syslist[m][m].xt
                for t in syslist[m][m].Ni_from:
                    xt = np.append(xt, syslist[t][t].xt)

                # Apply optimal input to the system
                if syslist[m][m].ucl == []:
                    syslist[m][m].ucl = np.append(syslist[m][m].ucl,ut)
                    # syslist[m][m].xcl = np.append([syslist[m][m].xcl], [lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], syslist[m][m].xt, ut)])
                    syslist[m][m].xcl = np.append([syslist[m][m].xcl], [lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], xt, ut)])
                    #syslist[m][m].xcl = np.concatenate([syslist[m][m].xcl[0], lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], xt, ut)], axis = 0)
                else:
                    #syslist[m][m].ucl = np.stack((syslist[m][m].ucl, ut))
                    syslist[m][m].ucl = np.append(syslist[m][m].ucl, ut)
                    # syslist[m][m].xcl = np.stack(([syslist[m][m].xcl], [lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], syslist[m][m].xt, ut)]))
                    syslist[m][m].xcl = np.append(syslist[m][m].xcl, lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], xt, ut))
                    #syslist[m][m].xcl = np.stack(([syslist[m][m].xcl], [lmpc_ADMM.ftocp_ADMM[m].model(syslist[m][m], xt, ut)]))
            time += 1


            #except:
                # Save the track object
              # filename = 'track_object.pkl'
              #filehandler = open(filename, 'wb')
             #pickle.dump(syslist, filehandler)
          #  if not os.path.exists('/'.join((FILE_DIR, 'track.npz'))):
            print('Saving tracking')
            np.savez('/'.join((FILE_DIR, 'tracksyslist')), syslist=syslist)

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
