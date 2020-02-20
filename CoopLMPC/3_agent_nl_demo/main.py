from __future__ import division

import numpy as np
import numpy.linalg as la
import multiprocessing as mp

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.pyplot as plt

import os, sys, time, copy, pickle, itertools, pdb, argparse

os.environ['TZ'] = 'America/Los_Angeles'
time.tzset()
FILE_DIR =  os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')))
BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-2]))
sys.path.append(BASE_DIR)

from LTV_FTOCP import LTV_FTOCP
from NL_FTOCP import NL_FTOCP
from NL_LMPC import NL_LMPC
from agents import DT_Kin_Bike_Agent

from utils.plot_bike_utils import plot_bike_agent_trajs
from utils.lmpc_visualizer import lmpc_visualizer
from utils.safe_set_utils import get_safe_set, inspect_safe_set
import utils.utils

def solve_init_traj(ftocp, x_0, model_agent, tol=-7):
	n_x = ftocp.n_x
	n_u = ftocp.n_u

	xcl_feas = [x_0]
	ucl_feas = []
	t_span = []

	model_dt = model_agent.dt
	control_dt = ftocp.agent.dt
	n_control = int(np.around(control_dt/model_dt))

	waypts = ftocp.get_x_refs()
	waypt_idx = ftocp.get_reference_idx()

	t = 0
	counter = 0

	# time Loop (Perform the task until close to the origin)
	while True:
		x_t = xcl_feas[-1] # Read measurements

		if np.mod(counter, n_control) == 0:
			x_pred, u_pred = ftocp.solve(x_t, t, verbose=False)
			u_t = u_pred[:,0]
			print('t: %g, d: %g, x: %g, y: %g, phi: %g, v: %g' % (t, la.norm(x_t[:2] - waypts[waypt_idx][:2]), x_t[0], x_t[1], x_t[2]*180.0/np.pi, x_t[3]))

		# Read input and apply it to the system
		x_tp1 = model_agent.sim(x_t, u_t)

		ucl_feas.append(u_t)
		xcl_feas.append(x_tp1)

		# Close within tolerance
		d = la.norm(x_tp1[:2] - waypts[waypt_idx][:2])
		v = x_tp1[3] - waypts[waypt_idx][3]
		if d <= 0.5 and waypt_idx < len(waypts)-1:
			print('Waypoint %i reached' % waypt_idx)
			ftocp.advance_reference_idx()
			waypt_idx = ftocp.get_reference_idx()
		elif d <= 1.0 and np.abs(v) <= 10**tol and waypt_idx == len(waypts)-1:
			print('Goal state reached')
			break

		t += model_dt
		counter += 1

	xcl_feas = np.array(xcl_feas).T
	ucl_feas = np.array(ucl_feas).T

	return xcl_feas, ucl_feas

def solve_lmpc(lmpc, x_0, x_f, model_agent, verbose=False, visualizer=None, pause=False, tol=-7):
	n_x = lmpc.n_x
	n_u = lmpc.n_u

	x_pred_log = []
	u_pred_log = []

	xcl = x_0.reshape((-1,1)) # initialize system state
	ucl = np.empty((n_u,0))

	inspect = False

	t = 0
	# time Loop (Perform the task until close to the origin)
	while True:
		x_t = xcl[:,t] # Read measurements
		x_pred, u_pred, cost, SS, N = lmpc.solve(t, x_t, x_f, tol, verbose=verbose) # Solve FTOCP
		# Inspect incomplete trajectory
		if x_pred is None or u_pred is None and visualizer is not None:
			# visualizer.traj_inspector(xcl, ucl, x_pred_log, u_pred_log, t, lmpc.SS_t, lmpc.expl_constrs)
			sys.exit()
		else:
			x_pred_log.append(x_pred)
			u_pred_log.append(u_pred)

		u_t = u_pred[:,0]
		# Read input and apply it to the system
		x_tp1 = model_agent.sim(x_t, u_t)

		ucl = np.append(ucl, u_t.reshape((-1,1)), axis=1)
		xcl = np.append(xcl, x_tp1.reshape((-1,1)), axis=1)

		print('ts: %g, d: %g, x: %g, y: %g, phi: %g, v: %g' % (t, la.norm(x_tp1-x_f, ord=2), x_t[0], x_t[1], x_t[2]*180.0/np.pi, x_t[3]))

		if visualizer is not None:
			visualizer.plot_traj(xcl, ucl, x_pred, u_pred, t, SS, N, lmpc.expl_constrs, shade=True)

		if la.norm(x_tp1-x_f, ord=2) <= 10**tol:
			print('Tolerance reached, agent reached goal state')
			break

		if pause:
			pdb.set_trace()

		t += 1

	# Inspection mode after iteration completion
	# if inspect:
		# visualizer.traj_inspector(xcl, ucl, x_pred_log, u_pred_log, t, lmpc.SS_t, lmpc.expl_constrs)

	# print np.round(np.array(xcl).T, decimals=2) # Uncomment to print trajectory
	return xcl, ucl

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
	parallel = False # Parallelization flag
	plot_init = False # Plot initial trajectory
	pause_each_solve = False # Pause on each FTOCP solution

	plot_lims = [[-10, 10], [-10, 10]]
	tol = -4

	model_dt = 0.1
	control_dt = 0.1

	n_a = 3 # Number of agents
	n_x = 4 # State dimension
	n_u = 2 # Input dimension

	# Car dimensions
	l_r = 0.5
	l_f = 0.5
	w = 0.5

	# Initial Conditions
	x_0 = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
	x_0[0] = np.array([0.0, 5.0, np.pi, 0.0])
	x_0[1] = np.array([-5.0, -5.0, 0.0, 0.0])
	x_0[2] = np.array([5.0, -5.0, np.pi/2.0, 0.0])

	# Initialize dynamics and control agents (allows for dynamics to be simulated with higher resolution than control rate)
	model_agents = [DT_Kin_Bike_Agent(l_r, l_f, w, model_dt, x_0[i]) for i in range(n_a)]
	mpc_control_agents = [DT_Kin_Bike_Agent(l_r, l_f, w, control_dt, x_0[i], a_lim=[-1.0, 1.0], df_lim=[-0.5, 0.5], da_lim=[-1.5, 1.5], ddf_lim=[-0.3, 0.3]) for i in range(n_a)]
	lmpc_control_agents = [DT_Kin_Bike_Agent(l_r, l_f, w, control_dt, x_0[i]) for i in range(n_a)]

	if not args.init_traj:
		# ====================================================================================
		# Run LTV MPC to compute feasible solutions for all agents
		# ====================================================================================
		# Goal conditions (these will be updated once the initial trajectories are found)
		x_f = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
		x_f[0] = np.array([0.0, -5.0, 2*np.pi, 0.0])
		x_f[1] = np.array([5.0, 5.0, np.pi/2, 0.0])
		x_f[2] = np.array([-5.0, 5.0, np.pi, 0.0])

		# Check to make sure all agent dynamics, inital, and goal states have been defined
		if np.any(np.isnan(x_0)) or np.any(np.isnan(x_f)):
			raise(ValueError('Initial or goal states have empty entries'))

		# Intermediate waypoint to ensure collision-free trajectory
		waypts = [[] for _ in range(n_a)]
		# waypts[0] = [np.array([-5.0, 0.0, 3.0*np.pi/2.0, 1.0])]
		# waypts[1] = [np.array([0.0, -5.0, 0.0, 1.0])]
		# waypts[2] = [np.array([5.0, 0.0, np.pi/2, 1.0])]
		waypts[0] = [np.array([-5.0, 0.0, 3.0*np.pi/2.0, 0.5])]
		waypts[1] = [np.array([0.0, -5.0, 0.0, 0.5])]
		waypts[2] = [np.array([5.0, 0.0, np.pi/2, 0.5])]
		for i in range(n_a):
			waypts[i].append(x_f[i])

		# Initialize lists to store feasible trajectories for agents
		xcl_feas = []
		ucl_feas = []

		# Initialize FTOCP objects
		# Initial trajectory MPC parameters for each agent
		Q = np.diag([20.0, 20.0, 15.0, 25.0])
		R = np.diag([10.0, 30.0])
		Rd = np.diag([10.0, 30.0])
		# R = np.diag([100.0, 50.0])
		# Rd = np.diag([100.0, 50.0])
		P = Q
		N = 15
		ltv_ftocps = [LTV_FTOCP(Q, P, R, Rd, N, mpc_control_agents[i], x_refs=waypts[i]) for i in range(n_a)]

		if Q.shape[0] != Q.shape[1] or len(np.diag(Q)) != n_x:
			raise(ValueError('Q matrix not shaped properly'))
		if R.shape[0] != R.shape[1] or len(np.diag(R)) != n_u:
			raise(ValueError('Q matrix not shaped properly'))

		start = time.time()
		if parallel:
			# Create threads
			pool = mp.Pool(processes=n_a)
			# Assign thread to agent trajectory
			results = [pool.apply_async(solve_init_traj, args=(ltv_ftocps[i], x_0[i], tol)) for i in range(n_a)]
			# Sync point
			init_trajs = [r.get() for r in results]

			(xcl_feas, ucl_feas) = zip(*init_trajs)
			xcl_feas = list(xcl_feas)
			ucl_feas = list(ucl_feas)
		else:
			for i in range(n_a):
				print('Solving for initial trajectory for agent %i' % (i+1))
				x, u = solve_init_traj(ltv_ftocps[i], x_0[i], model_agents[i], tol=tol)
				xcl_feas.append(x)
				ucl_feas.append(u)
		end = time.time()

		for i in range(n_a):
			x_last = xcl_feas[i][:,-1]
			x_last[3] = 0.0
			# xcl_feas[i] = np.append(xcl_feas[i], x_last.reshape((n_x,1)), axis=1)
			ucl_feas[i] = np.append(ucl_feas[i], np.zeros((n_u,1)), axis=1)

			# Save initial trajecotry if file doesn't exist
			if not os.path.exists('/'.join((FILE_DIR, 'init_traj_%i.npz' % i))):
				print('Saving initial trajectory for agent %i' % (i+1))
				np.savez('/'.join((FILE_DIR, 'init_traj_%i.npz' % i)), x=xcl_feas[i], u=ucl_feas[i])

		print('Time elapsed: %g s' % (end - start))
	else:
		# Load initial trajectory from file
		x_f = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
		xcl_feas = []
		ucl_feas = []
		for i in range(n_a):
			init_traj = np.load('/'.join((FILE_DIR, 'init_traj_%i.npz' % i)), allow_pickle=True)
			xcl_feas.append(init_traj['x'])
			ucl_feas.append(init_traj['u'])

	if plot_init:
		plot_bike_agent_trajs(xcl_feas, ucl_feas, model_agents, model_dt, trail=True, plot_lims=plot_lims, it=0)

	# Set goal state to be last state of initial trajectories
	for i in range(n_a):
		x_f[i] = xcl_feas[i][:,-1]

	# pdb.set_trace()
	# ====================================================================================

	# ====================================================================================
	# Run LMPC
	# ====================================================================================

	# Initialize LMPC objects for each agent
	N_LMPC = [20, 20, 20] # horizon lengths
	# N_LMPC = [15, 15, 15] # horizon lengths
	lmpc_ftocp = [NL_FTOCP(lmpc_control_agents[i]) for i in range(n_a)]# ftocp solve by LMPC
	lmpc = [NL_LMPC(f, N_LMPC[i]) for f in lmpc_ftocp]# Initialize the LMPC

	xcls = [copy.copy(xcl_feas)]
	ucls = [copy.copy(ucl_feas)]

	ss_n_t = 30
	# ss_n_t = 5
	ss_n_j = 5

	totalIterations = 5 # Number of iterations to perform
	start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	exp_dir = '/'.join((out_dir, start_time))
	os.makedirs(exp_dir)

	# Initialize visualizer for each agent
	# lmpc_vis = [lmpc_visualizer(pos_dims=[0,1], n_state_dims=n_x, n_act_dims=n_u, agent_id=i, plot_lims=plot_lims) for i in range(n_a)]
	lmpc_vis = [None for i in range(n_a)]

	raw_input('Ready to run LMPC, press enter to continue...')

	# run simulation
	# iteration loop
	for it in range(totalIterations):
		print('****************** Iteration %i ******************' % (it+1))
		plot_bike_agent_trajs(xcls[-1], ucls[-1], model_agents, model_dt, trail=True, plot_lims=plot_lims, save_dir=exp_dir, it=it)

		pdb.set_trace()

		# Compute safe sets and exploration spaces along previous trajectory
		ss_idxs, expl_constrs = get_safe_set(xcls, x_f, lmpc_control_agents, ss_n_t, ss_n_j)

		# inspect_safe_set(xcls, ucls, ss_idxs, expl_constrs, plot_lims)
		# pdb.set_trace()

		for i in range(n_a):
			print('Adding trajectories and updating safe sets for agent %i' % (i+1))
			lmpc[i].addTrajectory(xcls[-1][i], ucls[-1][i], x_f[i]) # Add feasible trajectory to the safe set
			lmpc[i].update_safe_sets(ss_idxs[i])
			lmpc[i].update_exploration_constraints(expl_constrs[i])

		# pdb.set_trace()

		for lv in lmpc_vis:
			if lv is not None:
				lv.update_prev_trajs(state_traj=xcls[-1], act_traj=ucls[-1])

		it_start = time.time()

		x_it = []
		u_it = []
		# agent loop
		for i in range(n_a):
			print('Solving trajectory for agent %i' % (i+1))
			agent_start = time.time()
			agent_dir = '/'.join((exp_dir, 'it_%i' % (it+1), 'agent_%i' % (i+1)))
			os.makedirs(agent_dir)
			if lmpc_vis[i] is not None:
				lmpc_vis[i].set_plot_dir(agent_dir)

			xcl, ucl = solve_lmpc(lmpc[i], x_0[i], x_f[i], model_agents[i], visualizer=lmpc_vis[i], pause=pause_each_solve, tol=tol)
			# opt_cost = lmpc[i].addTrajectory(xcl, ucl)
			# obj_plot.update(np.array([it, opt_cost]).T, i)
			ucl= np.append(ucl, np.zeros((n_u,1)), axis=1)

			x_it.append(xcl)
			u_it.append(ucl)

			agent_end = time.time()
			print('Time elapsed: %g, trajectory length: %i' % (agent_end-agent_start, xcl.shape[1]))

		xcls.append(x_it)
		ucls.append(u_it)

		it_end = time.time()
		print('Time elapsed for iteration %i: %g s' % (it+1, it_end - it_start))

		pickle.dump(lmpc, open('/'.join((exp_dir, 'it_%i.pkl' % (it+1))), 'wb'))

	# Plot last trajectory
	plot_bike_agent_trajs(xcls[-1], ucls[-1], model_agents, model_dt, trail=True, plot_lims=plot_lims, save_dir=exp_dir, it=it)
	#=====================================================================================

	plt.show()

if __name__== "__main__":
  main()
