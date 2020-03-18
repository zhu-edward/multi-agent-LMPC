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
from agents import Centralized_DT_Kin_Bike_Agent, DT_Kin_Bike_Agent

from utils.plot_bike_utils import plot_bike_agent_trajs
from utils.lmpc_visualizer import lmpc_visualizer
from utils.safe_set_utils import get_safe_set_cent, inspect_safe_set
import utils.utils

def solve_init_traj(ftocp, x_0, model_agent, visualizer=None, tol=-7):
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

		if visualizer is not None:
			visualizer.plot_traj(np.array(xcl_feas).T, np.array(ucl_feas).T, x_pred, u_pred, counter, shade=False)

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

	x_ol = []
	u_ol = []

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
			# visualizer.traj_inspector(xcl, ucl, x_ol, u_ol, t, lmpc.SS_t, lmpc.expl_constrs)
			sys.exit()
		else:
			x_ol.append(x_pred)
			u_ol.append(u_pred)

		u_t = u_pred[:,0]
		# Read input and apply it to the system
		x_tp1 = model_agent.sim(x_t, u_t)

		ucl = np.append(ucl, u_t.reshape((-1,1)), axis=1)
		xcl = np.append(xcl, x_tp1.reshape((-1,1)), axis=1)

		print('ts: %g, d: %g, x: %g, y: %g, phi: %g, v: %g' % (t, la.norm(x_tp1-x_f, ord=2), x_t[0], x_t[1], x_t[2]*180.0/np.pi, x_t[3]))

		if visualizer is not None:
			visualizer.plot_traj(xcl, ucl, x_pred, u_pred, t, SS, lmpc.expl_constrs, shade=False)

		if la.norm(x_tp1-x_f, ord=2) <= 10**tol:
			print('Tolerance reached, agent reached goal state')
			break

		if pause:
			pdb.set_trace()

		t += 1

	# Inspection mode after iteration completion
	# if inspect:
		# visualizer.traj_inspector(xcl, ucl, x_pred_log, u_pred_log, t, lmpc.SS_t, lmpc.expl_constrs)

	return xcl, ucl, x_ol, u_ol

def solve_lmpc_cent(lmpc, x_0, x_f, model_agent, verbose=False, visualizer=None, pause=False, tol=-7):
	n_x = lmpc.n_x
	n_u = lmpc.n_u
	n_a = lmpc.n_a

	x_ol = []
	u_ol = []

	xcl = x_0.reshape((-1,1)) # initialize system state
	ucl = np.empty((n_u*n_a,0))

	inspect = False
	solve_times = []

	t = 0
	# time Loop (Perform the task until close to the origin)
	while True:
		goal_reached = True
		x_t = xcl[:,t] # Read measurements
		solve_start = time.time()
		x_pred, u_pred, cost, SS, N = lmpc.solve(t, x_t, x_f, tol, verbose=verbose) # Solve FTOCP
		solve_end = time.time()
		solve_time = solve_end - solve_start
		# Inspect incomplete trajectory
		if x_pred is None or u_pred is None and visualizer is not None:
			# visualizer.traj_inspector(xcl, ucl, x_ol, u_ol, t, lmpc.SS_t, lmpc.expl_constrs)
			sys.exit()
		else:
			x_ol.append(x_pred)
			u_ol.append(u_pred)

		u_t = u_pred[:,0]
		# Read input and apply it to the system
		x_tp1 = model_agent.sim(x_t, u_t)

		ucl = np.append(ucl, u_t.reshape((-1,1)), axis=1)
		xcl = np.append(xcl, x_tp1.reshape((-1,1)), axis=1)

		solve_times.append(solve_time)
		print('ts: %g, d: %g, x: %g, y: %g, phi: %g, v: %g' % (t, la.norm(x_tp1-x_f, ord=2), x_t[0], x_t[1], x_t[2]*180.0/np.pi, x_t[3]))

		if visualizer is not None:
			for i in range(n_a):
				visualizer[i].plot_traj(xcl[i*n_x:(i+1)*n_x,:], ucl[i*n_u:(i+1)*n_u,:], x_pred[i*n_x:(i+1)*n_x,:], u_pred[i*n_u:(i+1)*n_u,:], t, SS=SS[i*n_x:(i+1)*n_x,:], shade=False)

		for i in range(n_a):
			if la.norm(x_tp1[i*n_x:(i+1)*n_x]-x_f[i*n_x:(i+1)*n_x], ord=2) > 10**tol:
				goal_reached &= False

		if goal_reached:
			print('Tolerance reached, agent reached goal state')
			break

		if pause:
			pdb.set_trace()

		t += 1

	# Inspection mode after iteration completion
	# if inspect:
		# visualizer.traj_inspector(xcl, ucl, x_pred_log, u_pred_log, t, lmpc.SS_t, lmpc.expl_constrs)

	return xcl, ucl, x_ol, u_ol, solve_times

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

	col_buf = 0.2

	# Initial Conditions
	x_0 = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]
	# Example 1, 2
	# x_0[0] = np.array([0.0, 5.0, np.pi, 0.0])
	# x_0[1] = np.array([-5.0, -5.0, 0.0, 0.0])
	# x_0[2] = np.array([5.0, -5.0, np.pi/2.0, 0.0])

	# Agent intersection
	x_0[0] = np.array([0.0, 5.0, 3*np.pi/2, 0.0])
	x_0[1] = np.array([-5.0, -5.0, np.pi/4, 0.0])
	x_0[2] = np.array([5.0, -5.0, 3*np.pi/4, 0.0])
	# x_0 = np.concatenate(x_0)

	# Initialize dynamics and control agents (allows for dynamics to be simulated with higher resolution than control rate)
	model_agents = [DT_Kin_Bike_Agent(l_r, l_f, w, model_dt, col_buf=col_buf, v_lim=[-0.05, 10.0]) for i in range(n_a)]
	mpc_control_agents = [DT_Kin_Bike_Agent(l_r, l_f, w, control_dt, col_buf=col_buf, a_lim=[-1.0, 1.0], df_lim=[-0.5, 0.5], da_lim=[-1.5, 1.5], ddf_lim=[-0.3, 0.3]) for i in range(n_a)]
	cent_model_agent = Centralized_DT_Kin_Bike_Agent(l_r, l_f, w, model_dt, n_a, col_buf=col_buf, v_lim=[-0.05, 10.0])
	lmpc_control_agent = Centralized_DT_Kin_Bike_Agent(l_r, l_f, w, control_dt, n_a, col_buf=col_buf, v_lim=[-0.05, 10.0])

	if not args.init_traj:
		# ====================================================================================
		# Run LTV MPC to compute feasible solutions for all agents
		# ====================================================================================
		# Goal conditions (these will be updated once the initial trajectories are found)
		x_f = [np.nan*np.ones((n_x, 1)) for _ in range(n_a)]

		# Example 1
		# x_f[0] = np.array([0.0, -5.0, 2*np.pi, 0.0])
		# x_f[1] = np.array([5.0, 5.0, np.pi/2, 0.0])
		# x_f[2] = np.array([-5.0, 5.0, np.pi, 0.0])

		# Example 2
		# x_f[0] = np.array([0.0, -5.0, 7*np.pi/4, 0.0])
		# x_f[1] = np.array([5.0, 5.0, np.pi/4, 0.0])
		# x_f[2] = np.array([-5.0, 5.0, 3*np.pi/4, 0.0])

		# Agent intersection
		x_f[0] = np.array([0.0, -5.0, 3*np.pi/2, 0.0])
		x_f[1] = np.array([5.0, 5.0, np.pi/4, 0.0])
		x_f[2] = np.array([-5.0, 5.0, 3*np.pi/4, 0.0])

		# Check to make sure all agent dynamics, inital, and goal states have been defined
		if np.any(np.isnan(x_0)) or np.any(np.isnan(x_f)):
			raise(ValueError('Initial or goal states have empty entries'))

		# Intermediate waypoint to ensure collision-free trajectory
		waypts = [[] for _ in range(n_a)]
		# Example 1
		# waypts[0] = [np.array([-5.0, 0.0, 3.0*np.pi/2.0, 0.5])]
		# waypts[1] = [np.array([0.0, -5.0, 0.0, 0.5])]
		# waypts[2] = [np.array([5.0, 0.0, np.pi/2, 0.5])]

		# Example 2
		# waypts[0] = [np.array([-5.0, 0.0, 3.0*np.pi/2.0, 0.5])]
		# waypts[1] = [np.array([0.0, -6.0, 0.0, 0.5])]
		# waypts[2] = [np.array([4.0, 0.0, np.pi/2, 0.5])]
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

		mpc_vis = [lmpc_visualizer(pos_dims=[0,1], n_state_dims=n_x, n_act_dims=n_u, agent_id=i, n_agents=n_a, plot_lims=plot_lims) for i in range(n_a)]

		if Q.shape[0] != Q.shape[1] or len(np.diag(Q)) != n_x:
			raise(ValueError('Q matrix not shaped properly'))
		if R.shape[0] != R.shape[1] or len(np.diag(R)) != n_u:
			raise(ValueError('Q matrix not shaped properly'))

		start = time.time()
		for i in range(n_a):
			print('Solving for initial trajectory for agent %i' % (i+1))
			x, u = solve_init_traj(ltv_ftocps[i], x_0[i], model_agents[i], visualizer=mpc_vis[i], tol=tol)
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

			mpc_vis[i].close_figure()

		del ltv_ftocps, mpc_vis

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

	# Shift agent trajectories in time so that they occur sequentially
	# (no collisions)
	xcl_lens = [xcl_feas[i].shape[1] for i in range(n_a)]

	for i in range(n_a):
		before_len = 50*i
		after_len = np.amax(xcl_lens) - xcl_lens[i] + 50*(n_a-(i+1))

		# xcl_feas[i] = np.hstack((np.tile(x_0[i].reshape((-1,1)), before_len), xcl_feas[i]))
		# ucl_feas[i] = np.hstack((np.zeros((n_u, before_len)), ucl_feas[i]))
		xcl_feas[i] = np.hstack((np.tile(x_0[i].reshape((-1,1)), before_len), xcl_feas[i], np.tile(xcl_feas[i][:,-1].reshape((-1,1)), after_len)))
		ucl_feas[i] = np.hstack((np.zeros((n_u, before_len)), ucl_feas[i], np.zeros((n_u, after_len))))

	if plot_init:
		plot_bike_agent_trajs(xcl_feas, ucl_feas, model_agents, model_dt, trail=True, plot_lims=plot_lims, it=0)

	# Set goal state to be last state of initial trajectories
	for i in range(n_a):
		x_f[i] = xcl_feas[i][:,-1]
	x_0 = np.concatenate(x_0)
	x_f = np.concatenate(x_f)

	xcl_feas_cent = np.concatenate(xcl_feas, axis=0)
	ucl_feas_cent = np.concatenate(ucl_feas, axis=0)

	xcls = [copy.copy(xcl_feas)]
	ucls = [copy.copy(ucl_feas)]

	del xcl_feas, ucl_feas

	pdb.set_trace()

	# ====================================================================================

	# ====================================================================================
	# Run LMPC
	# ====================================================================================

	# Initialize LMPC objects for each agent
	N_LMPC = 20 # horizon lengths
	# N_LMPC = 10 # horizon lengths
	lmpc_ftocp = NL_FTOCP(lmpc_control_agent) # ftocp solve by LMPC
	lmpc = NL_LMPC(lmpc_ftocp, N_LMPC) # Initialize the LMPC

	xcls_cent = [copy.copy(xcl_feas_cent)]
	ucls_cent = [copy.copy(ucl_feas_cent)]

	ss_n_t = 175
	ss_n_j = 2

	totalIterations = 100 # Number of iterations to perform
	start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	exp_dir = '/'.join((out_dir, start_time))
	os.makedirs(exp_dir)

	# Initialize visualizer for each agent
	lmpc_vis = [lmpc_visualizer(pos_dims=[0,1], n_state_dims=n_x, n_act_dims=n_u, agent_id=i, n_agents=n_a, plot_lims=plot_lims) for i in range(n_a)]
	# lmpc_vis = None

	it_times = []
	solve_times = []
	print('Starting multi-agent LMPC...')
	# run simulation
	# iteration loop
	for it in range(totalIterations):
		print('****************** Iteration %i ******************' % (it+1))
		it_dir = '/'.join((exp_dir, 'it_%i' % (it+1)))
		os.makedirs(it_dir)

		if lmpc_vis is not None:
			for lv in lmpc_vis:
				lv.set_save_dir(it_dir)

		plot_bike_agent_trajs(xcls[-1], ucls[-1], model_agents, model_dt, trail=True, plot_lims=plot_lims, save_dir=exp_dir, save_video=True, it=it)

		it_start = time.time()

		# Compute safe sets and exploration spaces along previous trajectory
		ss_idxs = get_safe_set_cent(xcls_cent, ss_n_t, ss_n_j)

		# inspect_safe_set(xcls, ucls, ss_idxs, expl_constrs, plot_lims)

		print('Adding trajectories and updating safe sets')
		lmpc.addTrajectory(xcls_cent[-1], ucls_cent[-1], x_f) # Add feasible trajectory to the safe set
		lmpc.update_safe_sets(ss_idxs)

		if lmpc_vis is not None:
			for lv in lmpc_vis:
				lv.update_prev_trajs(state_traj=xcls, act_traj=ucls)

		x_ol_it = []
		u_ol_it = []

		print('Solving trajectory for iteration %i' % (it+1))

		x_cl, u_cl, x_ol, u_ol, solve_t = solve_lmpc_cent(lmpc, x_0, x_f, cent_model_agent, visualizer=lmpc_vis, pause=pause_each_solve, tol=tol)
		u_cl= np.append(u_cl, np.zeros((n_a*n_u,1)), axis=1)

		it_end = time.time()
		print('Time elapsed for iteration %i: %g s, trajectory length: %i' % (it+1, it_end - it_start,  x_cl.shape[1]))

		x_ol_it.append(x_ol)
		u_ol_it.append(u_ol)
		xcls_cent.append(x_cl)
		ucls_cent.append(u_cl)
		it_times.append(it_end - it_start)
		solve_times.append(solve_t)

		xcls.append([x_cl[i*n_x:(i+1)*n_x,:] for i in range(n_a)])
		ucls.append([u_cl[i*n_u:(i+1)*n_u,:] for i in range(n_a)])

		# Save iteration data
		pickle.dump(lmpc, open('/'.join((it_dir, 'lmpc.pkl')), 'wb'))
		pickle.dump(ss_idxs, open('/'.join((it_dir, 'ss.pkl')), 'wb'))
		# pickle.dump(expl_constrs, open('/'.join((it_dir, 'exp_constr.pkl')), 'wb'))
		pickle.dump(xcls_cent, open('/'.join((it_dir, 'x_cls.pkl')), 'wb'))
		pickle.dump(ucls_cent, open('/'.join((it_dir, 'u_cls.pkl')), 'wb'))
		pickle.dump(x_ol_it, open('/'.join((it_dir, 'x_ol.pkl')), 'wb'))
		pickle.dump(u_ol_it, open('/'.join((it_dir, 'u_ol.pkl')), 'wb'))
		pickle.dump(it_times, open('/'.join((it_dir, 'it_times.pkl')), 'wb'))
		pickle.dump(solve_times, open('/'.join((it_dir, 'solve_times.pkl')), 'wb'))

	# Plot last trajectory
	plot_bike_agent_trajs(xcls[-1], ucls[-1], model_agents, model_dt, trail=True, plot_lims=plot_lims, save_dir=exp_dir, save_video=True, it=it)
	#=====================================================================================

	plt.show()

if __name__== "__main__":
  main()
