from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.spatial
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import svm
import pdb, itertools, matplotlib

def get_safe_set(x_cls, xf, agents, des_num_ts='all', des_num_iters='all'):
	n_a = len(x_cls[0])
	n_x = x_cls[0][0].shape[0]

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	# Enumerate pairs of agents
	pairs = list(itertools.combinations(range(n_a), 2))

	# Get the minimum distance for collision avoidance between agents based on the geometry of their occupied space
	min_dist = []
	r_a = [agents[i].get_collision_buff_r() for i in range(n_a)]
	for p in pairs:
		dist = r_a[p[0]] + r_a[p[1]]
		min_dist.append(dist)
	num_ts = 0
	num_iters = len(x_cls)
	cl_lens = []

	# Get the longest trajectory over the last iteration
	it_start = max(0, num_iters-des_num_iters)
	orig_range = range(it_start, num_iters)
	for j in orig_range:
		iter_cls = x_cls[j]
		it_cl_lens = []
		for agent_cl in iter_cls:
			it_cl_lens.append(agent_cl.shape[1])
			if agent_cl.shape[1] > num_ts and j == orig_range[-1]:
				num_ts = agent_cl.shape[1]
		cl_lens.append(it_cl_lens)

	# Set number of time steps to be included to the trajectory length if it was larger
	if num_ts < des_num_ts:
		des_num_ts = num_ts

	if des_num_iters == 'all':
		des_num_iters = num_iters
	if des_num_ts == 'all':
		des_num_ts = num_ts

	# safe_set_idxs = [agent_0_ss_idxs, agent_1_ss_idxs, ... , agent_M_ss_idxs]
	# agent_#_ss_idxs = [ss_idxs_0, ss_idxs_1, ... , ss_idxs_T]
	safe_sets_idxs = [[] for _ in range(n_a)]
	exploration_spaces = [[] for _ in range(n_a)]
	last_invalid_t = -1

	for t in range(num_ts):
		# Determine starting iteration index and ending time step index
		it_start = max(0, num_iters-des_num_iters)
		# ts_end = min(num_ts, t+des_num_ts)
		ts_end = t+des_num_ts
		H_t = [[] for _ in range(n_a)]
		g_t = [[] for _ in range(n_a)]
		while True:
			# Construct candidate safe set
			print('Constructing safe set from iteration %i to %i and time %i to %i' % (it_start, num_iters-1, t, ts_end-1))
			safe_set_cand_t = []
			for a in range(n_a):
				it_range = range(it_start, num_iters)
				ts_range = []
				for j in it_range:
					i = orig_range.index(j)
					ts_range.append(range(min(t, cl_lens[i][a]-1), min(ts_end, cl_lens[i][a])))
					# print(range(min(t, cl_lens[i][a]-1), min(ts_end, cl_lens[i][a])), x_cls[j][a].shape)
				ss_idxs = {'it_range' : it_range, 'ts_range' : ts_range}
				safe_set_cand_t.append(ss_idxs) # Candidate safe sets at this time step

			# Check for potential overlap and minimum distance between agent safe sets
			all_valid = True
			for (p, d) in zip(pairs, min_dist):
				collision = False
				# Collision only defined for position states
				safe_set_pos_0 = np.empty((2,0))
				safe_set_pos_1 = np.empty((2,0))
				for (i, j) in enumerate(safe_set_cand_t[p[0]]['it_range']):
					safe_set_pos_0 = np.append(safe_set_pos_0, x_cls[j][p[0]][:2,safe_set_cand_t[p[0]]['ts_range'][i]], axis=1)
					safe_set_pos_1 = np.append(safe_set_pos_1, x_cls[j][p[1]][:2,safe_set_cand_t[p[1]]['ts_range'][i]], axis=1)

				# Stack safe set position vectors into data matrix and assign labels agent p[0]: -1, agent p[1]: 1
				X = np.append(safe_set_pos_0, safe_set_pos_1, axis=1).T
				y = np.append(-np.ones(safe_set_pos_0.shape[1]), np.ones(safe_set_pos_1.shape[1]))

				# if t == 68:
				# 	pdb.set_trace()

				# Use SVM with linear kernel and no regularization (w'x + b <= -a_0 for agent p[0], w'x + b >= a_1 for agent p[1])
				# clf = svm.SVC(kernel='linear', C=1000)
				clf = svm.SVC(kernel='linear', C=1000, max_iter=1000)
				clf.fit(X, y)
				w = np.squeeze(clf.coef_)
				b = np.squeeze(clf.intercept_)

				# Calculate classifier margin
				margin = 2/la.norm(w, 2)

				# Check for misclassification of support vectors. This indicates that the safe sets are not linearlly separable
				for i in clf.support_:
					pred_label = clf.predict(X[i].reshape((1,-1)))
					# pred_val = clf.decision_function(X[i].reshape((1,-1)))
					if pred_label != y[i]:
						collision = True
						print('Potential for collision between agents %i and %i' % (p[0],p[1]))
						break
				# Check for distance between safe sets
				if not collision and margin < d:
					print('Margin between safe sets for agents %i and %i is too small' % (p[0],p[1]))

				# If collision is possible or margin is less than minimum required distance between safe sets, reduce safe set
				# iteration and/or time range
				# Currently, we reduce iteration range first. If iteration range cannot be reduced any further then we reduce time step range
				if collision or margin < d:
					all_valid = False
					it_start += 1
					if it_start >= num_iters:
						it_start = max(0, num_iters-des_num_iters)
						ts_end -= 1

					# Update the time step when a range reduction was last required, we will use this at the end to iterate through
					# the safe sets up to this time and make sure that all safe sets use the same iteration and time range
					last_invalid_t = t

					# Reset the candidate exploration spaces
					H_t = [[] for _ in range(n_a)]
					g_t = [[] for _ in range(n_a)]
					break

				# Distance between hyperplanes is (a_0+a_1)/\|w\|
				a_0_min = d*la.norm(w, 2)/(1 + r_a[p[1]]/r_a[p[0]])
				a_1_min = d*la.norm(w, 2)/(1 + r_a[p[0]]/r_a[p[1]])

				ratio_remain_0 = la.norm(x_cls[0][p[0]][:2,-1] - safe_set_pos_0[:,0], 2)/la.norm(x_cls[0][p[0]][:2,-1] - x_cls[0][p[0]][:2,0], 2)
				ratio_remain_1 = la.norm(x_cls[0][p[1]][:2,-1] - safe_set_pos_1[:,0], 2)/la.norm(x_cls[0][p[1]][:2,-1] - x_cls[0][p[1]][:2,0], 2)
				w_0 = 1.0 # w_0 = np.exp(35*ratio_remain_0-3)/(np.exp(35*ratio_remain_0-3)+1)
				w_1 = 1.0 # w_1 = np.exp(35*ratio_remain_1-3)/(np.exp(35*ratio_remain_1-3)+1)

				# Solve for tight hyperplane bounds for both collections of points
				z = cp.Variable(1)
				cost = z
				constr = []
				for i in range(safe_set_pos_0.shape[1]):
					constr += [w.dot(safe_set_pos_0[:,i]) + b <= z]
				problem = cp.Problem(cp.Minimize(cost), constr)
				problem.solve(solver=cp.MOSEK, verbose=False)
				# problem.solve(verbose=False)
				a_0_max = -z.value[0]

				z = cp.Variable(1)
				cost = z
				constr = []
				for i in range(safe_set_pos_1.shape[1]):
					constr += [-w.dot(safe_set_pos_1[:,i]) - b <= z]
				problem = cp.Problem(cp.Minimize(cost), constr)
				problem.solve(solver=cp.MOSEK, verbose=False)
				# problem.solve(verbose=False)
				a_1_max = -z.value[0]

				if a_0_max > a_0_min and a_1_max > a_1_min:
					if w_0 <= w_1:
						a_shift = (a_0_max - a_0_min)*(1-w_0/w_1)
						a_0 = a_0_min + a_shift
						a_1 = a_1_min - a_shift
					else:
						a_shift = (a_1_max - a_1_min)*(1-w_1/w_0)
						a_0 = a_0_min - a_shift
						a_1 = a_1_min + a_shift
				else:
					a_0 = a_0_max - 1e-5 # Deal with precision issues when a point in the safe set is on the exploration space boundary
					a_1 = a_1_max - 1e-5

				# Exploration spaces
				H_t[p[0]].append(w)
				g_t[p[0]].append(b+a_0)
				H_t[p[1]].append(-w)
				g_t[p[1]].append(-b+a_1)

				# plot_svm_results(X, y, clf)

			# all_valid flag is true if all pair-wise collision and margin checks were passed
			if all_valid:
				# Save iteration and time range from this time step, start with these values next time step
				des_num_iters = num_iters - it_start
				des_num_ts = ts_end - t
				for a in range(n_a):
					H_t[a] = np.array(H_t[a])
					g_t[a] = np.array(g_t[a])
				print('Safe set construction successful for t = %i, using iteration range %i and time range %i for next time step' % (t, des_num_iters, des_num_ts))
				break # Break from while loop

		for a in range(n_a):
			safe_sets_idxs[a].append(safe_set_cand_t[a])
			exploration_spaces[a].append((H_t[a], g_t[a]))

	# Adjust safe sets from before last_invalid_t to have the same iteration and time range and test that safe sets are contained
	# in the exploration spaces at each time step
	for t in range(num_ts-1):
		for a in range(n_a):
			if t <=  last_invalid_t:
				old_it_len = len(safe_sets_idxs[a][t]['it_range'])
				safe_sets_idxs[a][t]['it_range'] = safe_sets_idxs[a][last_invalid_t+1]['it_range'] # Update iteration range
				new_it_len = len(safe_sets_idxs[a][t]['it_range'])
				for _ in range(old_it_len - new_it_len):
					safe_sets_idxs[a][t]['ts_range'].pop(0) # Throw away iterations that we don't include anymore
				for i in range(new_it_len):
					n_ss = len(safe_sets_idxs[a][t]['ts_range'][i])
					if n_ss > des_num_ts:
						safe_sets_idxs[a][t]['ts_range'][i] = safe_sets_idxs[a][t]['ts_range'][i][:des_num_ts] # Update time range for remaining iterations

			safe_set_pos = np.empty((2,0))
			for (i, j) in enumerate(safe_sets_idxs[a][t]['it_range']):
				safe_set_pos = np.append(safe_set_pos, x_cls[j][a][:2,safe_sets_idxs[a][t]['ts_range'][i]], axis=1)
			in_exp_space = (exploration_spaces[a][t][0].dot(safe_set_pos) + exploration_spaces[a][t][1].reshape((-1,1)) <= 0)
			if not np.all(in_exp_space):
				raise(ValueError('Safe set not contained in exploration space at time %i' % t))

	# pdb.set_trace()
	return safe_sets_idxs, exploration_spaces

def get_safe_set_cent(x_cls):
	num_ts = 0
	num_iters = len(x_cls)
	cl_lens = []
	for i in range(num_iters):
		cl_lens.append(x_cls[i].shape[1])
		if x_cls[i].shape[1] > num_ts:
			num_ts = x_cls[i].shape[1]

	it_range = range(num_iters)
	ts_range = []
	for i in it_range:
		ts_range.append(range(cl_lens[i]))
	ss_idxs = {'it_range' : it_range, 'ts_range' : ts_range}

	safe_set_idxs = [ss_idxs]

	return safe_set_idxs

def inspect_safe_set(x, u, safe_sets_idxs, exploration_spaces, plot_lims=None):
	n_a = len(x[-1])
	n_SS = len(safe_sets_idxs[0])

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	plt.ion()

	fig = plt.figure()
	xy_ax = fig.add_axes([0, 0, 1, 1])
	# psi_ax = fig.add_axes([1.1, 0.9, 1, 0.2])
	# psi_ax.set_xticks([])
	# v_ax = fig.add_axes([1.1, 0.6, 1, 0.2])
	# v_ax.set_xticks([])
	# df_ax = fig.add_axes([1.1, 0.3, 1, 0.2])
	# df_ax.set_xticks([])
	# a_ax = fig.add_axes([1.1, 0.0, 1, 0.2])

	xy_ax.set_xlabel('x')
	xy_ax.set_ylabel('y')
	if plot_lims is not None:
		xy_ax.set_xlim(plot_lims[0])
		xy_ax.set_ylim(plot_lims[1])
	xy_ax.set_aspect('equal')

	t = 0
	new_plot = False

	print('step = %i' % t)
	while True:
		input = raw_input('(debug) ')
		# Quit inspector
		if input == 'q':
			break
		# Move forward 1 time step
		elif input == 'f':
			if t == n_SS-1:
				print('End reached')
				continue
			else:
				t += 1
				new_plot = True
				print('t = %i' % t)
		# Move backward 1 time step
		elif input == 'b':
			if t == 0:
				print('Start reached')
				continue
			else:
				t -= 1
				new_plot = True
				print('t = %i' % t)
		else:
			print('Input not recognized')
			print('Press q to exit, f/b to move forward/backwards through iteration time steps')

		if new_plot:
			xy_ax.clear()

			for a in range(n_a):
				ss_x = []
				ss_u = []
				ss_it_idxs = safe_sets_idxs[a][t]['it_range']
				ss_ts_idxs = safe_sets_idxs[a][t]['ts_range']
				print(ss_it_idxs, ss_ts_idxs)
				for i in ss_it_idxs:
					for j in ss_ts_idxs:
						ss_x.append(x[i][a][:,j])
						ss_u.append(u[i][a][:,j])
				ss_x = np.array(ss_x)
				ss_u = np.array(ss_u)

				H_t = exploration_spaces[a][t][0]
				g_t = exploration_spaces[a][t][1]

				xy_ax.plot(ss_x[:,0], ss_x[:,1], '.', c=c[a])

				y_0 = (-H_t[0,0]*np.array(plot_lims[0])-g_t[0])/H_t[0,1]
				y_1 = (-H_t[1,0]*np.array(plot_lims[0])-g_t[1])/H_t[1,1]
				xy_ax.plot(plot_lims[0], y_0, c=c[a])
				xy_ax.plot(plot_lims[0], y_1, c=c[a])

				xy_ax.set_xlabel('x')
				xy_ax.set_ylabel('y')
				if plot_lims is not None:
					xy_ax.set_xlim(plot_lims[0])
					xy_ax.set_ylim(plot_lims[1])
				xy_ax.set_aspect('equal')

			fig.canvas.draw()

			new_plot = False
