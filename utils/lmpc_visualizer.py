import numpy as np
import copy, pickle, pdb, time, sys, os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

class lmpc_visualizer(object):
	def __init__(self, pos_dims, n_state_dims, n_act_dims, agent_id, plot_lims=None, plot_dir=None):
		if len(pos_dims) > 2:
			raise(ValueError('Can only plot 2 position dimensions'))
		self.agent_id = agent_id
		self.pos_dims = pos_dims
		self.n_state_dims = n_state_dims
		self.n_act_dims = n_act_dims
		self.plot_dir = plot_dir
		self.plot_lims = plot_lims

		plt.ion()

		# Initialize position plot
		self.pos_fig = plt.figure()
		self.pos_ax = plt.gca()
		if self.plot_lims is not None:
			self.pos_ax.set_xlim(self.plot_lims[0])
			self.pos_ax.set_ylim(self.plot_lims[1])
		self.pos_ax.set_xlabel('$x$')
		self.pos_ax.set_ylabel('$y$')
		self.pos_ax.set_title('Agent %i' % (agent_id+1))
		self.pos_fig.canvas.set_window_title('agent %i positions' % (agent_id+1))
		self.pos_fig.canvas.draw()

		# Initialize velocity plot
		self.state_fig = plt.figure(figsize=(5,4))
		self.state_axs = [self.state_fig.add_subplot(n_state_dims, 1, i+1) for i in range(n_state_dims)]
		for (i, a) in enumerate(self.state_axs):
			a.set_ylabel('$x_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (agent_id+1))
			if i < len(self.state_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.state_axs)-1:
				a.set_xlabel('$t$')
		self.state_fig.canvas.set_window_title('agent %i states' % (agent_id+1))
		self.state_fig.canvas.draw()

		# Initialize input plot
		self.act_fig = plt.figure(figsize=(5,4))
		self.act_axs = [self.act_fig.add_subplot(n_act_dims, 1, i+1) for i in range(n_act_dims)]
		for (i, a) in enumerate(self.act_axs):
			a.set_ylabel('$u_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (agent_id+1))
			if i < len(self.act_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.act_axs)-1:
				a.set_xlabel('$t$')
		self.act_fig.canvas.set_window_title('agent %i inputs' % (agent_id+1))
		self.act_fig.canvas.draw()

		self.prev_pos_cl = None
		self.prev_state_cl = None
		self.prev_act_cl = None

		self.it = 0

	def set_plot_dir(self, plot_dir):
		self.plot_dir = plot_dir

	def clear_state_plots(self):
		self.pos_ax.clear()
		if self.plot_lims is not None:
			self.pos_ax.set_xlim(self.plot_lims[0])
			self.pos_ax.set_ylim(self.plot_lims[1])
		self.pos_ax.set_xlabel('$x$')
		self.pos_ax.set_ylabel('$y$')
		self.pos_ax.set_title('Agent %i' % (self.agent_id+1))
		for a in self.state_axs:
			a.clear()
		for (i, a) in enumerate(self.state_axs):
			a.set_ylabel('$x_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (self.agent_id+1))
			if i < len(self.state_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.state_axs)-1:
				a.set_xlabel('$t$')

	def clear_act_plot(self):
		for a in self.act_axs:
			a.clear()
		for (i, a) in enumerate(self.act_axs):
			a.set_ylabel('$u_%i$' % (i+1))
			if i == 0:
				a.set_title('Agent %i' % (self.agent_id+1))
			if i < len(self.act_axs)-1:
				a.xaxis.set_ticklabels([])
			if i == len(self.act_axs)-1:
				a.set_xlabel('$t$')

	def update_prev_trajs(self, state_traj=None, act_traj=None):
		# state_traj is a list of numpy arrays. Each numpy array is the closed-loop trajectory of an agent.
		if state_traj is not None:
			self.prev_pos_cl = [s[self.pos_dims,:] for s in state_traj]
			self.prev_state_cl = state_traj
		if act_traj is not None:
			self.prev_act_cl = act_traj

		self.it += 1

	def plot_state_traj(self, state_cl, state_preds, t, expl_con=None, shade=False):
		self.clear_state_plots()

		if expl_con is not None and 'lin' in expl_con:
			lin_con = expl_con['lin']
		if expl_con is not None and 'ell' in expl_con:
			ell_con = expl_con['ell']

		if shade:
			p1 = np.linspace(self.plot_lims[0][0], self.plot_lims[0][1], 15)
			p2 = np.linspace(self.plot_lims[1][0], self.plot_lims[1][1], 15)
			P1, P2 = np.meshgrid(p1, p2)

		pos_preds = state_preds[self.pos_dims, :]
		pred_len = state_preds.shape[1]

		# Pick out position and velocity dimensions
		pos_cl = state_cl[self.pos_dims, :]
		cl_len = state_cl.shape[1]

		c_pred = [matplotlib.cm.get_cmap('jet')(i*(1./(pred_len-1))) for i in range(pred_len)]

		# Plot entire previous closed loop trajectory for comparison
		if self.prev_pos_cl is not None:

			n_a = len(self.prev_pos_cl)
			c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]
			for (i, s) in enumerate(self.prev_pos_cl):
				plot_t = min(t, s.shape[1]-1)
				self.pos_ax.plot(s[0,:], s[1,:], '.', c=c[i], markersize=1)
				self.pos_ax.plot(s[0,plot_t], s[1,plot_t], 'o', c=c[i])
				# self.pos_ax.text(s[0,0]+0.1, s[1,0]+0.1, 'Agent %i' % (i+1), fontsize=12, bbox=dict(facecolor='white', alpha=1.))

			agent_prev_cl = self.prev_pos_cl[self.agent_id]

			# Plot the ellipsoidal constraint
			if expl_con is not None and 'ell' in expl_con:
				for i in range(t, t+pred_len):
					plot_t = min(i, agent_prev_cl.shape[1]-1)
					self.pos_ax.plot(agent_prev_cl[0,plot_t]+ell_con[plot_t]*np.cos(np.linspace(0,2*np.pi,100)),
						agent_prev_cl[1,plot_t]+ell_con[plot_t]*np.sin(np.linspace(0,2*np.pi,100)), '--', linewidth=0.7, c=c_pred[i-t])
			if expl_con is not None and 'lin' in expl_con:
				boundary_x = np.linspace(self.plot_lims[0][0], self.plot_lims[0][1], 50)
				# for i in range(t, t+pred_len):
				for i in range(t+pred_len-1, t-1, -1): # Go backwards so that earlier stuff is plotted on top
					plot_t = min(i, agent_prev_cl.shape[1]-1)
					H = lin_con[0][plot_t]
					g = lin_con[1][plot_t]
					for j in range(H.shape[0]):
						boundary_y = (-H[j,0]*boundary_x - g[j])/(H[j,1]+1e-10)
						self.pos_ax.plot(boundary_x, boundary_y, '--', linewidth=0.7, c=c_pred[i-t])

					if shade:
						for j in range(P1.shape[0]):
							for k in range(P2.shape[1]):
								test_pt = np.array([P1[j,k], P2[j,k]])
								if np.all(H.dot(test_pt) + g <= 0):
									self.pos_ax.plot(test_pt[0], test_pt[1], '.', c=c_pred[i-t], markersize=1)

		# Plot the closed loop position trajectory up to this iteration and the optimal solution at this iteration

		self.pos_ax.scatter(pos_preds[0,:], pos_preds[1,:], marker='.', c=c_pred)
		self.pos_ax.plot(pos_cl[0,:], pos_cl[1,:], 'k.')

		# Plot the closed loop state trajectory up to this iteration and the optimal solution at this iteration
		for (i, a) in enumerate(self.state_axs):
			if self.prev_state_cl is not None:
				l = self.prev_state_cl[self.agent_id].shape[1]
				a.plot(range(l), self.prev_state_cl[self.agent_id][i,:], 'b.')
			a.plot(range(t, t+pred_len), state_preds[i,:], 'g.')
			a.plot(range(t, t+pred_len), state_preds[i,:], 'g')
			a.plot(range(cl_len), state_cl[i,:], 'k.')

		try:
			self.pos_fig.canvas.draw()
			self.state_fig.canvas.draw()
		except KeyboardInterrupt:
			sys.exit()

		# Save plots if plot_dir was specified
		if self.plot_dir is not None:
			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('pos', f_name))
			self.pos_fig.savefig('/'.join((self.plot_dir, f_name)))

			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('state', f_name))
			self.state_fig.savefig('/'.join((self.plot_dir, f_name)))

	def plot_act_traj(self, act_cl, act_preds, t):
		self.clear_act_plot()

		cl_len = act_cl.shape[1]
		pred_len = act_preds.shape[1]

		# Plot the closed loop input trajectory up to this iteration and the optimal solution at this iteration
		for (i, a) in enumerate(self.act_axs):
			if self.prev_act_cl is not None:
				l = self.prev_act_cl[self.agent_id].shape[1]
				a.plot(range(l), self.prev_act_cl[self.agent_id][i,:], 'b.')
			a.plot(range(t, t+pred_len), act_preds[i,:], 'g.')
			a.plot(range(t, t+pred_len), act_preds[i,:], 'g')
			a.plot(range(cl_len), act_cl[i,:], 'k.')

		try:
			self.act_fig.canvas.draw()
		except KeyboardInterrupt:
			sys.exit()

		if self.plot_dir is not None:
			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('act', f_name))
			self.act_fig.savefig('/'.join((self.plot_dir, f_name)))

	# A tool for inspecting the trajectory at an iteration, when this function is called, the program will enter into a while loop which waits for user input to inspect the trajectory
	def traj_inspector(self, start_t, xcl, x_preds, u_preds, expl_con=None):
		t = start_t

		# Get the max time of the trajectory
		end_times = [xcl_shape[1]-1]
		if expl_con is not None and 'lin' in expl_con:
			end_times.append(len(lin_con[0])-1)
		if expl_con is not None and 'ell' in expl_con:
			end_times.append(len(ball_con)-1)
		max_time = np.amax(end_times)

		print('t = %i' % t)
		print('Press q to exit, f/b to move forward/backwards through iteration time steps')
		while True:
			input = raw_input('(debug) ')
			# Quit inspector
			if input == 'q':
				break
			# Move forward 1 time step
			elif input == 'f':
				if t == max_time:
					print('End reached')
					continue
				else:
					t += 1
					print('t = %i' % t)
					self.plot_state_traj(xcl[:,:min(t,start_t)], x_preds[min(t,start_t-1)], t, expl_con=expl_con, shade=True)
			# Move backward 1 time step
			elif input == 'b':
				if t == 0:
					print('Start reached')
					continue
				else:
					t -= 1
					print('t = %i' % t)
					self.plot_state_traj(xcl[:,:min(t,start_t)], x_preds[min(t,start_t-1)], t, expl_con=expl_con, shade=True)
			else:
				print('Input not recognized')
				print('Press q to exit, f/b to move forward/backwards through iteration time steps')
