import numpy as np
import copy, pickle, pdb, time, sys, os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

class lmpc_visualizer(object):
	def __init__(self, pos_dims, n_state_dims, n_act_dims, agent_id, n_agents, plot_lims=None, plot_dir=None):
		if len(pos_dims) > 2:
			raise(ValueError('Can only plot 2 position dimensions'))
		self.agent_id = agent_id
		self.pos_dims = pos_dims
		self.n_state_dims = n_state_dims
		self.n_act_dims = n_act_dims
		self.plot_dir = plot_dir
		self.plot_lims = plot_lims

		self.n_a = n_agents
		self.c = [matplotlib.cm.get_cmap('jet')(i*(1./(self.n_a-1))) for i in range(self.n_a)]

		self.ax_labels = ['$a$', '$df$', '$v$', '$\phi$']

		x_data = []
		y_data = []

		plt.ion()
		f_w = 12
		f_h = 7
		self.fig = plt.figure(figsize=(f_w,f_h))
		# self.fig.subplots_adjust(bottom=0.3)

		# Initialize position plot
		self.pos_ax = self.fig.add_axes([0.03, 0.03, 0.48, 0.95])

		self.prev_trajs_xy = []
		self.prev_pos = []
		for i in range(self.n_a):
			t, = self.pos_ax.plot(x_data, y_data, '.', markersize=1, c=self.c[i])
			p, = self.pos_ax.plot(x_data, y_data, 'o', c=self.c[i])
			self.prev_trajs_xy.append(t)
			self.prev_pos.append(p)

		self.pred_xy = []
		for i in range(50):
			pr, = self.pos_ax.plot(x_data, y_data, 'o', markersize=2)
			self.pred_xy.append(pr)

		self.expl_bound_xy = []
		for i in range(50):
			ex, = self.pos_ax.plot(x_data, y_data, linewidth=0.7)
			self.expl_bound_xy.append(ex)

		self.traj_line, = self.pos_ax.plot(x_data, y_data, 'k.')
		self.ss_line, = self.pos_ax.plot(x_data, y_data, 'ko', fillstyle='none', markersize=5)

		if self.plot_lims is not None:
			self.pos_ax.set_xlim(self.plot_lims[0])
			self.pos_ax.set_ylim(self.plot_lims[1])
		self.pos_ax.set_xlabel('$x$')
		self.pos_ax.set_ylabel('$y$')

		# Initialize timeseries plot
		self.ts_axs = [self.fig.add_axes([0.58, 0.03+i*1.0/(n_state_dims+n_act_dims-2), 0.4, 1.0/(n_state_dims+n_act_dims-2)-0.05]) for i in range(n_act_dims+n_state_dims-2)]
		self.prev_traj_ts = []
		self.curr_traj_ts = []
		self.ss_ts = []
		self.pred_ts = []
		for (i, a) in enumerate(self.ts_axs):
			a.set_ylabel(self.ax_labels[i])
			pt, = a.plot(x_data, y_data, 'b.-', markersize=2)
			ct, = a.plot(x_data, y_data, 'k.-', markersize=2)
			pr, = a.plot(x_data, y_data, 'g.-', markersize=2)
			ss, = a.plot(x_data, y_data, 'ko', fillstyle='none', markersize=5)
			self.prev_traj_ts.append(pt)
			self.curr_traj_ts.append(ct)
			self.ss_ts.append(ss)
			self.pred_ts.append(pr)
			if i > 0:
				a.set_xticks([])
			if i == 0:
				a.set_xlabel('$t$')

		self.fig.canvas.set_window_title('Agent %i' % (agent_id+1))
		self.fig.canvas.draw()

		self.prev_pos_cl = None
		self.prev_state_cl = None
		self.prev_act_cl = None

		self.it = 0

	def set_plot_dir(self, plot_dir):
		self.plot_dir = plot_dir

	def clear_plots(self):
		self.pos_ax.clear()
		if self.plot_lims is not None:
			self.pos_ax.set_xlim(self.plot_lims[0])
			self.pos_ax.set_ylim(self.plot_lims[1])
		self.pos_ax.set_xlabel('$x$')
		self.pos_ax.set_ylabel('$y$')

		for a in self.ts_axs:
			a.clear()
		for (i, a) in enumerate(self.ts_axs):
			a.set_ylabel(self.ax_labels[i])
			if i > 0:
				a.set_xticks([])
			if i == 0:
				a.set_xlabel('$t$')

	def update_prev_trajs(self, state_traj=None, act_traj=None):
		# state_traj is a list of numpy arrays. Each numpy array is the closed-loop trajectory of an agent.
		if state_traj is not None:
			self.prev_pos_cl = [s[self.pos_dims,:] for s in state_traj]
			self.prev_state_cl = state_traj
		if act_traj is not None:
			self.prev_act_cl = act_traj

		self.it += 1

	def plot_traj(self, state_cl, act_cl, state_preds, act_preds, t, SS, expl_con=None, shade=False):
		# self.clear_plots()

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
			for (i, s) in enumerate(self.prev_pos_cl):
				if t == 0:
					self.prev_trajs_xy[i].set_data(s[0,:], s[1,:])
				plot_t = min(t, s.shape[1]-1)
				self.prev_pos[i].set_data(s[0,plot_t], s[1,plot_t])

			agent_prev_cl = self.prev_pos_cl[self.agent_id]

			if expl_con is not None:
				boundary_x = np.linspace(self.plot_lims[0][0], self.plot_lims[0][1], 50)
				# for i in range(t, t+pred_len):
				counter = 0
				for i in range(t+pred_len-1, t-1, -1): # Go backwards so that earlier stuff is plotted on top
					plot_t = min(i, agent_prev_cl.shape[1]-1)
					H = expl_con[plot_t][0]
					g = expl_con[plot_t][1]
					for j in range(H.shape[0]):
						boundary_y = (-H[j,0]*boundary_x - g[j])/(H[j,1]+1e-10)
						self.expl_bound_xy[counter].set_data(boundary_x, boundary_y)
						self.expl_bound_xy[counter].set_color(c_pred[i-t])
						counter += 1
						# self.pos_ax.plot(boundary_x, boundary_y, '--', linewidth=0.7, c=c_pred[i-t])

					if shade:
						for j in range(P1.shape[0]):
							for k in range(P2.shape[1]):
								test_pt = np.array([P1[j,k], P2[j,k]])
								if np.all(H.dot(test_pt) + g <= 0):
									self.pos_ax.plot(test_pt[0], test_pt[1], '.', c=c_pred[i-t], markersize=1)

		# Plot the closed loop position trajectory up to this iteration and the optimal solution at this iteration
		self.ss_line.set_data(SS[0,:], SS[1,:])
		for i in range(pred_len):
			self.pred_xy[i].set_data(pos_preds[0,i], pos_preds[1,i])
			self.pred_xy[i].set_color(c_pred[i])
		self.traj_line.set_data(pos_cl[0,:], pos_cl[1,:])

		# Plot the closed loop state trajectory up to this iteration and the optimal solution at this iteration
		for (i, a) in enumerate(self.ts_axs):
			if i < self.n_act_dims:
				plot_idx = self.n_act_dims-i-1
				if self.prev_act_cl is not None:
					l = self.prev_act_cl[self.agent_id].shape[1]
					self.prev_traj_ts[i].set_data(range(l), self.prev_act_cl[self.agent_id][plot_idx,:])
				self.pred_ts[i].set_data(range(t, t+pred_len-1), act_preds[plot_idx,:])
				self.curr_traj_ts[i].set_data(range(cl_len-1), act_cl[plot_idx,:])
				a.relim()
				a.autoscale_view()
			else:
				plot_idx = (self.n_state_dims-1)-(i-self.n_act_dims)
				if self.prev_state_cl is not None:
					l = self.prev_state_cl[self.agent_id].shape[1]
					self.prev_traj_ts[i].set_data(range(l), self.prev_state_cl[self.agent_id][plot_idx,:])
					a.set_xlim([0, l+1])
				self.ss_ts[i].set_data(range(t+pred_len, t+pred_len+SS.shape[1]), SS[plot_idx,:])
				self.pred_ts[i].set_data(range(t, t+pred_len), state_preds[plot_idx,:])
				self.curr_traj_ts[i].set_data(range(cl_len-1), state_cl[plot_idx,:-1])
				a.relim()
				a.autoscale_view()
		try:
			self.fig.canvas.draw()
		except KeyboardInterrupt:
			sys.exit()

		# Save plots if plot_dir was specified
		if self.plot_dir is not None:
			f_name = 'it_%i_time_%i.png' % (self.it, t)
			if self.agent_id is not None:
				f_name = '_'.join((('agent_%i' % (self.agent_id+1)), f_name))
			f_name = '_'.join(('pos', f_name))
			self.fig.savefig('/'.join((self.plot_dir, f_name)))

	# A tool for inspecting the trajectory at an iteration, when this function is called, the program will enter into a while loop which waits for user input to inspect the trajectory
	def traj_inspector(self, xcl, ucl, x_preds, u_preds, start_t, expl_con=None):
		t = start_t

		# Get the max time of the trajectory
		end_times = [xcl_shape[1]-1]
		# if expl_con is not None and 'lin' in expl_con:
		# 	end_times.append(len(lin_con[0])-1)
		# if expl_con is not None and 'ell' in expl_con:
		# 	end_times.append(len(ball_con)-1)
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
