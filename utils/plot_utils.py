import numpy as np
import copy, pickle, pdb, time, sys, os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=False)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Trajectory animation with circular exploration constraints
def plot_agent_trajs(x, expl_con=None, r_a=None, trail=False, shade=False, plot_lims=None, save_dir=None, save_video=False, it=None):
	dpi = 200

	if expl_con is not None and 'lin' in expl_con:
		H_cl = expl_con['lin'][0]
		g_cl = expl_con['lin'][1]
		boundary_x = np.linspace(plot_lims[0][0], plot_lims[0][1], 50)
		if shade:
			p1 = np.linspace(plot_lims[0][0], plot_lims[0][1], 10)
			p2 = np.linspace(plot_lims[1][0], plot_lims[1][1], 10)
			P1, P2 = np.meshgrid(p1, p2)
	if expl_con is not None and 'ell' in expl_con:
		ell_con = expl_con['ell']

	n_a = len(x)

	if r_a is None:
		r_a = [0 for _ in range(n_a)]

	traj_lens = [x[i].shape[1] for i in range(n_a)]
	end_flags = [False for i in range(n_a)]

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	if save_video:
		imgs = []
		fig_a = plt.figure(dpi=dpi)
	else:
		fig_a = []

	plt.ion()
	fig = plt.figure(dpi=dpi)
	ax = fig.gca()
	if plot_lims is not None:
		ax.set_xlim(plot_lims[0])
		ax.set_ylim(plot_lims[1])

	t = 0
	text_vars = []
	while not np.all(end_flags):
		plt.figure(fig.number)
		if len(text_vars) != 0:
			for txt in text_vars:
				txt.remove()
			text_vars = []

		if not trail:
			ax.clear()
			if plot_lims is not None:
				ax.set_xlim(plot_lims[0])
				ax.set_ylim(plot_lims[1])

		for i in range(n_a):
			# plot_t = min(t, traj_lens[i]-1)
			if t <= traj_lens[i]-1:
				ax.plot(x[i][0,t], x[i][1,t], '.', c=c[i])
				text_vars.append(ax.text(x[i][0,t]+r_a[i]+0.05,
					x[i][1,t]+r_a[i]+0.05, str(i+1), fontsize=12,
					bbox=dict(facecolor='white', alpha=1.)))
				if r_a[i] > 0:
					ax.plot(x[i][0,t]+r_a[i]*np.cos(np.linspace(0,2*np.pi,100)),
						x[i][1,t]+r_a[i]*np.sin(np.linspace(0,2*np.pi,100)),
						c=c[i])
			else:
				text_vars.append(ax.text(x[i][0,-1]+r_a[i]+0.05,
					x[i][1,-1]+r_a[i]+0.05, str(i+1), fontsize=12,
					bbox=dict(facecolor='white', alpha=1.)))
					# ax.plot(x[i][0,t]+l*np.array([-1, -1, 1, 1, -1]), x[i][1,t]+l*np.array([-1, 1, 1, -1, -1]), c=c[i])

			if expl_con is not None and 'ell' in expl_con:
				ax.plot(x[i][0,t]+ell_con[i,t]*np.cos(np.linspace(0,2*np.pi,100)),
					x[i][1,t]+ell_con[i,t]*np.sin(np.linspace(0,2*np.pi,100)),
					'--', c=c[i], linewidth=0.7)
			if expl_con is not None and 'lin' in expl_con:
				H = H_cl[i][t]
				g = g_cl[i][t]
				boundary_y = (-H[0,0]*boundary_x - g[0])/(H[0,1]+1e-10)

				if shade:
					for j in range(P1.shape[0]):
						for k in range(P2.shape[1]):
							test_pt = np.array([P1[j,k], P2[j,k]])
							if np.all(H.dot(test_pt) + g <= 0):
								ax.plot(test_pt[0], test_pt[1], '.', c=c[i], markersize=0.5)

				ax.plot(boundary_x, boundary_y, '--', c=c[i])

			if not end_flags[i] and t >= traj_lens[i]-1:
				end_flags[i] = True

		if it is not None:
			ax.set_title('Iteration: %i' % it)
		ax.set_aspect('equal')

		fig.canvas.draw()
		# time.sleep(0.02)
		plt.ioff()

		if save_video:
			width, height = fig.get_size_inches() * fig.get_dpi()
			img_arr = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
			plt.figure(fig_a.number)
			img = plt.imshow(img_arr, animated=True)
			plt.axis('off')
			imgs.append([img])

		t += 1

	if save_video:
		ani = animation.ArtistAnimation(fig_a, imgs, interval=50, blit=True)
		ani.save('/'.join((save_dir, 'it_%i.mp4' % it)), dpi=dpi)

	if save_dir is not None and not save_video:
		fig.savefig('/'.join((save_dir, 'it_%i.png' % it)))

	return fig, fig_a



def plot_ts(x, title=None, x_label=None, y_labels=None):
	plt.figure()
	for i in range(x.shape[0]):
		plt.subplot(x.shape[0], 1, i+1)
		plt.plot(range(x.shape[1]), x[i,:])
		if i == 0 and title is not None:
			plt.title(title)
		if i == x.shape[0]-1 and x_label is not None:
			plt.xlabel(x_label)
		if y_labels is not None:
			plt.ylabel(y_labels[i])

class updateable_plot(object):
	def __init__(self, n_seq, title=None, x_label=None, y_label=None):
		plt.ion()
		self.fig = plt.figure()
		self.ax = plt.gca()
		self.ax.set_xlim([0, 5])
		self.ax.set_ylim([0, 5])

		self.n_seq = n_seq
		self.title = title
		self.x_label = x_label
		self.y_label = y_label

		self.data = [np.empty((2,1)) for _ in range(n_seq)]
		self.c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_seq-1))) for i in range(n_seq)]

	def clear(self):
		self.ax.clear()

	def update(self, d, seq_idx):
		self.data[seq_idx] = np.append(self.data[seq_idx], d, axis=1)
		self.ax.clear()
		for i in range(self.n_seq):
			self.ax.plot(self.data[i][0,:], self.data[i][1,:], '.-', c=c[i])
			if self.title is not None:
				self.set_title(self.title)
			if self.x_label is not None:
				self.set_xlabel(self.x_label)
			if self.y_label is not None:
				self.set_xlabel(self.y_label)

		self.fig.canvas.draw()

class updateable_ts(object):
	def __init__(self, n_seq, title=None, x_label=None, y_label=None):
		plt.ion()
		self.fig = plt.figure()
		self.axs = [self.fig.add_subplot(n_seq, 1, i+1) for i in range(n_seq)]
		for (i, a) in enumerate(self.axs):
			if y_label is not None:
				a.set_ylabel(y_label[i])
			if title is not None and i == 0:
				a.set_title(title)
			if x_label is not None and i == len(self.axs)-1:
				a.set_xlabel(x_label)

		self.fig.canvas.draw()

		self.n_seq = n_seq
		self.title = title
		self.x_label = x_label
		self.y_label = y_label

		# self.data = [np.empty((2,1)) for _ in range(n_seq)]
		# self.c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_seq-1))) for i in range(n_seq)]

	def clear(self):
		for a in self.axs:
			a.clear()

	def update(self, d):
		t = range(d.shape[1])
		for (i, a) in enumerate(self.axs):
			a.plot(t, d[i,:])

		self.fig.canvas.draw()
