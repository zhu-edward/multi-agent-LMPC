import numpy as np
import copy, pickle, pdb, time, sys, os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Trajectory animation for kinematic bicycle agents
def plot_bike_agent_trajs(x, u, agents, dt, trail=False, shade=False, plot_lims=None, save_dir=None, save_video=False, it=None):
	dpi = 100
	n_a = len(x)
	r_a = [agents[i].get_collision_buff_r() for i in range(n_a)] # Agents are circles with radius r_a

	traj_lens = [x[i].shape[1] for i in range(n_a)]
	end_flags = [False for i in range(n_a)]

	c = [matplotlib.cm.get_cmap('jet')(i*(1./(n_a-1))) for i in range(n_a)]

	if save_video:
		imgs = []
		fig_a = plt.figure(dpi=dpi)
	else:
		fig_a = []

	plt.ion()
	# fig = plt.figure(dpi=dpi, figsize=(10,5))
	fig = plt.figure(figsize=(12,7))
	# ax = fig.gca()
	ax = fig.add_axes([0.03, 0.03, 0.48, 0.95])
	if plot_lims is not None:
		ax.set_xlim(plot_lims[0])
		ax.set_ylim(plot_lims[1])

	psi_ax = fig.add_axes([0.58, 0.78, 0.4, 0.2])
	psi_ax.set_xticks([])
	v_ax = fig.add_axes([0.58, 0.53, 0.4, 0.2])
	v_ax.set_xticks([])
	df_ax = fig.add_axes([0.58, 0.28, 0.4, 0.2])
	df_ax.set_xticks([])
	a_ax = fig.add_axes([0.58, 0.03, 0.4, 0.2])

	for i in range(n_a):
		psi_ax.plot(dt*np.arange(0,traj_lens[i]), x[i][2,:])
		v_ax.plot(dt*np.arange(0,traj_lens[i]), x[i][3,:])
		df_ax.plot(dt*np.arange(0,traj_lens[i]), u[i][0,:])
		a_ax.plot(dt*np.arange(0,traj_lens[i]), u[i][1,:])

		psi_ax.set_ylabel('psi')
		v_ax.set_ylabel('v')
		df_ax.set_ylabel('df')
		a_ax.set_ylabel('a')
		a_ax.set_xlabel('t')

		fig.canvas.draw()

	t = 0
	text_lines = []
	car_lines = []
	wheel_lines = []
	bound_lines = []
	while not np.all(end_flags):
		plt.figure(fig.number)
		# Clear text labels and car boundary from previous plot
		if t > 0:
			if len(text_lines) > 0:
				for txt in text_lines:
					txt.remove()
				text_lines = []
			if len(car_lines) > 0:
				for l in car_lines:
					l.remove()
				car_lines = []
			if len(wheel_lines) > 0:
				for l in wheel_lines:
					l.remove()
				wheel_lines = []
			if len(bound_lines) > 0:
				for l in bound_lines:
					l.remove()
				bound_lines = []

		if not trail:
			ax.clear()
			if plot_lims is not None:
				ax.set_xlim(plot_lims[0])
				ax.set_ylim(plot_lims[1])

		for i in range(n_a):
			x_t = x[i][:,min(t, traj_lens[i]-1)]
			u_t = u[i][:,min(t, traj_lens[i]-1)]
			l_f = agents[i].l_f
			l_r = agents[i].l_r
			w = agents[i].w

			car_x = [x_t[0] + l_f*np.cos(x_t[2]) + w*np.sin(x_t[2])/2,
				x_t[0] + l_f*np.cos(x_t[2]) - w*np.sin(x_t[2])/2,
				x_t[0] - l_r*np.cos(x_t[2]) - w*np.sin(x_t[2])/2,
				x_t[0] - l_r*np.cos(x_t[2]) + w*np.sin(x_t[2])/2,
				x_t[0] + l_f*np.cos(x_t[2]) + w*np.sin(x_t[2])/2]
			car_y = [x_t[1] + l_f*np.sin(x_t[2]) - w*np.cos(x_t[2])/2,
				x_t[1] + l_f*np.sin(x_t[2]) + w*np.cos(x_t[2])/2,
				x_t[1] - l_r*np.sin(x_t[2]) + w*np.cos(x_t[2])/2,
				x_t[1] - l_r*np.sin(x_t[2]) - w*np.cos(x_t[2])/2,
				x_t[1] + l_f*np.sin(x_t[2]) - w*np.cos(x_t[2])/2]

			wheel_x = [x_t[0] + l_f*np.cos(x_t[2]) + 0.2*np.cos(x_t[2]+u_t[0]), x_t[0] + l_f*np.cos(x_t[2]) - 0.2*np.cos(x_t[2]+u_t[0])]
			wheel_y = [x_t[1] + l_f*np.sin(x_t[2]) + 0.2*np.sin(x_t[2]+u_t[0]), x_t[1] + l_f*np.sin(x_t[2]) - 0.2*np.sin(x_t[2]+u_t[0])]

			ax.plot(x_t[0], x_t[1], '.', c=c[i], markersize=1)

			car_lines.append(ax.plot(car_x, car_y, c=c[i], linewidth=0.75).pop(0))
			wheel_lines.append(ax.plot(wheel_x, wheel_y, c=c[i], linewidth=0.75).pop(0))

			bound_lines.append(ax.plot(x_t[0]+r_a[i]*np.cos(np.linspace(0,2*np.pi,100)),
				x_t[1]+r_a[i]*np.sin(np.linspace(0,2*np.pi,100)), '--',
				c=c[i], linewidth=0.75).pop(0))

			text_lines.append(ax.text(x_t[0]+r_a[i]+0.5,
				x_t[1]+r_a[i]+0.5, str(i+1), fontsize=6,
				bbox=dict(facecolor='white', alpha=1.)))

			if not end_flags[i] and t >= traj_lens[i]-1:
				end_flags[i] = True

		if it is not None:
			ax.set_title('Iteration: %i, Time: %g s' % (it, t*dt))
		ax.set_aspect('equal')

		fig.canvas.draw()
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
