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
	dpi = 500
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
	fig = plt.figure(figsize=(12,7))
	ax = fig.add_axes([0.03, 0.03, 0.48, 0.95])
	psi_ax = fig.add_axes([0.58, 0.78, 0.4, 0.2])
	psi_ax.set_xticks([])
	v_ax = fig.add_axes([0.58, 0.53, 0.4, 0.2])
	v_ax.set_xticks([])
	df_ax = fig.add_axes([0.58, 0.28, 0.4, 0.2])
	df_ax.set_xticks([])
	a_ax = fig.add_axes([0.58, 0.03, 0.4, 0.2])

	psi_ax.set_ylabel('$\psi$')
	v_ax.set_ylabel('$v$')
	df_ax.set_ylabel('$df$')
	a_ax.set_ylabel('$a$')
	a_ax.set_xlabel('$t$')

	xy_lines = []
	car_lines = []
	wheel_lines = []
	bound_lines = []
	psi_lines = []
	v_lines = []
	df_lines = []
	a_lines = []

	for i in range(n_a):
		xy, = ax.plot([], [], '.', c=c[i], markersize=2)
		car, = ax.plot([], [], c=c[i], linewidth=1.0)
		wheel, = ax.plot([], [], c=c[i], linewidth=0.75)
		col, = ax.plot([], [], '--', c=c[i], linewidth=0.75)
		psi, = psi_ax.plot([], [], '.', c=c[i])
		v, = v_ax.plot([], [], '.', c=c[i])
		df, = df_ax.plot([], [], '.', c=c[i])
		a, = a_ax.plot([], [], '.', c=c[i])
		xy_lines.append(xy)
		car_lines.append(car)
		wheel_lines.append(wheel)
		bound_lines.append(col)
		psi_lines.append(psi)
		v_lines.append(v)
		df_lines.append(df)
		a_lines.append(a)

	if plot_lims is not None:
		ax.set_xlim(plot_lims[0])
		ax.set_ylim(plot_lims[1])

	fig.canvas.draw()

	t = 0
	text_lines = []
	while not np.all(end_flags):
		# Clear text labels and car boundary from previous plot
		if t > 0:
			if len(text_lines) > 0:
				for txt in text_lines:
					txt.remove()
				text_lines = []

		for i in range(n_a):
			psi_lines[i].set_data(dt*np.arange(min(t+1, traj_lens[i]-1)), x[i][2,:min(t+1, traj_lens[i]-1)])
			v_lines[i].set_data(dt*np.arange(min(t+1, traj_lens[i]-1)), x[i][3,:min(t+1, traj_lens[i]-1)])
			df_lines[i].set_data(dt*np.arange(min(t+1, traj_lens[i]-1)), u[i][0,:min(t+1, traj_lens[i]-1)])
			a_lines[i].set_data(dt*np.arange(min(t+1, traj_lens[i]-1)), u[i][1,:min(t+1, traj_lens[i]-1)])

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

			bound_x = x_t[0]+r_a[i]*np.cos(np.linspace(0,2*np.pi,100))
			bound_y = x_t[1]+r_a[i]*np.sin(np.linspace(0,2*np.pi,100))

			if not trail:
				xy_lines[i].set_data(x_t[0], x_t[1])
			else:
				xy_lines[i].set_data(x[i][0,:min(t+1, traj_lens[i]-1)], x[i][1,:min(t+1, traj_lens[i]-1)])

			car_lines[i].set_data(car_x, car_y)
			wheel_lines[i].set_data(wheel_x, wheel_y)
			bound_lines[i].set_data(bound_x, bound_y)

			text_lines.append(ax.text(x_t[0]+r_a[i]+0.5,
				x_t[1]+r_a[i]+0.5, str(i+1), fontsize=6,
				bbox=dict(facecolor='white', alpha=1.)))

			if not end_flags[i] and t >= traj_lens[i]-1:
				end_flags[i] = True

			psi_ax.relim()
			psi_ax.autoscale_view()
			v_ax.relim()
			v_ax.autoscale_view()
			df_ax.relim()
			df_ax.autoscale_view()
			a_ax.relim()
			a_ax.autoscale_view()

		if it is not None:
			ax.set_title('Iteration: %i, Time: %g s' % (it, t*dt))
		ax.set_aspect('equal')

		try:
			fig.canvas.draw()
		except KeyboardInterrupt:
			sys.exit()

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
		ani.save('/'.join((save_dir, 'it_%i.mp4' % (it+1))), dpi=dpi)
		plt.close(fig_a)

	if save_dir is not None:
		fig.savefig('/'.join((save_dir, 'it_%i.png' % (it+1))))

	return fig, fig_a
