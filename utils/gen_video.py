import numpy as np
import copy, pickle, pdb, time, sys, os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.animation as manimation
import matplotlib.pyplot as plt

FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=10)

BASE_DIR = '/'.join(str.split(os.path.abspath('') , '/')[:-1])
sys.path.append(BASE_DIR)
DATA_DIR = BASE_DIR + '/out'

def main():
    exp_dir = '/2020-03-15_16-39-46'

    it = 20
    n_a = 3
    dt = 0.1
    l_f = 0.5
    l_r = 0.5
    w = 0.5
    r = 0.759

    it_dir = DATA_DIR + exp_dir + ('/it_%i' % (it))

    x_cls = pickle.load(open(it_dir + '/x_cls.pkl', 'rb'), encoding='latin1')
    u_cls = pickle.load(open(it_dir + '/u_cls.pkl', 'rb'), encoding='latin1')
    x_ol = pickle.load(open(it_dir + '/x_ol.pkl', 'rb'), encoding='latin1')
    u_ol = pickle.load(open(it_dir + '/u_ol.pkl', 'rb'), encoding='latin1')

    n_cls = len(x_cls)

    agent_colors = np.array([[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.6350, 0.0780, 0.1840]])

    x_lim = [-6, 6]
    y_lim = [-6, 6]
    pos_fig_w = 7
    pos_fig_h = 7

    dpi = 100

    plot_cls = range(n_cls)
    # plot_cls = [0]

    for i in plot_cls:
        imgs = []
        it_x_cl = x_cls[i]
        it_u_cl = u_cls[i]

        agent_lens = [cl.shape[1] for cl in it_x_cl]
        max_len = np.amax(agent_lens)

        pos_fig = plt.figure(figsize=(pos_fig_w, pos_fig_h), dpi=dpi)
        pos_ax = pos_fig.gca()

        pos_ax.plot([], [])
        pos_ax.set_xlabel('$x$ [m]', fontsize=15)
        pos_ax.set_xlabel('$y$ [m]', fontsize=15)
        pos_ax.set_xlim(x_lim)
        pos_ax.set_ylim(y_lim)
        pos_ax.set_xticks([-5, 0, 5])
        pos_ax.set_yticks([-5, 0, 5])
        plt.setp(pos_ax.get_xticklabels(), fontsize=15)
        plt.setp(pos_ax.get_yticklabels(), fontsize=15)
        pos_ax.set_aspect('equal')
        pos_fig.canvas.draw()

        vid_name = 'it_%i.mp4' % (i)

        with writer.saving(pos_fig, vid_name, dpi):
            for t in range(max_len):
            # for t in range(50):
                print('it %i, t %i' % (i,t))
                pos_ax.clear()

                for j in range(n_a):
                    x = it_x_cl[j][0,:min(t+1,agent_lens[j])]
                    y = it_x_cl[j][1,:min(t+1,agent_lens[j])]
                    psi = it_x_cl[j][2,:min(t+1,agent_lens[j])]
                    df = it_u_cl[j][0,:min(t+1,agent_lens[j])]

                    # car_x = [x[-1] + l_f*np.cos(psi[-1]),
                    #     x[-1] - l_r*np.cos(psi[-1])]
                    # car_y = [y[-1] + l_f*np.sin(psi[-1]),
                    #     y[-1] - l_r*np.sin(psi[-1])]

                    # car_x = [x[-1] + l_f*np.cos(psi[-1]) + w*np.sin(psi[-1])/2,
                    #     x[-1] + l_f*np.cos(psi[-1]) - w*np.sin(psi[-1])/2,
                    #     x[-1] - l_r*np.cos(psi[-1]) - w*np.sin(psi[-1])/2,
                    #     x[-1] - l_r*np.cos(psi[-1]) + w*np.sin(psi[-1])/2,
                    #     x[-1] + l_f*np.cos(psi[-1]) + w*np.sin(psi[-1])/2]
                    # car_y = [y[-1] + l_f*np.sin(psi[-1]) - w*np.cos(psi[-1])/2,
                    #     y[-1] + l_f*np.sin(psi[-1]) + w*np.cos(psi[-1])/2,
                    #     y[-1] - l_r*np.sin(psi[-1]) + w*np.cos(psi[-1])/2,
                    #     y[-1] - l_r*np.sin(psi[-1]) - w*np.cos(psi[-1])/2,
                    #     y[-1] + l_f*np.sin(psi[-1]) - w*np.cos(psi[-1])/2]

                    car_x = np.array([x[-1] + l_f*np.cos(psi[-1]) + w*np.sin(psi[-1])/2,
                        x[-1] + l_f*np.cos(psi[-1]) - w*np.sin(psi[-1])/2,
                        x[-1] - l_r*np.cos(psi[-1]) - w*np.sin(psi[-1])/2,
                        x[-1] - l_r*np.cos(psi[-1]) + w*np.sin(psi[-1])/2]).reshape((-1,1))
                    car_y = np.array([y[-1] + l_f*np.sin(psi[-1]) - w*np.cos(psi[-1])/2,
                        y[-1] + l_f*np.sin(psi[-1]) + w*np.cos(psi[-1])/2,
                        y[-1] - l_r*np.sin(psi[-1]) + w*np.cos(psi[-1])/2,
                        y[-1] - l_r*np.sin(psi[-1]) - w*np.cos(psi[-1])/2]).reshape((-1,1))
                    car_xy = np.hstack((car_x, car_y))
                    car_rec = matplotlib.patches.Polygon(car_xy, alpha=0.5, fc=agent_colors[j], ec=agent_colors[j], zorder=10)

                    wheel_x = [x[-1] + l_f*np.cos(psi[-1]) + 0.15*np.cos(psi[-1]+df[-1]), x[-1] + l_f*np.cos(psi[-1]) - 0.15*np.cos(psi[-1]+df[-1])]
                    wheel_y = [y[-1] + l_f*np.sin(psi[-1]) + 0.15*np.sin(psi[-1]+df[-1]), y[-1] + l_f*np.sin(psi[-1]) - 0.15*np.sin(psi[-1]+df[-1])]

                    top_ang = np.linspace(0, np.pi, 100)
                    bottom_ang = np.linspace(0, -np.pi, 100)
                    bound_x = r*np.cos(top_ang) + x[-1]
                    top_y = r*np.sin(top_ang) + y[-1]
                    bottom_y = r*np.sin(bottom_ang) + y[-1]
                    pos_ax.plot(bound_x, top_y, color=agent_colors[j])
                    pos_ax.plot(bound_x, bottom_y, color=agent_colors[j])
                    # pos_ax.fill_between(bound_x, top_y, bottom_y, color=agent_colors[j], alpha=0.5)

                    pos_ax.scatter(x, y, s=3, color=agent_colors[j].reshape((1,-1)), alpha=0.5, label=('Agent %i' % (j+1)))

                    # pos_ax.plot(car_x, car_y, color=agent_colors[j])
                    pos_ax.add_patch(car_rec)
                    pos_ax.plot(wheel_x, wheel_y, linewidth=2, color=agent_colors[j])

                # pos_ax.set_title('Iteration: %i, Time: %g s' % (i, t*dt), fontsize=15)
                pos_ax.set_title('Time: %g s' % (t*dt), fontsize=20)
                pos_ax.set_xlabel('$x$ [m]', fontsize=15)
                pos_ax.set_ylabel('$y$ [m]', fontsize=15)
                pos_ax.set_xlim(x_lim)
                pos_ax.set_ylim(y_lim)
                pos_ax.set_aspect('equal')
                pos_ax.set_xticks([-5, 0, 5])
                pos_ax.set_yticks([-5, 0, 5])
                plt.setp(pos_ax.get_xticklabels(), fontsize=15)
                plt.setp(pos_ax.get_yticklabels(), fontsize=15)

                pos_fig.canvas.draw()

                writer.grab_frame()

if __name__== "__main__":
    main()
