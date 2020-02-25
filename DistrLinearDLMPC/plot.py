import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
import copy
import pickle
import pdb
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

filehandler = open('lmpc_object.pkl', 'rb')
#filehandler = codecs.open('lmpc_object.pkl', 'r', encoding='ISO-8859-1')
lmpc = pickle.load(filehandler)

# =========================================================
# Plot closed-loop trajectories at iteration j
# =========================================================
it = 19
plt.figure()
xcl = np.array(lmpc.SS[0]).T
plt.plot(xcl[0,:], xcl[1,:], 'sr', label='sampled Safe Set 1')
plt.plot(xcl[2,:], xcl[3,:], 'or', label='sampled Safe Set 2')
plt.plot(xcl[4,:], xcl[5,:], '*r', label='sampled Safe Set 3')
for i in range(1,it-1):
	xcl = np.array(lmpc.SS[i]).T
	plt.plot(xcl[0,:], xcl[1,:], 'sr')
	plt.plot(xcl[2,:], xcl[3, :], 'or')
	plt.plot(xcl[4,:], xcl[5,:], '*r')

i = it-1
xcl = np.array(lmpc.SS[i]).T
plt.plot(xcl[0,:], xcl[1,:], '-sb', label='LMPC closed-loop 1')
plt.plot(xcl[2,:], xcl[3,:], '-ob', label='LMPC closed-loop 2')
plt.plot(xcl[4,:], xcl[5,:], '-*b', label='LMPC closed-loop 3')
plt.plot(lmpc.xOpt[0,:], lmpc.xOpt[1,:], '--sk', label='Optimal trajectory 1')
plt.plot(lmpc.xOpt[2,:], lmpc.xOpt[3,:], '--ok', label='Optimal trajectory 2')
plt.plot(lmpc.xOpt[4,:], lmpc.xOpt[5,:], '--*k', label='Optimal trajectory 3')
plt.legend(fontsize=16)

plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
plt.xlim([-16,1])
plt.ylim([-0.5,7])
plt.show()

# =========================================================
# Plot closed-loop trajectories
# =========================================================
it = lmpc.it
plt.figure()
xcl = np.array(lmpc.SS[0]).T
plt.plot(xcl[0,:], xcl[1,:], '-dg', label='Initial feasible trajectory')
plt.plot(xcl[2,:], xcl[3,:], '-dg', label='Initial feasible trajectory')
plt.plot(xcl[4,:], xcl[5,:], '-dg', label='Initial feasible trajectory')
for i in range(1,it-1):
	xcl = np.array(lmpc.SS[i]).T
	plt.plot(xcl[0,:], xcl[1,:], 'sr')
	plt.plot(xcl[2,:], xcl[3, :], 'sr')
	plt.plot(xcl[4,:], xcl[5,:], 'sr')

plt.plot(0, 0, 'sr', label='Stored data')

i = it-1
xcl = np.array(lmpc.SS[i]).T
plt.plot(xcl[0,:], xcl[1,:], '-ob', label='LMPC closed-loop')
plt.plot(xcl[2,:], xcl[3,:], '-ob', label='LMPC closed-loop')
plt.plot(xcl[4,:], xcl[5,:], '-ob', label='LMPC closed-loop')
plt.plot(lmpc.xOpt[0,:], lmpc.xOpt[1,:], '--*k', label='Optimal trajectory')
plt.plot(lmpc.xOpt[2,:], lmpc.xOpt[3,:], '--*k', label='Optimal trajectory')
plt.plot(lmpc.xOpt[4,:], lmpc.xOpt[5,:], '--*k', label='Optimal trajectory')
plt.legend(fontsize=16)

plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)

# =========================================================
# Plot iteration cost
# =========================================================
plt.figure()
totCost = []
for i in range(0,it):
	xcl = np.array(lmpc.SS[i]).T
	totCost.append(lmpc.Qfun[i][0])

	
plt.plot(totCost, '-ob', label='Iteration Cost')
plt.plot([0, it-1], [lmpc.optCost, lmpc.optCost], '--k', label='Optimal cost')

plt.xlabel('$\mathrm{Iteration}$', fontsize=20)
plt.legend(fontsize=16)

# =========================================================
# Plot iteration cost just LMPC
# =========================================================
plt.figure()
totCost = []
for i in range(1,it):
	xcl = np.array(lmpc.SS[i]).T
	totCost.append(lmpc.Qfun[i][0])

plt.plot(range(1, it, 1),totCost, '-ob', label='Iteration Cost')
plt.plot([0, it-1], [lmpc.optCost, lmpc.optCost], '--k', label='Optimal cost')

plt.xlabel('$\mathrm{Iteration}$', fontsize=20)
plt.legend(fontsize=16)

print("Percentage deviation: ", np.abs((lmpc.optCost-totCost[-1])/lmpc.optCost)*100)

plt.show()