
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import gridspec


r_wo_vi_ = np.load('new_exp/w_reward.npy')
r_vi_ = np.load('new_exp/wo_reward.npy')


def moving_average(window = 10):
	rvi = []
	rwovi = []
	for i in range(len(r_vi_) - window):
		rvi.append(np.average(r_vi_[i : i*window]))
		rwovi.append(np.average(r_wo_vi_[i : i*window]))
	return rvi, rwovi


# fig, axes = plt.subplots(5,2,figsize=(14,20))
# ax = axes.flatten()

# r_vi, r_wo_vi = r_vi_, r_wo_vi_
# ax[0].plot(r_vi, 'b', label='with vi')
# ax[0].plot(r_wo_vi,'r', label='without vi')
# ax[0].legend()

# r_vi, r_wo_vi = moving_average(2)
# ax[1].plot(r_vi, 'b', label='with vi')
# ax[1].plot(r_wo_vi,'r', label='without vi')
# ax[1].legend()

# r_vi, r_wo_vi = moving_average(3)
# ax[2].plot(r_vi, 'b', label='with vi')
# ax[2].plot(r_wo_vi,'r', label='without vi')
# ax[2].legend()

# r_vi, r_wo_vi = moving_average(5)
# ax[3].plot(r_vi, 'b', label='with vi')
# ax[3].plot(r_wo_vi,'r', label='without vi')
# ax[3].legend()

# r_vi, r_wo_vi = moving_average(10)
# ax[4].plot(r_vi, 'b', label='with vi')
# ax[4].plot(r_wo_vi,'r', label='without vi')
# ax[4].legend()

# plt.show()


fig, ax = plt.subplots(1,2,figsize=(14,5))

r_vi, r_wo_vi = moving_average(5)
ax[0].plot(r_vi, marker='o', markersize=8, color='orange', linewidth=2, alpha=0.7, label='with vi')
ax[0].plot(r_wo_vi,'r', marker='o', markersize=8, color='blue', linewidth=2, alpha=0.7, label='without vi')
ax[0].legend()
# marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4
count = 9300
height = [40000, 40000 - count]
bars = ["without vi", "with vi"]
y_pos = np.arange(len(bars))
ax[1].bar(y_pos, height, color=['blue','orange'], width = 0.3, edgecolor = 'black', capsize=7)
# ax[1].set_xticks([0, 1], ["without vi", "with vi"])
plt.xticks(y_pos, bars)
fig.savefig('new_exp/results733.png')
# plt.show()








fig, ax = plt.subplots(1,2,figsize=(14,5))
fig = plt.figure(figsize=(14, 5)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
ax0 = plt.subplot(gs[0])
r_vi, r_wo_vi = moving_average(5)
ax0.plot(r_vi, marker='o', markersize=8, color='orange', linewidth=2, alpha=0.7, label='with vi')
ax0.plot(r_wo_vi,'r', marker='o', markersize=8, color='blue', linewidth=2, alpha=0.7, label='without vi')
ax0.legend()


ax1 = plt.subplot(gs[1])
count = 9300
height = [40000, 40000 - count]
bars = ["without vi", "with vi"]
y_pos = np.arange(len(bars))
ax1.bar(y_pos, height, color=['blue','orange'], width = 0.3, edgecolor = 'black', capsize=7)

# ax1.set_xticks([0, 1], ["without vi", "with vi"])
plt.xticks(y_pos, bars)

ax1.set_ylabel('sample size')
ax1.legend()






fig.savefig('new_exp/results733.png')
# plt.show()



