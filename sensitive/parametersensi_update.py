import numpy as np

x_del = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6]
x_add = [.9, .95, 1]
x_lambda = [.1, .25, .5, .75]
x_t = [5, 10, 15, 20, 25, 30]

pos_del=np.load('sensi/pos_del.npy')
pos_del_std=np.load('sensi/pos_del_std.npy')
neg_del=np.load('sensi/neg_del.npy')
neg_del_std=np.load('sensi/neg_del_std.npy')
pos_add=np.load('sensi/pos_add.npy')
pos_add_std=np.load('sensi/pos_add_std.npy')
neg_add=np.load('sensi/neg_add.npy')
neg_add_std=np.load('sensi/neg_add_std.npy')
lambda0=np.load('sensi/lambda0.npy')
lambda0_std=np.load('sensi/lambda0_std.npy')
t=np.load('sensi/tt.npy')
t_std=np.load('sensi/tt_std.npy')

import matplotlib.pyplot as plt

# Set global font to 'Linux Libertine'
plt.rcParams['font.family'] = 'Linux Libertine'  # 'Linux Libertine'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

datasets=['bitcoin-alpha', 'bitcoin-otc', 'Epinions', 'Slashdot', 'wiki-elec', 'wiki-RfA']
metrics=['AUC','F1','F1-micro','F1-macro']

original_colors = ['#25828e', '#35b778', '#b5dd2b', '#7200a8', '#bc3685', '#ec7853']
light_colors = ['#dbebec', '#d6f0e4', '#f0f8d4', '#e2cced', '#f1d6e6', '#fbe4dc']


for i in range(4):

    fig, ax = plt.subplots(2, 3, figsize=(18, 9), tight_layout=True)
    for ii in range(2):
        for jj in range(3):
            ax[(ii,jj)].grid(True)
    for j,dataset in enumerate(datasets):

        ax[(0, 0)].fill_between(x_del, pos_del[j][i] - pos_del_std[j][i], pos_del[j][i] + pos_del_std[j][i], alpha=0.3,
                                color=light_colors[j])
        ax[(0, 1)].fill_between(x_del, neg_del[j][i] - neg_del_std[j][i], neg_del[j][i] + neg_del_std[j][i], alpha=0.3,
                                color=light_colors[j])
        ax[(0, 2)].fill_between(x_add, pos_add[j][i] - pos_add_std[j][i], pos_add[j][i] + pos_add_std[j][i], alpha=0.3,
                                color=light_colors[j])
        ax[(1, 0)].fill_between(x_add, neg_add[j][i] - neg_add_std[j][i], neg_add[j][i] + neg_add_std[j][i], alpha=0.3,
                                color=light_colors[j])
        ax[(1, 1)].fill_between(x_t, t[j][i] - t_std[j][i], t[j][i] + t_std[j][i], alpha=0.3,
                                color=light_colors[j])
        ax[(1, 2)].fill_between(x_lambda, lambda0[j][i] - lambda0_std[j][i], lambda0[j][i] + lambda0_std[j][i],
                                alpha=0.3,color=light_colors[j])
        ax[(0, 0)].plot(x_del, pos_del[j][i], color=original_colors[j],label=dataset)
        ax[(0, 1)].plot(x_del, neg_del[j][i], color=original_colors[j],label=dataset)
        ax[(0, 2)].plot(x_add, pos_add[j][i], color=original_colors[j],label=dataset)
        ax[(1, 0)].plot(x_add, neg_add[j][i], color=original_colors[j],label=dataset)
        ax[(1, 1)].plot(x_t, t[j][i], color=original_colors[j],label=dataset)
        ax[(1, 2)].plot(x_lambda, lambda0[j][i], color=original_colors[j],label=dataset)

    ax[(0, 0)].set_xlabel(r'(a) Setting of $\epsilon_{del}^{+}$', fontsize=14)
    ax[(0, 0)].set_xticks(x_del[::2])
    ax[(0, 1)].set_xlabel(r'(b) Setting of $\epsilon_{del}^{-}$', fontsize=14)
    ax[(0, 1)].set_xticks(x_del[::2])
    ax[(0, 2)].set_xlabel(r'(c) Setting of $\epsilon_{add}^{+}$', fontsize=14)
    ax[(0, 2)].set_xticks(x_add)
    ax[(1, 0)].set_xlabel(r'(d) Setting of $\epsilon_{add}^{-}$', fontsize=14)
    ax[(1, 0)].set_xticks(x_add)
    ax[(1, 1)].set_xlabel(r'(e) Setting of $T$', fontsize=14)
    ax[(1, 1)].set_xticks(x_t)
    ax[(1, 2)].set_xlabel(r'(f) Setting of $\lambda_0$', fontsize=14)
    ax[(1, 2)].set_xticks(x_lambda)

    ax[(0, 0)].set_ylabel(metrics[i], fontsize=14)
    ax[(1, 0)].set_ylabel(metrics[i], fontsize=14)

    handles, labels = ax[(1, 2)].get_legend_handles_labels()
    ax[(0, 2)].legend(handles, labels, loc="lower right", ncol=2, frameon=False, fontsize=13)
    plt.savefig('Parameter_sensitivity_{}'.format(metrics[i]), dpi=1000)
    plt.close()
