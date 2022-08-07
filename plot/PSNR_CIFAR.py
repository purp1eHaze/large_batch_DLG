import matplotlib.pyplot as plt
import numpy as np

color_list = []

for i in np.arange(0, 1, 0.2):
    color_list.append(plt.cm.Set1(i))

for i in np.arange(0, 1, 0.2):
    color_list.append(plt.cm.Set1(i))

def plot_img(X, Y, label, xlabel, ylabel):
    # sns.set_theme(style="whitegrid")
    marker = ['p', '*', 'o', '^', 'd', 'h']
    linestyle = ['--', '-', '-', '-']
    
    
    # plt.rcParams['axes.facecolor']='snow'
    fig = plt.figure(figsize=(16, 12))
    

    # plt.style.use('ggplot')
    # plt.grid(color='white')
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(X, Y[0], color= "orange", label=label[0], linewidth=4, marker=marker[0], markersize=16,
                        markerfacecolor= "orange",  markeredgecolor='black', linestyle=linestyle[0])

    l2 = ax1.plot(X, Y[1], color= 'royalblue', label=label[1], linewidth=4, marker=marker[1], markersize=20,
                  markerfacecolor= 'royalblue',  markeredgecolor='black', linestyle=linestyle[0])

    l3 = ax1.plot(X, Y[2], color=color_list[2], label=label[2], linewidth=4, marker=marker[2], markersize=16,
                  markerfacecolor= color_list[2],  markeredgecolor='black', linestyle=linestyle[0])
   
    l4 = ax1.plot(X, Y[3], color="c", label=label[3], linewidth=4, marker=marker[3], markersize=16,
                  markerfacecolor= "c",  markeredgecolor='black', linestyle=linestyle[0])

    l5 = ax1.plot(X, Y[4], color="green", label=label[4], linewidth=4, marker=marker[4], markersize=16,
                  markerfacecolor= "green",  markeredgecolor='black', linestyle=linestyle[0])

    l6 = ax1.plot(X, Y[5], color=color_list[0], label=label[5], linewidth=4, marker=marker[5], markersize=16,
                  markerfacecolor= color_list[0],  markeredgecolor='black', linestyle=linestyle[1])

    lns = l1 + l2 + l3 + l4 +l5 + l6
    labs = [l.get_label() for l in lns]
    print(labs)
    ax1.grid(linewidth=1.2)
    ax1.legend(lns, labs, loc=0, fontsize=24)
    ax1.patch.set_facecolor("whitesmoke")
    ax1.patch.set_alpha(0.70)
    ax1.tick_params(axis='x', labelsize=42)
    ax1.tick_params(axis='y', labelsize=36)
    # ax1.set_ylim(0.65, 1.02)

    ax1.set_ylabel(ylabel, fontsize = 44)
    ax1.set_xlabel(xlabel, fontsize = 44)

    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    plt.xscale("log")
    X_list = np.array(X)
    plt.xticks(X_list, labels = X_list.astype(int))

    #plt.axvline(x=7, ls='--', c='black', lw=2)
    plt.show()


DLG = [ 16.77, 15.73, 15.12, 14.22, 13.64, 12.67, 12.14, 11.74]

SAPAG = [ 16.77, 10.73, 10.74,   9.64, 9.67, 9.22, 8.14,7.12 ]

geiping = [ 36.07, 23.86, 18.40, 17.45, 15.38, 14.28, 13.89, 13.75]
                        
BN = [20.24, 19.75, 18.93, 17.76, 16.64, 16.54, 15.98, 15.54]

GC = [20.34, 19.94, 18.93, 17.76, 16.94, 16.58, 16.47, 16.43]

RGIA = [60.96, 60.87, 59.45, 36.75, 31.77, 26.81, 18.48, 16.48]

plot_img([1, 2, 4, 8, 16, 32, 64, 128], [DLG,  SAPAG, geiping, BN, GC, RGIA], 
         [r'DLG', r'SAPAG', r'Cosine Similarity', r'BN Regularizer', r'GC Regularizer', r'RGIA'], 'Batch Size', 'PSNR / dB')

plt.savefig("test.png")









# def plot_img(X, Y, label, xlabel, ylabel):
#     # sns.set_theme(style="whitegrid")
#     marker = ['p', '*', '*', 'x']
#     linestyle = ['--', '-', '-', '-']

#     plt.grid(color='white')
#     fig = plt.figure(figsize=(13.5, 10.2))
#     ax1 = fig.add_subplot(111)
#     l1 = ax1.plot(X, Y[0], color=color_list[2], label=label[0], linewidth=5, marker=marker[2], markersize=16,
#                   linestyle=linestyle[0])
    
#     # l2 = ax2.plot(X, Y[1], color=color_list[2], label=label[1], linewidth=5, marker=marker[2], markersize=16,
#     #               linestyle=linestyle[1])
#     # l3 = ax2.plot(X, Y[2], color=color_list[2], label=label[2], linewidth=5, marker=marker[2], markersize=16,
#     #               linestyle=linestyle[2])
   
#     # l4 = ax2.plot(X, Y[3], color=color_list[3], label=label[3], linewidth=5, marker=marker[2], markersize=16,
#     #           linestyle=linestyle[3])

#     lns = l1 #+ l2 #+ l3 + l4
#     labs = [l.get_label() for l in lns]
#     print(labs)
#     ax1.grid()
#     ax1.legend(lns, labs, loc=4, fontsize=24)
#     ax1.patch.set_facecolor("white")
#     ax1.patch.set_alpha(0.20)
#     ax1.tick_params(axis='x', labelsize=42)
#     ax1.tick_params(axis='y', labelsize=36)

#     ax1.set_ylim(0.0, 1.02)

#     ax1.set_ylabel("Detection Rate ", fontsize = 30)
#     ax1.set_xlabel("Training Epoch", fontsize = 30)

#     plt.axvline(x=7, ls='--', c='black', lw=2)
#     plt.savefig("backdoor.pdf")
#     plt.show()

# ###pruning
# plt.figure(figsize=(12, 8))
# percent = [0, 1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # fid_gamma_p = [0.9066,  0.9065, 0.9063, 0.9062, 0.9058, 0.9044, 0.9026, 0.8934, 0.8755, 0.8566, 0.8366, 0.8264, 0.8066,
# #                0.7849,  0.1]

# wm_p_in = [ 0.0900, 0.250, 0.350, 0.4750,  0.5550,  0.5800, 0.6200, 0.6700, 0.7100,  0.7500, 0.8200, 0.8500, 0.8900, 0.9300, 0.9400 ]


# plot_img(percent[:-1], [   wm_p_in[:-1]],
#     [  r'$\eta_T$ of Trigger',], 'Training Epoch',
#          'Accuracy')

###finetuning


