import sys
sys.path.append("..")
from load_results.load_cifar import ACR,BASE
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
import math
from matplotlib import style
#style.use('ggplot')

plt.figure(figsize=(10, 5), dpi=70)
fig,ax=plt.subplots()
matplotlib.rcParams.update({'font.size': 20})

as_parameters=[(100000,0.1),(100000,0.05),(200000,0.05),(200000,0.1),(500000,0.1)]
list_acras=[]
list_acrbase=[]
xs=[]
k=1/50000*5195/60/(1/70919*40910/60)

sigma=0.5
for ss, col in [(100000, 'seagreen'), (200000, 'orange')]:
    acr_base,t= BASE(sigma, ss)
    t=np.round(t/60)
    plt.plot([20,120],[acr_base]*2,alpha=0.7, color=col, linewidth=3.0, linestyle='--')
    plt.annotate(f'IAS-{ss /10000:.0f}: time={t:.0f}', xy=(40, acr_base+0.002), xytext=(40, acr_base+0.002),
                 family='Times New Roman',  # 标注文本字体为Times New Roman
                 fontsize=15,  # 文本大小为18
                 fontweight='bold',  # 文本为粗体
                 color=col, ha='center')

for i,(ss,al) in enumerate(as_parameters):
 
    x_ = -20*k
    y_ = 0.003
    if i==0:
        x_=20*k
        y_=0.004
    if i==3:
        y_=0.008
    if i==4:
        y_=0.002
    ax.set_xlim((20, 120))
    ax.set_ylim((0.62,0.688))

    acr_as, acr_base, avgss,t=ACR(sigma,ss//200,al)
    t = np.round(t / 60)
    list_acras.append(acr_as)
    list_acrbase.append(acr_base)
    xs.append(t)
    plt.annotate(f'ISS-{ss/10000:.0f}-{al}', xy=(t,acr_as), xytext=(t+x_, acr_as+y_),
                 family='Times New Roman',  # 标注文本字体为Times New Roman
                 fontsize=15,  # 文本大小为18
                 fontweight='bold',  # 文本为粗体
                 color='royalblue', ha='center')
    x__=50*k
    y__=-0.005
    if i== len(as_parameters)-1:
        x__=-30*k
        y__=-0.008
    if i== 0:
        x__=60*k
        y__=0
    plt.annotate(f'IAS-{avgss/10000:.1f}', xy=(t,acr_base), xytext=(t+x__, acr_base+y__),
                 family='Times New Roman',  # 标注文本字体为Times New Roman
                 fontsize=15,  # 文本大小为18
                 fontweight='bold',  # 文本为粗体
                 color='indianred', ha='center')

xs.sort()
list_acrbase.sort()
list_acras.sort()
plt.plot(xs,list_acrbase, '--', color='indianred',
        marker = 'o',
        markersize=8, linewidth=2,
        markerfacecolor='indianred',
        markeredgecolor='indianred',
        markeredgewidth=2,label='Cohen et al 2019: IAS')

plt.plot(xs,list_acras, '--', color='royalblue',
        marker = '^',
        markersize=8, linewidth=2,
        markerfacecolor='royalblue',
        markeredgecolor='grey',
        markeredgewidth=2,label='Our method: ISS')

# ax.set_xticks(np.linspace(10000,100000,5))
ax.set_xlabel('Time (min)')
ax.set_ylabel('Average certified radius')
plt.tick_params(labelsize=13)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(14)
# if sigma==1.2:
#     ax.set_ylim((2.89,1.01))
# if sigma==2.5:
#     ax.set_xlim((100,1000))
#     ax.set_ylim((2.74, 2.84))
plt.grid(True, linestyle = "-", color = "grey", linewidth = 0.5)
plt.legend(loc='lower right',fontsize=15)
plt.savefig(f'cifar_{sigma}.png')
plt.close()

