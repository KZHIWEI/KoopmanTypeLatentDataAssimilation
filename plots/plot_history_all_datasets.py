import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _plot_paths import RESULTS, FIGURES, DEFAULT_FONT

sources={k:json.loads((RESULTS / 'observation_history'/f'{k}.json').read_text(encoding='utf-8')) for k in ['dmd','tensorvar']}
datasets=['lorenz96','cylinder','shallow_water','era5']
names=['Lorenz–96','Cylinder','Shallow water','ERA5–Coarse']
ratios=[1,5,15,30,50]
colors={1:'#666666',5:'#78ADD2',15:'#4094C3',30:'#1769AA',50:'#083775'}
markers={1:'D',5:'o',15:'s',30:'^',50:'v'}
plt.rcParams.update({'font.family':DEFAULT_FONT,'font.size':18,'axes.titlesize':23,'axes.titleweight':'bold','axes.labelsize':20,'axes.labelweight':'bold','xtick.labelsize':18,'ytick.labelsize':18,'axes.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(20,9),sharey=True)
for row,model in enumerate(['dmd','tensorvar']):
 for col,ds in enumerate(datasets):
  ax=axes[row,col]
  for ratio in ratios:
   records=[v for v in sources[model].values() if isinstance(v,dict) and v.get('dataset')==ds and v.get('density')==ratio]
   if not records: continue
   points=[]
   for v in records:

    n=v['n_obs_frames']
    points.append((n,v['rrmse'],'rrmse',v['m']))
   points.sort()
   x=[p[0] for p in points]; y=[p[1] for p in points]
   assert len(x)==len(set(x)) and np.isfinite(y).all()
   ax.plot(x,y,linestyle='-',marker=markers[ratio],color=colors[ratio],lw=2.8,ms=7,markeredgecolor='white',markeredgewidth=.6)
  ticks=[1,5,10,20] if ds=='lorenz96' else ([1,3,5] if ds in ['cylinder','shallow_water'] else [1,5,10])
  ax.set_xticks(ticks);ax.set_ylim(0,1.0);ax.set_yticks(np.arange(0,1.01,.2))
  ax.tick_params(labelleft=True,width=1.0,length=4)
  if row==0: ax.set_title(names[col],pad=10)
  ax.grid(axis='y',color='#D5D5D5',alpha=.7,lw=.7);ax.set_axisbelow(True)
  ax.spines[['top','right']].set_visible(False)
  if row==1: ax.set_xlabel('Number of observed frames',labelpad=7)
  if col==0: ax.set_ylabel('RRMSE',labelpad=8)
fig.legend(handles=[Line2D([0],[0],color=colors[r],marker=markers[r],lw=2.8,ms=7,markeredgecolor='white',markeredgewidth=.6,label=f'obs {r}%') for r in ratios],loc='upper center',bbox_to_anchor=(.5,.995),ncol=5,frameon=False,prop={'weight':'bold','size':21},handlelength=2.2,columnspacing=1.5)
fig.subplots_adjust(left=.10,right=.99,top=.85,bottom=.12,wspace=.23,hspace=.26)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for row,label in enumerate(['DMD','Tensor-Var']):
 pos=axes[row,0].get_position()
 box=axes[row,0].yaxis.label.get_window_extent(renderer)
 title=fig.text(0,(pos.y0+pos.y1)/2,label,rotation=90,ha='center',va='center',fontsize=23,fontweight='bold')
 fig.canvas.draw()
 title_width=title.get_window_extent(renderer).width
 x_pixels=box.x0-7*fig.dpi/72-title_width/2
 title.set_x(x_pixels/fig.bbox.width)
FIGURES.mkdir(parents=True,exist_ok=True)
fig.savefig(FIGURES/'history_all_datasets.png',dpi=300,bbox_inches='tight',facecolor='white')
print(FIGURES/'history_all_datasets.png')
