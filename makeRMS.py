import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import cycle

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

## prevent color repeating 
colormap = plt.cm.nipy_spectral
number_of_plots = 12
colors = colormap(np.linspace(0, 1, number_of_plots))



othervars = ['fj_mass', 'fj_sdmass', 'fj_corrsdmass', 'fj_sdmass_fromsubjets', 'pfParticleNetMassRegressionJetTags_mass']
rmslog_fig, rmslog_ax = plt.subplots()
mmslog_fig, mmslog_ax = plt.subplots()
rmslin_fig, rmslin_ax = plt.subplots()
sentlin_fig ,sentlin_ax = plt.subplots()

## prevent color repeating
rmslog_ax.set_prop_cycle( 'color', colors)
mmslog_ax.set_prop_cycle( 'color', colors)
rmslin_ax.set_prop_cycle( 'color', colors)
sentlin_ax.set_prop_cycle('color', colors)

for f in os.listdir('csv'):
    if ('trend' not in f) or ('~' in f):
        continue
    df = pd.read_csv(f'csv/{f}')
    massranges = df['mass_range']
    xvals = range(len(massranges))
    name = f.replace('.csv', '').replace('trend_', '')
    linestyle = next(linecycler)
    rmslog_ax.plot(df['RMS_logRatio'], linestyle=linestyle,label=name)
    mmslog_ax.plot(df['MMS_logRatio'], linestyle=linestyle,label=name)
    rmslin_ax.plot(df['RMS_ratio'], linestyle=linestyle,label=name)
    sentlin_ax.plot(df['sensitivity2'], linestyle=linestyle,label=name)
    # mass_range,MMS_logRatio,RMS_logRatio,RMS_ratio,sensitivity2

rmslog_ax.legend()
mmslog_ax.legend()
rmslin_ax.legend()
sentlin_ax.legend()

rmslog_ax.set_title('RMS logRatio')
mmslog_ax.set_title('MMS logRatio')
rmslin_ax.set_title('RMS ratio')
sentlin_ax.set_title('Sensitivity^2')

arr = [0,1,2,3,4,5]
xticks = [x.replace('(','').replace(')','').split(', ') for x in massranges]
xticks = [f'{x[0]}-{x[1]}' for x in xticks]
xticks[-1] = '180<'
print(xticks)



rmslog_ax.set_xticks(arr,  xticks)
mmslog_ax.set_xticks(arr,  xticks)
rmslin_ax.set_xticks(arr,  xticks)
sentlin_ax.set_xticks(arr, xticks)

for axis in [rmslog_ax, mmslog_ax, rmslin_ax, sentlin_ax]:
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

rmslog_fig.savefig('plots/trend_rmslog.png')
mmslog_fig.savefig('plots/trend_mmslog.png')
rmslin_fig.savefig('plots/trend_rmslin.png')
sentlin_fig.savefig('plots/trend_sensitivity.png')

    
import sys
sys.exit()

df = pd.read_csv('RMS_MMS_masspoints.csv')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots() 
for key in df:
    print(key)
    if 'rms2' in key:
        ax3.plot(df['masspoints'], df[key], label=key)
    elif 'sensitivity' in key:
        ax4.plot(df['masspoints'], df[key], label=key)
    elif 'rms' in key:
        ax1.plot(df['masspoints'], df[key], label=key)
    elif 'mms' in key:
        ax2.plot(df['masspoints'], df[key], label=key)
    

ax1.set_title(f'a1 RMS vs mass')
ax1.set_xlabel('Mass (GeV)')
ax1.set_ylabel('RMS')
ax1.legend()
fig1.savefig(f'plots/trend_RMS_masspoints.png', bbox_inches='tight')

ax2.set_title(f'a1 MMS vs mass')
ax2.set_xlabel('Mass (GeV)')
ax2.set_ylabel('MMS')
ax2.legend()
fig2.savefig(f'plots/trend_MMS_masspoints.png', bbox_inches='tight')

ax3.set_title(f'a1 RMS (no log2) vs mass')
ax3.set_xlabel('Mass (GeV)')
ax3.set_ylabel('RMS (no log2)')
ax3.legend()
fig3.savefig(f'plots/trend_RMS_nolog2_masspoints.png', bbox_inches='tight')

ax4.set_title(f'a1 sensitivity vs mass')
ax4.set_xlabel('Mass (GeV)')
ax4.set_ylabel('sensitivity')
ax4.legend()
fig4.savefig(f'plots/trend_sensitivity_masspoints.png', bbox_inches='tight')
    





