import uproot 
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
import pandas as pd
import matplotlib as mpl
import os
import itertools 
from matplotlib.colors import LogNorm
from itertools import cycle

mp1 = [12,20,30,40,50,60]
mp2 = [15,25,35,45,55]

parts = ['H_calc', 'a1', 'a2']
mods = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT', 'ptOverMass']
mods.remove('1OverMass')
#parts.remove('a2')
dfvariables = ['output', 'target_mass', 'fj_pt', 'label_H_aa_bbbb', 'event_no']

def returnToMass(df, mod):
    if mod == 'mass':
        pass
    elif mod == 'logMass':
        df['target_mass'] = np.exp(df['target_mass'])
        df['output'] = np.exp(df['output'])
    elif mod == '1OverMass':
        df['target_mass'] = 1/df['target_mass']
        df['output'] = 1/df['output']
    elif mod == 'massOverPT':
        df['target_mass'] = df['target_mass']*df['fj_pt']
        df['output'] = df['output']*df['fj_pt']
    elif mod == 'logMassOverPT':
        df['target_mass'] = np.exp(df['target_mass'])*df['fj_pt']
        df['output'] = np.exp(df['output'])*df['fj_pt']
    elif mod == 'ptOverMass':
        df['target_mass'] = df['fj_pt']/df['target_mass']
        df['output'] = df['fj_pt']/df['output']
    else:
        print('something wrong return to mass')

    return df 

def makeResPlots(mpnts, iseven):
    res_fig, res_ax = plt.subplots()
    rms_logs = []
    mms_logs = []

    dist_fig, dist_ax = plt.subplots()

    for mpnt in mpnts:
        f = uproot.open(f'predict/predict_central_{part}_M{mpnt}_{mod}_regr.root')
        g = f['Events']
        df = pd.DataFrame()
        for dfvar in dfvariables:
            df[dfvar] = g[dfvar].array()
        df = df[df['label_H_aa_bbbb']==1]
        df = returnToMass(df, mod)
        df = df[(df['target_mass']>0) & (df['output']>0) & (df['event_no']%2==1)]
        
        ratio = df['output']/df['target_mass']
        
        hist_kwargs = {'bins':50, 'histtype':'step', 'label':f'M-{mpnt}', 'range':(-2,2)}
        res_ax.hist(np.clip(np.log2(ratio), a_min=-2, a_max=2), **hist_kwargs)
        distrange = (0,400) if part=='H_calc' else (0,70)
        dist_ax.hist(np.clip(df['output'], a_min=distrange[0], a_max=distrange[1]), bins=100, range=(distrange), histtype='step', label=f'M-{mpnt}')

        rms = np.std(np.log2(ratio))
        mms = np.mean(np.log2(ratio))
        rms_logs.append(f'{rms:.4E}')
        mms_logs.append(f'{mms:.4E}')

    res_ax.table(
        colLabels=[f'M-{mp}' for mp in mpnts],
        rowLabels=['MMS', 'RMS'],
        cellText=[mms_logs, rms_logs],
        bbox=[0.1, -0.3, 0.9, 0.2]
    )
    evenodd = 'even' if iseven else 'odd'
    res_ax.legend()
    res_ax.set_title(f'resolution {part} {mod} {evenodd}')
    res_fig.savefig(f'plots/central/resolution_{part}_{mod}_{evenodd}.png', bbox_inches='tight')


    dist_ax.legend()
    dist_ax.set_title(f'distribution {part} {mod} {evenodd}')
    dist_fig.savefig(f'plots/central/distribution_{part}_{mod}_{evenodd}.png', bbox_inches='tight')

    plt.close('all')
    return rms_logs, mms_logs

lines = ["-","--","-.",":"]
linecycler = cycle(lines)
#predict_a1_M12_1OverMass_regr.root
for part in parts:
    rms_fig, rms_ax = plt.subplots()
    mms_fig, mms_ax = plt.subplots()
    trend_fig, trend_ax = plt.subplots()
    for mod in mods:
        evenrms, evenmms = makeResPlots(mp1, True)
        oddrms, oddmms = makeResPlots(mp2, False)
        rmsList = []
        mmsList = []
        for i in range(5):
            rmsList.append(float(evenrms[i]))
            rmsList.append(float(oddrms[i]))
            mmsList.append(float(evenmms[i]))
            mmsList.append(float(oddmms[i]))
        rmsList.append(float(evenrms[-1]))
        mmsList.append(float(evenmms[-1]))
        masspoints = sorted(mp1+mp2)
        linestyle = next(linecycler)
        rms_ax.plot( rmsList, linestyle=linestyle, label=mod)
        mms_ax.plot( mmsList, linestyle=linestyle, label=mod)
        rms_ax.set_xticks(list(range(11)), masspoints)
        mms_ax.set_xticks(list(range(11)), masspoints)
        print(mmsList)
        

    rms_ax.legend()
    rms_ax.set_title(f'RMS trend {part} centrally produced ggH')
    rms_fig.savefig(f'plots/central/trend_{part}_rmslog.png', bbox_inches='tight')
    mms_ax.legend()
    mms_ax.set_title(f'MMS trend {part} centrally produced ggH')
    mms_fig.savefig(f'plots/central/trend_{part}_mmslog.png', bbox_inches='tight')
