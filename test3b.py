## need to be in anaconda (base) and not (weaver) to run this. don't know why it doesn't work in weaver
from pathlib import Path
import uproot 
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
import pandas as pd
import matplotlib as mpl
import os
import sys
import itertools 
from matplotlib.colors import LogNorm
from benchmark import pairwise 
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('epoch')
#args = parser.parse_args()

#predict_wide_H_calc_mass_regr_loss1.root
mods = ['mass', ]#'logMass']#['mass', 'logMass', 'massOverfj_mass']
loss_modes = [0 ]#,3]#[0,1,3]
#massrange = [0,80,95,110,135,180,99999]
#loss_modes.remove(2)
vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no' , 'fj_gen_H_aa_bbbb_num_b_AK8', 'fj_nbHadrons']
#epochs = [0, 1, 2, 4, 6, 8, 12, 16] # [10,15,20,25,30,35,40]
from itertools import cycle

lines = ["-","--","-.",":"]
linecycler = cycle(lines)


flist = [
 ## mass 3b
    'predict/34b/predict_h125_a1_M10.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M10.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M11.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M11.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M12.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M12_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M13.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M13.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M14.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M15_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M16.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M17.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M18.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M20_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M21.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M23.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M25_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M27.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M30_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M32.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M35_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M37.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M40_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M42.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M45_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M47.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M50_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M52.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M55_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M57.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M60_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M62.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M8.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M9.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M9.5_mass_3b_regr.root',
]


## strip after M-.....root, strip.root,
massranges = [x[x.find('_M')+2:x.find('_mass')].split('.root')[0] for x in flist]
#massranges = [float(x) for x in massranges]
#massranges.sort()
#massranges = massranges[::2]


figsize = (9,6.7)
rmslintestoverlays = plt.subplots(figsize=figsize)
senlintestoverlays = plt.subplots(figsize=figsize)
mmslogtestoverlays = plt.subplots(figsize=figsize)
rmslogtestoverlays = plt.subplots(figsize=figsize)
rmslindiffoverlays = plt.subplots(figsize=figsize) 
senlindiffoverlays = plt.subplots(figsize=figsize) 
rmslogdiffoverlays = plt.subplots(figsize=figsize) 
mmslogdiffoverlays = plt.subplots(figsize=figsize) 

## test overtrain for each mod, multiple epochs, one mass range
for mod in mods:
    for lm in loss_modes:
        rmslintestlist = []
        senlintestlist = []
        mmslogtestlist = []
        rmslogtestlist = []
        
        rmslintrainlist = []
        senlintrainlist = []
        mmslogtrainlist = []
        rmslogtrainlist = []
        for massrange in massranges:            
            fn = f'predict/34b/predict_h125_a1_M{massrange}_mass_3b_regr.root'
            df = pd.DataFrame()
            f = uproot.open(fn)
            g = f['Events']
            
            for v in vs:
                df[v] = np.array(g[v].array())
            if mod == 'logMass':
                df['output'] = np.exp(df['output'])
                df['target_mass'] = np.exp(df['target_mass'])
            dftest = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1) & (df['event_no']%2==1)]
            dftest['ratio'] = np.divide(dftest['output'] , dftest['target_mass'])
            dftest['log2ratio'] = np.clip(np.log2(dftest['ratio']), a_min=-2, a_max=2)


            
            ## calculate rms and stuff
            rmslintestlist.append(np.std(dftest['ratio']))
            mmslogtestlist.append(np.mean(dftest['log2ratio']))
            rmslogtestlist.append(np.std(dftest['log2ratio']))

            s_hist, s_edge = np.histogram(dftest['ratio'], bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            senlintestlist.append(sensitivity)

        mmslog = plt.subplots(figsize=figsize)
        rmslin = plt.subplots(figsize=figsize)
        rmslog = plt.subplots(figsize=figsize)
        senplt = plt.subplots(figsize=figsize)

            
        for pltobj, testval, savename, yrange in zip([mmslog, rmslin, rmslog, senplt],
                                                               [mmslogtestlist, rmslintestlist, rmslogtestlist, senlintestlist],                        
                                                               ['mms log ratio', 'rms ratio', 'rms log ratio', 'sensitivity'],
                                                               [(None, None), (None,None), (None, None), (None, None)]):

            print(testval)
            #xticks = [str(x) for x in massranges]
            #arr = list(range(len(xticks)))
            floatmassranges= [float(x) for x in massranges]
            pltobj[1].plot(floatmassranges, testval, 'bo')
            #pltobj[1].set_xticks(arr, xticks,)
            pltobj[1].set_xlabel('mass points')
            plt.grid()
            #pltobj[1].legend()
            pltobj[1].set_title(f'{savename} performance of mass target 3b agaisnt masspoint')
            #pltobj[1].set_ylim([yrange[0], yrange[1]])
            #Path(f"plots/testepoch/").mkdir(parents=True, exist_ok=True)
            pltobj[0].savefig(f'plots/34b/{savename}_3b_score_vs_mass.png', bbox_inches='tight')
            plt.close(pltobj[0])


## create distribution of predicted mass points
## how to do this for the 2018...may have to split it into 3 regions
masseslist = [[11.0,11.5,12,12.5], [13.5,14.0,15,16.0,17.0], [18.5, 20, 21.5, 23.0, 25], [27.5, 30, 32.5, 35, 37.5], [40, 42.5,45,47.5, 50], [52.5, 55, 57.5, 60, 62.5]]#[[12,20,30,40,50,60], [15,25,35,45,55], [8.5,9,9.5,10,10.5,11, 11.5]]
rangelists = [[10,60], [10,60], [10, 60], [15,60], [23,65], [20,70]]
binsizelist = [.5, .5, .5, .5, 1,1]#[0.1, 0.1, 0.2, 0.5, 1, 1]


for masses, binsize, binrange in zip(masseslist, binsizelist, rangelists):
    fig, ax = plt.subplots(figsize=figsize)
    for mass in masses:
        fn = f'predict/34b/predict_h125_a1_M{mass}_mass_3b_regr.root' 
        f = uproot.open(fn)
        g = f['Events']
        df = pd.DataFrame()
        for v in vs:
            df[v] = np.array(g[v].array())
        df = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['fj_gen_H_aa_bbbb_num_b_AK8'] == 3) & (df['fj_nbHadrons'] == 3)]
        numbins = round((binrange[1]-binrange[0])/(binsize))
        #binbounderies = [] 
        #print(df['output'])
        #dfcut = df# [np.isclose(df['target_mass'],mass, rtol=1e-03, atol=1e-05)]
        #ax.hist(dfcut['output'], bins=numbins, range=(masses[0]*.8, masses[-1]*1.2), label=f'M-{mass}', histtype='step')
        #clippedoutput = dfcut['output']#np.clip(dfcut['output'], a_min=binrange[0], a_max=binrange[1])
        ax.hist(df['output'], bins=numbins, range=(binrange[0], binrange[1]), label=f'M-{mass}', histtype='step')
    ax.set_title(f'm(a) 3b regressor trained on mH-125  range {masses[0]}-{masses[-1]}')
    ax.set_xlabel('Predicted m(a) [GeV]')
    ax.set_ylabel('Events')
    ax.legend()
    fig.savefig(f'plots/34b/distributions_mass_3b_range{masses[0]}-{masses[-1]}.png', bbox_inches='tight')
    plt.close(fig)
            


sys.exit()

