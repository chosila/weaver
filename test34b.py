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

mods = ['mass','logMass']
loss_modes = [0, 3]
vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no', 'fj_gen_H_aa_bbbb_num_b_AK8', 'fj_nbHadrons']
from itertools import cycle

lines = ["-","--","-.",":"]
linecycler = cycle(lines)


flist = [
 ## mass 3b
    'predict/34b/predict_h125_a1_M10.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M23.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M10.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M27.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M11.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M32.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M11.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M37.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M12.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M42.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M13.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M47.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M13.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M52.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M14.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M57.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M16.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M62.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M17.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M8.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M18.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M9.0_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M21.5_mass_3b_regr.root',
    'predict/34b/predict_h125_a1_M9.5_mass_3b_regr.root',
]


## strip after M-.....root, strip.root,
massranges = [x[x.find('_M')+2:x.find('_mass')].split('.root')[0] for x in flist]
massranges = [float(x) for x in massranges if float(x) > 12]

massranges.sort()

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

for num_genb in [3,4]:
    for num_b in [3,4,34]:
        mmslog = plt.subplots(figsize=figsize)
        rmslin = plt.subplots(figsize=figsize)
        rmslog = plt.subplots(figsize=figsize)
        senplt = plt.subplots(figsize=figsize)
        for lm in loss_modes:
            for mod in mods:

                ## we only want  3b and 34b trainings for the 3gen
                ## and           34b and the 4b trainings for the 4gen 
                if (num_genb==3) and (num_b==4) : continue
                if (num_genb==4) and (num_b==3) : continue
                rmslinlist = []
                senlinlist = []
                mmsloglist = []
                rmsloglist = []
                
                for massrange in massranges:            
                    fn = f'predict/34b/predict_h125_a1_M{massrange}_{mod}_{num_b}b_lm{lm}_regr.root'
                    df = pd.DataFrame()
                    f = uproot.open(fn)
                    g = f['Events']

                    for v in vs: 
                        df[v] = np.array(g[v].array())
                    if mod == 'logMass':
                        df['output'] = np.exp(df['output'])
                        df['target_mass'] = np.exp(df['target_mass'])
                    dftest = df[(df['output'] > 0) & (df['target_mass'] > 0)]
                    if num_genb == 3:
                        df = df[(df['fj_gen_H_aa_bbbb_num_b_AK8']==3) & (df['fj_nbHadrons']==3)]
                    else:
                        df = df[(df['fj_gen_H_aa_bbbb_num_b_AK8']>=4) & (df['fj_nbHadrons']>=4)]

                    df['ratio'] = np.divide(df['output'] , df['target_mass'])
                    df['log2ratio'] = np.clip(np.log2(df['ratio']), a_min=-2, a_max=2)
        
        
                    ## calculate rms and stuff
                    rmslinlist.append(np.std( df['ratio']))
                    mmsloglist.append(np.mean(df['log2ratio']))
                    rmsloglist.append(np.std( df['log2ratio']))
        
                    s_hist, s_edge = np.histogram(df['ratio'], bins=101, range=(0,2.02))
                    binwidth = s_edge[1]-s_edge[0]
                    sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                    senlinlist.append(sensitivity)
                     
      
                for pltobj, testval, savename, yrange in zip([mmslog, rmslin, rmslog, senplt],
                                                         [mmsloglist, rmslinlist, rmsloglist, senlinlist],                        
                                                         ['mms log ratio', 'rms ratio', 'rms log ratio', 'sensitivity'],
                                                         [(-0.3, 1.3), (0,1), (0, 1), (None, None)]):

                    
                    floatmassranges= [float(x) for x in massranges]
                    pltobj[1].plot(floatmassranges, testval, label=f'{mod} loss{lm}')

            for pltobj, savename, yrange in zip([mmslog, rmslin, rmslog, senplt],
                                            ['mms log ratio', 'rms ratio', 'rms log ratio', 'sensitivity'],
                                            [(-0.3, 1.3), (0,1), (0, 1), (None, None)]):
                pltobj[1].set_xlabel('mass points')
                plt.grid()
                pltobj[1].legend()
                pltobj[1].set_title(f'{savename} vs mass  performance of {mod} target {num_genb}gen loss{lm} ')
                pltobj[0].savefig(f'plots/34b/score_vs_mass_{savename}_{num_b}_{num_genb}gen_.png', bbox_inches  ='tight')
                plt.close(pltobj[0])
        


## 4 gen 4 b 
#[11.0,11.5,12.0,12.5],
masseslist = [ [12.5, 13.5,14.0,15.0,16.0,17.0], [18.5, 20.0, 21.5, 23.0, 25.0], [27.5, 30.0, 32.5, 35.0, 37.5], [40.0, 42.5,45.0,47.5, 50.0], [52.5, 55.0, 57.5, 60.0, 62.5]]#[[12,20,30,40,50,60], [15,25,35,45,55], [8.5,9,9.5,10,10.5,11, 11.5]]
#[10,30],
rangelists = [ [10,30], [10, 45], [10,70], [10,70], [10,70]]
#0.5
binsizelist = [ 0.5, 0.5, 1, 1, 1]


import sys
sys.exit()

## create distribution of predicted mass points
for mod in mods: 
    for lm in loss_modes:
        for num_b in [3,4,34]: 
            masseslist = [[11.0,11.5,12.0,12.5], [13.5,14.0,15.0,16.0,17.0], [18.5, 20.0, 21.5, 23.0, 25.0], [27.5, 30.0, 32.5, 35.0, 37.5], [40.0, 42.5,45.0,47.5, 50.0], [52.5, 55.0, 57.5, 60.0, 62.5]]#[[12,20,30,40,50,60], [15,25,35,45,55], [8.5,9,9.5,10,10.5,11, 11.5]]
            #if num_b == 4: 
            rangelists = [[10,30], [10,30], [10, 45], [10,70], [10,70], [10,70]]# [[10,18], [10,25], [10, 35], [15,60], [23,65], [20,70]]
            binsizelist = [0.5, 0.5, 0.5, 1, 1, 1]
            # elif num_b == 3:
            #     rangelists = [[10,30], [10,30], [10, 45], [10,70], [10,70], [10,70]] # [[10,60], [10,60], [10, 60], [15,60], [23,65], [20,70]]
            #     binsizelist = [0.5, 0.5, 0.2, 1, 1, 1]
        
            for masses, binsize, binrange in zip(masseslist, binsizelist, rangelists):
                fig3b, ax3b = plt.subplots(figsize=figsize)
                fig4b, ax4b = plt.subplots(figsize=figsize)
                for mass in masses:
                    fn = f'predict/34b/predict_h125_a1_M{mass}_{mod}_{num_b}b_lm{lm}_regr.root' 
                    f = uproot.open(fn)
                    g = f['Events']
                    df = pd.DataFrame()
                    
                    for v in vs:
                        df[v] = np.array(g[v].array())
        
                    if mod == 'logMass':
                        df['output'] = np.exp(df['output'])
                        df['target_mass'] = np.exp(df['target_mass'])
                                
                    numbins = round((binrange[1]-binrange[0])/(binsize))
        
                    ## test on events with 3 gen b
                    if num_b !=4 : 
                        df = df[(df['output'] > 0) & (df['target_mass'] > 0)]
                        df['output'] = np.clip(df['output'], a_min=binrange[0], a_max=binrange[1])
                        df3b = df[(df['fj_gen_H_aa_bbbb_num_b_AK8']==3) & (df['fj_nbHadrons']==3)]
                        ax3b.hist(df3b['output'], bins=numbins, range=(binrange[0], binrange[1]), label=f'M-{mass}', histtype='step')
        
                    ## test on events with 4 gen b
                    # if num_b  == 4: 
                    #     df4b = df[df['label_H_aa_bbbb']==1]
                    # else:
                    df4b = df[(df['fj_gen_H_aa_bbbb_num_b_AK8']>=4) & (df['fj_nbHadrons']>=4)]
                    ax4b.hist(df4b['output'], bins=numbins, range=(binrange[0], binrange[1]), label=f'M-{mass}', histtype='step')
                    
                ax3b.set_title(f'{mod} {num_b}b tested on 3 gen b events range {masses[0]}-{masses[-1]} loss{lm}')
                ax3b.set_xlabel('Predicted m(a) [GeV]')
                ax3b.set_ylabel('Events')
                ax3b.legend()
                
                fig3b.savefig(f'plots/34b/distributions_{mod}_{num_b}b_3genb_range{masses[0]}-{masses[-1]}_loss{lm}.png', bbox_inches='tight')
        
                ax4b.set_title(f'{mod} {num_b}b tested on 4 gen b events range {masses[0]}-{masses[-1]} loss{lm}')
                ax4b.set_xlabel('Predicted m(a) [GeV]')
                ax4b.set_ylabel('Events')
                ax4b.legend()
                if num_b != 3 : 
                    fig4b.savefig(f'plots/34b/distributions_{mod}_{num_b}b_4genb_range{masses[0]}-{masses[-1]}_lm{lm}.png', bbox_inches='tight')
        
                plt.close(fig3b)
                plt.close(fig4b)
        




