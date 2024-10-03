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
import matplotlib.colors as colors
from textwrap import wrap


# https://baylorhep.slack.com/archives/DPXPFEXC0/p1718919825524739
# https://baylorhep.slack.com/archives/C013B0LRAEA/p1718211083165349
#predict_wide_H_calc_mass_regr_loss1.root
mods = ['mass', 'logMass']
loss_modes = [0,3]
massranges = [0,13,17,25,37,40,50,999]
vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no', 'fj_gen_H_aa_bbbb_mass_a1', 'fj_gen_H_aa_bbbb_mass_a2']

from itertools import cycle

lines = ["-","--","-.",":"]
linecycler = cycle(lines)


flist_m125 = ['/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/hadd/AK8_HToAATo4B_GluGluH_01J_Pt150_M-All.root']

## wide H samples
flist_avga1a2 = [
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mAA_regr_E15_Sig_mH-125.root'
]
flist_a1 = [
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_dR_a1_E10_Sig_mH-125.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_mass_a1_E10_Sig_mH-125.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_dR_a1_E10_Sig_mH-125.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_mass_a1_E10_Sig_mH-125.root'
]
flist_a2 = [
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_dR_a2_E10_Sig_mH-125.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_mass_a2_E10_Sig_mH-125.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_dR_a2_E10_Sig_mH-125.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_mass_a2_E10_Sig_mH-125.root'
]


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


## this is the overlay between centrally trained a1 and wideH average of a1 a2
## Overlay with either your mass_loss0 or mass_loss3 regression which was trained on the mass(H) = 125 samples.

fn = f'predict/testepoch/predict_a1_calc_mass_regr_loss3_epoch20.root'
df = pd.DataFrame()
f = uproot.open(fn)
g = f['Events']

rmslin_central_list = []
senlin_central_list = []
mmslog_central_list = []
rmslog_central_list = []

rmslin_wide_list = []
senlin_wide_list = []
mmslog_wide_list = []
rmslog_wide_list = []


for v in vs:
    df[v] = np.array(g[v].array())
dfcentral = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1) & (df['event_no']%2==1)]


f = uproot.open(flist_avga1a2[0])
g = f['Events']
df = pd.DataFrame()
for v in vs:
    df[v] = np.array(g[v].array())
dfwide = df

# dftest = dftest[np.isclose(dftest['target_mass'], massrange, rtol=1e-03, atol=1e-05)]    
dfcentral['ratio'] = np.divide(dfcentral['output'] , dfcentral['target_mass'])
dfcentral['log2ratio'] = np.clip(np.log2(dfcentral['ratio']), a_min=-2, a_max=2)

dfwide['ratio'] = np.divide(dfwide['output'], dfwide['target_mass'])
dfwide['log2ratio'] =  np.clip(np.log2(dfwide['ratio']), a_min=-2, a_max=2)


for lowermass, uppermass in zip(massranges[:-1], massranges[1:]):
    ## get only the events in the mass range
    dfcentralclipped = dfcentral[(dfcentral['target_mass'] >= lowermass) & (dfcentral['target_mass'] < uppermass)]
    dfwideclipped = dfwide[(dfwide['target_mass'] >= lowermass) & (dfwide['target_mass'] < uppermass)]

    ## calculate rms and stuff
    rmslin_central_list.append(np.std(dfcentralclipped['ratio']))
    mmslog_central_list.append(np.mean(dfcentralclipped['log2ratio']))
    rmslog_central_list.append(np.std(dfcentralclipped['log2ratio']))

    s_hist, s_edge = np.histogram(dfcentralclipped['ratio'], bins=101, range=(0,2.02))
    binwidth = s_edge[1]-s_edge[0]
    sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
    senlin_central_list.append(sensitivity)
    
    rmslin_wide_list.append(np.std(dfwideclipped['ratio']))
    mmslog_wide_list.append(np.mean(dfwideclipped['log2ratio']))
    rmslog_wide_list.append(np.std(dfwideclipped['log2ratio']))

    s_hist, s_edge = np.histogram(dfwideclipped['ratio'], bins=101, range=(0,2.02))
    binwidth = s_edge[1]-s_edge[0]
    sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))        
    senlin_wide_list.append(sensitivity)
        
    
mmslog = plt.subplots(figsize=figsize)
rmslin = plt.subplots(figsize=figsize)
rmslog = plt.subplots(figsize=figsize)
senplt = plt.subplots(figsize=figsize)

        
for pltobj, centralval, wideval, savename, yrange in zip([mmslog, rmslin, rmslog, senplt],
                                                       #[mmslogtestlist, rmslintestlist, rmslogtestlist, senlintestlist],
                                                       #[mmslogtrainlist, rmslintrainlist, rmslogtrainlist, senlintrainlist],
                                                       [mmslog_central_list, rmslin_central_list, rmslog_central_list, senlin_central_list],
                                                       [mmslog_wide_list, rmslin_wide_list, rmslog_wide_list, senlin_wide_list],
                                                       ['mms log ratio', 'rms ratio', 'rms log ratio', 'sensitivity'],
                                                       [(-.5, .4), (0, .4), (0,.5), (0,10)]):

    xticks = [f'{x}-{y}' for x,y in zip(massranges[:-2], massranges[1:])]
    arr = list(range(len(xticks)))
    pltobj[1].plot(centralval, label='central')
    pltobj[1].plot(wideval, label='wide')
    pltobj[1].set_xticks(arr, xticks,)
    pltobj[1].set_xlabel('mass points')
    plt.grid()
    pltobj[1].legend()
    pltobj[1].set_title(f'{savename} wideH regression to avg of a1+a2 and central a1, mass loss3')
    pltobj[1].set_ylim([yrange[0], yrange[1]])
    pltobj[0].savefig(f'plots/wideH_aMass_overlay/{savename}_avg_a1a2_mass_loss3.png', bbox_inches='tight')
    plt.close(pltobj[0])


plt.close('all')

# Regressions to mass a1 then a2
#Overlay 4 different trainings (bb_mass, bb_dR, bbbar_mass, bbbar_dR)
#Make separate sets of plots for "output vs. target_mass" and "output vs. fj_gen_H_aa_bbbb_mass_a1" (then a2)
for flist_a, apart, clipvals in zip ([flist_a1, flist_a2], ['a1', 'a2'], [(0,150), (0,80)]):    
        
    fs = [uproot.open(x) for x in flist_a]
    gs = [x['Events'] for x in fs]
    dfswide = [pd.DataFrame() for x in range(4)]
    for g,df in zip(gs,dfswide):
        for v in vs:
            tmpdf =  np.array(g[v].array())
            df[v] = np.array(g[v].array())
            
    
    wrapwidth = 50
    
    for df, fname in zip(dfswide, flist_a) :            
        figdist, axdist = plt.subplots()
        histkwarg = {'histtype':'step', 'range':(0,150), 'bins':50}
        axdist.hist(df['output'], label='output', **histkwarg)
        target = df['target_mass']
        axdist.hist(df['target_mass'], label='target mass',**histkwarg)
        name = fname.split('/')[-1].split('.')[-2]
        axdist.set_title('\n'.join(wrap(f'distribution {name}', wrapwidth)))
        axdist.legend()
        figdist.savefig(f'plots/wideH_aMass_overlay/{apart}_dist_overlay_{name}.png', bbox_inches='tight')

        ## 2d plots 
        target_output_plt = plt.subplots()
        hist2dkwarg = {'bins' :50, }
        clipvals = (min(df['target_mass']), max(df['target_mass']))
        colorbar = target_output_plt[1].hist2d(df['target_mass'], np.clip(df['output'], a_min=clipvals[0], a_max=clipvals[1]), norm = colors.LogNorm(), **hist2dkwarg )
        target_output_plt[1].set_xlabel('target')
        target_output_plt[1].set_ylabel('output')
        target_output_plt[1].set_title('\n'.join(wrap(f'{apart}_output_vs_target_{name}', wrapwidth)))
        target_output_plt[0].colorbar(colorbar[3], ax=target_output_plt[1])
        target_output_plt[0].savefig(f'plots/wideH_aMass_overlay/{apart}_output_vs_target_{name}.png', bbox_inches='tight')
    
    
        target_massa1_plt = plt.subplots()
    
        colorbar = target_massa1_plt[1].hist2d(df[f'fj_gen_H_aa_bbbb_mass_{apart}'], np.clip(df['output'], a_min=clipvals[0], a_max=clipvals[1]), norm = colors.LogNorm(), **hist2dkwarg)
        target_massa1_plt[1].set_xlabel(f'fj_gen_H_aa_bbbb_mass_{apart}')
        target_massa1_plt[1].set_ylabel('output')
        target_massa1_plt[1].set_title('\n'.join(wrap(f'{apart}_output_vs_genMass_{apart}_{name}', wrapwidth)))
        target_massa1_plt[0].colorbar(colorbar[3], ax=target_massa1_plt[1])
        target_massa1_plt[0].savefig(f'plots/wideH_aMass_overlay/{apart}_output_vs_{apart}_mass_{name}.png', bbox_inches='tight')

plt.close('all')




## MMS, RMS(s), and sensitivity overlays for each of the following groups of plots, where each group contains 4 regressions:
##  * a1_output_vs_a1_mass
##  * a1_output_vs_target
##  * a2_output_vs_a2_mass
##      * Even though "a1_mass" and "a2_mass" are identical for the mH-125 samples, go ahead and make the comparison to "a2_mass" for clarity.
##  * a2_output_vs_target
for divisor, plotname in zip(['fj_gen_H_aa_bbbb_mass_a1','fj_gen_H_aa_bbbb_mass_a2', 'target_mass'],
                             ['output_vs_mass', 'output_vs_mass', 'output_vs_target']) :
    
    for fnlist in [flist_a1, flist_a2]:
        ## we only want to compare a1 to a1 and a2 to a2, so if it is a1, a2 loop, skip
        ## not elegant but what are we gonna do 
        if (('a2' in divisor) and ('a1_E10_Sig' in fnlist[0])) or (('a1' in divisor) and ('a2_E10_Sig' in fnlist[0])):
            continue
        mmslog = plt.subplots(figsize=figsize)
        rmslin = plt.subplots(figsize=figsize)
        rmslog = plt.subplots(figsize=figsize)
        senlin = plt.subplots(figsize=figsize)
        for fn in fnlist:
            df = pd.DataFrame()
            g = uproot.open(fn)['Events']
            for v in vs:
                df[v] = np.array(g[v].array())
            df['ratio'] = np.divide(df['output'], df[divisor])
            df['log2ratio'] = np.clip(np.log2(df['ratio']), a_min=-2, a_max=2)
        
            mmslog_list = []
            rmslin_list = []
            rmslog_list = []
            senlin_list = []
        
            for lowermass, uppermass in zip(massranges[:-1], massranges[1:]):
                dfclipped = df[(df['target_mass'] >= lowermass) & (df['target_mass'] < uppermass)]
                
                rmslin_list.append(np.std( dfclipped['ratio']))
                mmslog_list.append(np.mean(dfclipped['log2ratio']))
                rmslog_list.append(np.std( dfclipped['log2ratio']))
                
                s_hist, s_edge = np.histogram(dfclipped['ratio'], bins=101, range=(0,2.02))
                binwidth = s_edge[1]-s_edge[0]
                sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                senlin_list.append(sensitivity)
        
            for pltobj, plotvals in zip( [mmslog, rmslin, rmslog, senlin],
                                         [mmslog_list, rmslin_list, rmslog_list, senlin_list],):
                label = fn.split('/')[-1]
                label = label.replace('predict_mA_regr_'     ,'').replace('_E10_Sig_mH-125.root' ,'')
                pltobj[1].plot(plotvals, label=label)
         
        for pltobj, savename, ylim in zip([mmslog, rmslin, rmslog, senlin],
                                          ['mms log ratio', 'rms ratio', 'rms log ratio', 'sensitivity'],
                                          [(-1.3, 0.7), (0, .75), (0, .9), (0, 5)]) :
        
            pltobj[1].legend()
            if 'a1' in fn:
                particlename = 'a1'
            elif 'a2' in fn:
                particlename = 'a2'
            pltobj[1].set_title(f'{particlename} {plotname} {savename}')
            pltobj[1].set_ylim([ylim[0], ylim[1]])
            pltobj[0].savefig(f'plots/wideH_aMass_overlay/{savename.replace(" ", "_")}_{particlename}_{plotname}.png', bbox_inches='tight')

            plt.close(pltobj[0])



##  (2D + MMS/RMS/sens.) from the wH-70 samples
## honestly just copy this file to a new file and change the flist names and output plot names. we don't have time to redo all of them properly right now 
flist_wH_avga1a2 = [
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mAA_regr_E15_Sig_wH-70.root'
]

flist_wH_a1 = [
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_dR_a1_E10_Sig_wH-70.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_mass_a1_E10_Sig_wH-70.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_dR_a1_E10_Sig_wH-70.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_mass_a1_E10_Sig_wH-70.root',
]

flist_wH_a2 = [
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_dR_a2_E10_Sig_wH-70.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bbbar_mass_a2_E10_Sig_wH-70.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_dR_a2_E10_Sig_wH-70.root',
    '/cms/data/abrinke1/CMSSW/HiggsToAA/ParticleNet/weaver/output/predict_mA_regr_bb_mass_a2_E10_Sig_wH-70.root',
]




#2D Distribution for the wide and central trainings, both run on m125
flist_avga1a2

for fn, plotname in zip([flist_avga1a2[0], 'predict/testepoch/predict_a1_calc_mass_regr_loss3_epoch20.root'],
                        ['avga1a2', 'central']):
    g = uproot.open(fn)['Events']
    df = pd.DataFrame()
    for v in vs:
        df[v] = np.array(g[v].array())
    
    fig, ax = plt.subplots()
    colorbar = ax.hist2d(np.clip(df['target_mass'], a_min=0, a_max=150),
                         np.clip(df['output'], a_min=0, a_max=150),
                         norm = colors.LogNorm(), bins=50)
    ax.set_xlabel('target')
    ax.set_ylabel('output')
    ax.set_title(f'{plotname}_output_vs_target')
    fig.colorbar(colorbar[3], ax=ax)
    fig.savefig(f'plots/wideH_aMass_overlay/{plotname}_output_vs_target.png', bbox_inches='tight')
            
