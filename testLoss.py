## need to be in anaconda (base) and not (weaver) to run this. don't know why it doesn't work in weaver

import uproot 
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
import pandas as pd
import matplotlib as mpl
import os
import itertools 
from matplotlib.colors import LogNorm
from benchmark import pairwise 

#predict_wide_H_calc_mass_regr_loss1.root
mods = ['mass', 'logMass', 'massOverfj_mass']
loss_modes = [0,1,3]
massrange = [0,80,95,110,135,180,99999]
#loss_modes.remove(2)
vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass' ]

rmslin_dict = {}
senlin_dict = {}
rmslog_dict = {}
mmslog_dict = {}
mmslin_dict = {}

for mod in mods:
    dis_fig, dis_ax = plt.subplots()
    res_fig, res_ax = plt.subplots()
    sen_fig, sen_ax = plt.subplots()
    rms_log = []
    mms_log = []
    rms_lin = []
    sen_lin = []
    for lm in loss_modes:
        fn = f'predict/testLoss/predict_wide_H_calc_{mod}_regr_loss{lm}.root'
        df = pd.DataFrame()
        f = uproot.open(fn)
        g = f['Events']
        for v in vs:
            df[v] = g[v].array() 
        if mod == 'mass':
            pass
        if mod == 'logMass':
            df['output'] = np.exp(df['output'])
            df['target_mass'] = np.exp(df['target_mass'])
        if mod == 'massOverfj_mass':
            df['output'] = df['output']*df['fj_mass']
            df['target_mass'] = df['target_mass']*df['fj_mass']

        df = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1)]

        # print(f'mod: {mod}, lm: {lm}')
        # print(df.shape)
                    
        ## distribution
        histkwarg = {'bins':100, 'histtype':'step', 'label':f'loss mode {lm}', }
        dis_ax.hist(df['output'], range=(0,400), **histkwarg)
        
        ## resolution
        ratio = np.divide(df['output'] , df['target_mass'])
        log2ratio = np.clip(np.log2(ratio), a_min=-2, a_max=2)
        ratio = np.clip(ratio, a_min=0, a_max=2)
        res_ax.hist(log2ratio, range=(-2,2),**histkwarg)
        ## sensitivity
        s_hist, s_edge = np.histogram(ratio, bins=101, range=(0,2.02))
        binwidth = s_edge[1]-s_edge[0]
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        sen_ax.step( s_edge[:-1] , s_hist, label='loss mode {lm}')

        rms_log.append(round(np.std(log2ratio),4))
        mms_log.append(round(np.mean(log2ratio),4))
        rms_lin.append(round(np.std(ratio),4))
        sen_lin.append(round(sensitivity,4))


        rmslin_dict[f'{mod}_loss{lm}'] = [] 
        senlin_dict[f'{mod}_loss{lm}'] = [] 
        rmslog_dict[f'{mod}_loss{lm}'] = [] 
        mmslog_dict[f'{mod}_loss{lm}'] = [] 
        mmslin_dict[f'{mod}_loss{lm}'] = [] 
        
        ## calculate scores for various mass ranges
        for lower, upper in pairwise(massrange):
            cutdf = df[(df['target_mass'] > lower) & (df['target_mass'] <= upper)]
            ratio = np.divide(cutdf['output'], cutdf['target_mass'])
            log2ratio = np.clip(np.log2(ratio), a_min=-2, a_max=2)
            ratio = np.clip(ratio, a_min=0, a_max=2)
            s_hist, s_edge = np.histogram(ratio, bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        
            if lm == 0:
                if mod == 'logMass':
                    print('rms lin : ' , np.std(ratio))
                
            
            rmslin_dict[f'{mod}_loss{lm}'].append(round(np.std(ratio),4))
            senlin_dict[f'{mod}_loss{lm}'].append(round(sensitivity,4))  
            rmslog_dict[f'{mod}_loss{lm}'].append(round(np.std(log2ratio),4))  
            mmslog_dict[f'{mod}_loss{lm}'].append(round(np.mean(log2ratio),4))
            mmslin_dict[f'{mod}_loss{lm}'].append(round(np.mean(ratio),4))  
            
            
    colLabels = [f'loss mode: {lm}' for lm in [0,1,3]]
    bbox = [0.1, -0.3, 0.9, 0.2]
    res_ax.table(
        colLabels=colLabels,
        rowLabels=['MMS', 'RMS'],
        cellText=[mms_log, rms_log],
        bbox=bbox
    )
    sen_ax.table(
        colLabels=colLabels,
        rowLabels=['sensitivity^2', 'RMS'],
        cellText=[sen_lin, rms_lin],
        bbox=bbox
    )


    dis_ax.legend()
    dis_ax.set_xlabel('Mass (GeV)')
    dis_ax.set_title(f'distribution {mod}') 
    dis_fig.savefig(f'plots/testLoss/distribution_{mod}.png', bbox_inches='tight')
    print(f'saved plots/testLoss/distribution_{mod}.png') 

    res_ax.legend()
    res_ax.set_title(f'resolution {mod}')
    res_fig.savefig(f'plots/testLoss/resolution_{mod}.png', bbox_inches='tight')
    print(f'saved plots/testLoss/resolution_{mod}.png')

    sen_ax.legend()
    sen_ax.set_title(f'sensitivity {mod}')
    sen_fig.savefig(f'plots/testLoss/sensitifvity_{mod}.png', bbox_inches='tight')
    print(f'saved plots/testLoss/sensitifvity_{mod}.png')

## trend
plt.close('all')
# trend plots
from itertools import cycle

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

rmslin_fig, rmslin_ax = plt.subplots()
senlin_fig, senlin_ax = plt.subplots()
rmslog_fig, rmslog_ax = plt.subplots()
mmslog_fig, mmslog_ax = plt.subplots()
mmslin_fig, mmslin_ax = plt.subplots()

for key in mmslog_dict:

    pltkwarg = {'linestyle':next(linecycler), 'label':key} 
    rmslin_ax.plot(rmslin_dict[key], **pltkwarg) 
    senlin_ax.plot(senlin_dict[key], **pltkwarg) 
    rmslog_ax.plot(rmslog_dict[key], **pltkwarg) 
    mmslog_ax.plot(mmslog_dict[key], **pltkwarg)
    mmslin_ax.plot(mmslin_dict[key], **pltkwarg)
    
rmslin_ax.legend()
senlin_ax.legend()
rmslog_ax.legend()
mmslog_ax.legend()
mmslin_ax.legend()

rmslin_ax.set_title('RMS ratio')
senlin_ax.set_title('Sensitivity^2')
rmslog_ax.set_title('RMS log ratio')
mmslog_ax.set_title('MMS log ratio')
mmslin_ax.set_title('MMS ratio')

arr = [0,1,2,3,4,5]
xticks = [f'{x[0]}-{x[1]}' for x in pairwise(massrange)]
xticks[-1] = '180<'

rmslog_ax.set_xticks(arr,  xticks)
mmslog_ax.set_xticks(arr,  xticks)
rmslin_ax.set_xticks(arr,  xticks)
senlin_ax.set_xticks(arr, xticks)
mmslin_ax.set_xticks(arr, xticks)

for axis in [rmslog_ax, mmslog_ax, rmslin_ax, senlin_ax, mmslin_ax]:
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

rmslog_fig.savefig('plots/testLoss/trend_rmslog.png'     , bbox_inches='tight')
print('save plots/testLoss/trend_rmslog.png')
mmslog_fig.savefig('plots/testLoss/trend_mmslog.png'     , bbox_inches='tight')
print('save plots/testLoss/trend_mmslog.png')
rmslin_fig.savefig('plots/testLoss/trend_rmslin.png'     , bbox_inches='tight')
print('save plots/testLoss/trend_rmslin.png')
senlin_fig.savefig('plots/testLoss/trend_sensitivity.png', bbox_inches='tight')
print('save plots/testLoss/trend_sensitivity.png')
mmslin_fig.savefig('plots/testLoss/trend_mmslin.png', bbox_inches='tight')
print('save plots/testLoss/trend_mmslin.png') 


## resolution plots for only some files
fns = [       f'predict/testLoss/predict_wide_H_calc_mass_regr_loss0.root',
              f'predict/testLoss/predict_wide_H_calc_mass_regr_loss3.root',       
              f'predict/testLoss/predict_wide_H_calc_logMass_regr_loss0.root',
              f'predict/testLoss/predict_wide_H_calc_massOverfj_mass_regr_loss0.root',
              f'predict/testLoss/predict_wide_H_calc_massOverfj_mass_regr_loss1.root',
              f'predict/testLoss/predict_wide_H_calc_massOverfj_mass_regr_loss3.root',
    ]

labels = [x[37:].replace('regr_', '').replace('.root', '') for x in fns]#['mass_loss0', 'mass_loss3' , 'logMass_loss0', 'massOverfj_mass_loss0', 'massOverfj_mass_loss1', 'massOverfj_mass_loss3', ]

res_fig, res_ax = plt.subplots()
sen_fig, sen_ax = plt.subplots()
logmass_fig, logmass_ax = plt.subplots()

rms_log = {}
mms_log = {}
rms_lin = {}
sen_lin = {}
mms_lin = {}

for lower, upper in pairwise(massrange):
    k = f'{lower}-{upper}'
    rms_log[k] = []
    mms_log[k] = []
    rms_lin[k] = []
    sen_lin[k] = []
    mms_lin[k] = []
    #resplts[k] = plt.subplots()
    #senplts[k] = plt.subplots()
    res_fig, res_ax = plt.subplots()
    sen_fig, sen_ax = plt.subplots()
    for fn, label in zip(fns, labels):
        df = pd.DataFrame()
        f = uproot.open(fn)
        g = f['Events']
        for v in vs:
            df[v] = g[v].array()
        if 'logMass' in label:
            df['output'] = np.exp(df['output'])
            df['target_mass'] = np.exp(df['target_mass'])
        elif 'massOverfj_mass' in label:
            df['output'] = df['output']*df['fj_mass']
            df['target_mass'] = df['target_mass']*df['fj_mass']

        df = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1)]
        cutdf = df[(df['target_mass'] > lower) & (df['target_mass'] <= upper)]
        ratio = cutdf['output']/cutdf['target_mass']
        log2ratio = np.clip(np.log2(ratio), a_min=-2, a_max=2)
        ratio = np.clip(ratio, a_min=0, a_max=2)
        histkwarg['label'] = label

        ## resolution
        #resplts[k][1].hist(np.log2(ratio), range=(-1,2), **histkwarg)
        res_ax.hist(log2ratio, range=(-2,2), **histkwarg)
        ## sensitivity
        s_hist, s_edge = np.histogram(ratio, bins=101, range=(0,2.02))
        binwidth = s_edge[1]-s_edge[0]
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        #senplts[k][1].step( s_edge[:-1] , s_hist, label='loss mode {lm}')
        #sen_ax.step( s_edge[:-1] , s_hist, label=label)
        sen_ax.hist(ratio, range=(0,2), **histkwarg)

        rms_log[k].append(round(np.std(log2ratio),4))
        mms_log[k].append(round(np.mean(log2ratio),4))
        rms_lin[k].append(round(np.std(ratio),4))
        sen_lin[k].append(round(sensitivity,4))
        mms_lin[k].append(round(np.mean(ratio),4))
        
        #if lower >= 110:
        #    logmass_ax.hist(np.clip(ratio, a_min=0, a_max=2), label=f'{lower}-{upper}:rms:{round(np.std(ratio),4)}', bins=100, histtype='step')
        #    print(max(ratio))
        # {'bins':100, 'histtype':'step', 'label':f'loss mode {lm}', }
    res_ax.legend()
    sen_ax.legend()
    res_ax.set_title(f'log2(ratio) massrange:{lower}-{upper}')
    sen_ax.set_title(f'sensitivity massrange:{lower}-{upper}')

    res_fig.savefig(f'plots/testLoss/resolution_mass({lower}-{upper}).png', bbox_inches='tight')
    print(f'save plots/testLoss/resolution_mass({lower}-{upper}).png')
    sen_fig.savefig(f'plots/testLoss/sensitivity_mass({lower}-{upper}).png', bbox_inches='tight')
    print(f'save plots/testLoss/sensitivity_mass({lower}-{upper}).png')
    plt.close('all')
    #logmass_ax.legend()
    #logmass_fig.savefig(f'plots/testLoss/logmassallrange.png', bbox_inches='tight')

## write out the things
rms_log= pd.DataFrame(rms_log)
mms_log = pd.DataFrame(mms_log)
rms_lin = pd.DataFrame(rms_lin)
sen_lin = pd.DataFrame(sen_lin)
mms_lin = pd.DataFrame(mms_lin)
print('rms_log: ', rms_log)
print('mss_log: ', mms_log)
print('rms_lin: ', rms_lin)
print('sen_lin: ', sen_lin)
print('mms_lin: ', mms_lin)
