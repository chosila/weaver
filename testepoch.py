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
mods = ['mass', 'logMass']#['mass', 'logMass', 'massOverfj_mass']
loss_modes = [0,3]#[0,1,3]
massrange = [0,80,95,110,135,180,99999]
#loss_modes.remove(2)
vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no' ]
epochs = [0, 1, 2, 4, 6, 8, 12, 16] # [10,15,20,25,30,35,40]
from itertools import cycle

lines = ["-","--","-.",":"]
linecycler = cycle(lines)


flist = [
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-10.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-10.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-11.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-11.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-12.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-12.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-13.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-13.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-14.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-15.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-16.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-17.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-18.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-20.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-21.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-23.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-25.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-27.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-30.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-32.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-35.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-37.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-40.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-42.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-45.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-47.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-50.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-52.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-55.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-57.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-60.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-62.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-8.5.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-9.0.root',
    '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-9.5.root',
]

## strip after M-.....root, strip.root,
massranges = [x[x.find('_M-')+3:].split('.root')[0] for x in flist]
massranges = [float(x) for x in massranges]
massranges.sort()

#massranges = [12.5, 16.0, 21.5, 32.5, 45, 57.5]

## test overtrain for each mod, multiple epochs, one mass range
for mod in mods:
    for lm in loss_modes:
        for massrange in massranges:
            rmslintestlist = []
            senlintestlist = []
            mmslogtestlist = []
            rmslogtestlist = []

            rmslintrainlist = []
            senlintrainlist = []
            mmslogtrainlist = []
            rmslogtrainlist = []
            
            for epoch in epochs: 
                fn = f'predict/testepoch/predict_a1_calc_{mod}_regr_loss{lm}_epoch{epoch}.root'
                df = pd.DataFrame()
                f = uproot.open(fn)
                g = f['Events']
                print(np.array(g['output'].array()))
                
                for v in vs:
                    
                    df[v] = np.array(g[v].array())
                if mod == 'logMass':
                    df['output'] = np.exp(df['output'])
                    df['target_mass'] = np.exp(df['target_mass'])
                dftest = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1) & (df['event_no']%2==1)]
                dftrain = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1) & (df['event_no']%2==0)]
                dftest = dftest[np.isclose(dftest['target_mass'], massrange, rtol=1e-03, atol=1e-05)]
                dftrain = dftrain[np.isclose(dftrain['target_mass'], massrange, rtol=1e-03, atol=1e-05)]   
                
                dftest['ratio'] = np.divide(dftest['output'] , dftest['target_mass'])
                dftrain['ratio'] = np.divide(dftrain['output'] , dftrain['target_mass'])
                dftest['log2ratio'] = np.clip(np.log2(dftest['ratio']), a_min=-2, a_max=2)
                dftrain['log2ratio'] = np.clip(np.log2(dftrain['ratio']), a_min=-2, a_max=2)

                
                ## calculate rms and stuff
                rmslintestlist.append(np.std(dftest['ratio']))
                mmslogtestlist.append(np.mean(dftest['log2ratio']))
                rmslogtestlist.append(np.std(dftest['log2ratio']))

                s_hist, s_edge = np.histogram(dftest['ratio'], bins=101, range=(0,2.02))
                binwidth = s_edge[1]-s_edge[0]
                sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                senlintestlist.append(sensitivity)

                rmslintrainlist.append(np.std(dftrain['ratio']))
                mmslogtrainlist.append(np.mean(dftrain['log2ratio']))
                rmslogtrainlist.append(np.std(dftrain['log2ratio']))

                s_hist, s_edge = np.histogram(dftrain['ratio'], bins=101, range=(0,2.02))
                binwidth = s_edge[1]-s_edge[0]
                sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                senlintrainlist.append(sensitivity)


            figsize = (9,6.7)
            mmslog = plt.subplots(figsize=figsize)
            rmslin = plt.subplots(figsize=figsize)
            rmslog = plt.subplots(figsize=figsize)
            senplt = plt.subplots(figsize=figsize)

                
            for pltobj, testval, trainval, savename, yrange in zip([mmslog, rmslin, rmslog, senplt],
                                                           [mmslogtestlist, rmslintestlist, rmslogtestlist, senlintestlist],
                                                           [mmslogtrainlist, rmslintrainlist, rmslogtrainlist, senlintrainlist],
                                                                   ['mms log ratio', 'rms ratio', 'rms log ratio', 'sensitivity'],
                                                                   [(-.4, .4), (0,.3), (0,.4), (0,15)]):

                xticks = [str(x) for x in epochs]
                arr = list(range(len(xticks)))
                pltobj[1].plot(trainval, label='train')
                pltobj[1].plot(testval, label='test')
                pltobj[1].set_xticks(arr, xticks,)
                pltobj[1].set_xlabel('epochs')
                plt.grid()
                pltobj[1].legend()
                pltobj[1].set_title(f'compare {savename} train/test performance {mod} loss{lm} mass point {massrange}')
                pltobj[1].set_ylim([yrange[0], yrange[1]])
                #Path(f"plots/testepoch/").mkdir(parents=True, exist_ok=True)
                pltobj[0].savefig(f'plots/testepoch/{savename}_train_test_{mod}_loss{lm}_masspoint{massrange}.png', bbox_inches='tight')
                plt.close(pltobj[0])
                

sys.exit()

## test epoch trends 
for mod in mods:
    for lm in loss_modes:
        dis_fig, dis_ax = plt.subplots(figsize=(6,6))
        disTarg_fig, disTarg_ax = plt.subplots(figsize=(6,6))
        res_fig, res_ax = plt.subplots(figsize=(6,6))
        sen_fig, sen_ax = plt.subplots(figsize=(6,6))
        rms_log = []
        mms_log = []
        rms_lin = []
        sen_lin = []
        for epoch in epochs: 
            fn = f'predict/testepoch/predict_a1_calc_{mod}_regr_loss{lm}_epoch{epoch}.root'
            df = pd.DataFrame()
            f = uproot.open(fn)
            g = f['Events']
            print(g.keys())
            for v in vs:
                print(fn)
                print(g[v])
                df[v] = g[v].array()
            if mod == 'logMass':
                df['output'] = np.exp(df['output'])
                df['target_mass'] = np.exp(df['target_mass'])

            df = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1) & (df['event_no']%2==1)]

            ## distribution
            histkwarg = {'bins':100, 'histtype':'step', 'label':f'epoch {epoch}', }
            dis_ax.hist(df['output'], range=(0,60), **histkwarg)
            disTarg_ax.hist(df['target_mass'], range=(0,60), **histkwarg)
            
            ## resolution
            ratio = np.divide(df['output'] , df['target_mass'])
            log2ratio = np.clip(np.log2(ratio), a_min=-2, a_max=2)
            ratio = np.clip(ratio, a_min=0, a_max=2)
            res_ax.hist(log2ratio, range=(-2,2),**histkwarg)

            ## sensitivity
            s_hist, s_edge = np.histogram(ratio, bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            sen_ax.step( s_edge[:-1] , s_hist, label=f'loss mode {lm}')

            rms_log.append(round(np.std(log2ratio),4))
            mms_log.append(round(np.mean(log2ratio),4))
            rms_lin.append(round(np.std(ratio),4))
            sen_lin.append(round(sensitivity,4))

            rmslin_dict[f'{mod}_loss{lm}'] = [] 
            senlin_dict[f'{mod}_loss{lm}'] = [] 
            rmslog_dict[f'{mod}_loss{lm}'] = [] 
            mmslog_dict[f'{mod}_loss{lm}'] = [] 
            mmslin_dict[f'{mod}_loss{lm}'] = []


        colLabels = [f'epoch: {epoch}' for epoch in epochs]
        bbox = [0.1, -0.3, 0.9, 0.2]

        pltkwarg = {'linestyle':next(linecycler), 'label': f'{mod} loss{lm}'}
        trendmmslog[1].plot(mms_log, **pltkwarg)
        trendrmslog[1].plot(rms_log, **pltkwarg)
        trendsens[1].plot(sen_lin, **pltkwarg)
        trendrmslin[1].plot(rms_lin, **pltkwarg)
        
        
        
        dis_ax.legend()
        dis_ax.set_xlabel('Mass (GeV)')
        dis_ax.set_title(f'distribution {mod} loss{lm}') 
        dis_fig.savefig(f'plots/testepoch/distribution_{mod}_loss{lm}.png', bbox_inches='tight')
        print(f'saved plots/testepoch/distribution_{mod}_loss{lm}.png') 

        disTarg_ax.legend()
        disTarg_ax.set_xlabel('Mass (GeV)')
        disTarg_ax.set_title(f'distribution target {mod} loss{lm}') 
        disTarg_fig.savefig(f'plots/testepoch/distribution_target_{mod}_loss{lm}.png', bbox_inches='tight')
        print(f'saved plots/testepoch/distribution_target_{mod}_loss{lm}.png') 

        
        res_ax.legend()
        res_ax.set_title(f'resolution {mod} loss{lm}')
        res_fig.savefig(f'plots/testepoch/resolution_{mod}_loss{lm}.png', bbox_inches='tight')
        print(f'saved plots/testepoch/resolution_{mod}_loss{lm}.png')

        sen_ax.legend()
        sen_ax.set_title(f'sensitivity {mod} loss{lm}')
        sen_fig.savefig(f'plots/testepoch/sensitifvity_{mod}_loss{lm}.png', bbox_inches='tight')
        print(f'saved plots/testepoch/sensitifvity_{mod}_loss{lm}.png')



trendmmslog[1].set_xlabel('Epoch')
trendmmslog[1].set_title('MMS log ratio')
trendmmslog[1].legend()
trendmmslog[0].savefig(f'plots/testepoch/trend_mmslog.png', bbox_inches='tight')

trendrmslog[1].set_xlabel('Epoch')
trendrmslog[1].set_title('RMS log ratio')
trendrmslog[1].legend()
trendrmslog[0].savefig(f'plots/testepoch/trend_rmslog.png', bbox_inches='tight')

trendsens[1].set_xlabel('Epoch')
trendsens[1].set_title('Sensitivity ^2')
trendsens[1].legend()
trendsens[0].savefig(f'plots/testepoch/trend_sensitivity.png', bbox_inches='tight')

trendrmslin[1].set_xlabel('Epoch')
trendrmslin[1].set_title('RMS ratio')
trendrmslin[1].legend()
trendrmslin[0].savefig(f'plots/testepoch/trend_rmslin.png', bbox_inches='tight')

sys.exit()

        
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
    print(rmslin_dict[key])
    print( senlin_dict[key])
    print( rmslog_dict[key])
    print( mmslog_dict[key])
    print( mmslin_dict[key])
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


xticks = epochs#[f'{x[0]}-{x[1]}' for x in pairwise(massrange)]
#xticks[-1] = '180<'
arr = list(range(len(epochs)))

#rmslog_ax.set_xticks(arr,  xticks)
#mmslog_ax.set_xticks(arr,  xticks)
#rmslin_ax.set_xticks(arr,  xticks)
#senlin_ax.set_xticks(arr, xticks)
#mmslin_ax.set_xticks(arr, xticks)

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
senlin_fig.savefig('plots/testepoch/trend_sensitivity.png', bbox_inches='tight')
print('save plots/testepoch/trend_sensitivity.png')
mmslin_fig.savefig('plots/testepoch/trend_mmslin.png', bbox_inches='tight')
print('save plots/testepoch/trend_mmslin.png') 


## resolution plots for only some files
# fns = [       f'predict/testepoch/predict_wide_H_calc_mass_regr_loss0.root',
#               f'predict/testepoch/predict_wide_H_calc_mass_regr_loss3.root',       
#               f'predict/testepoch/predict_wide_H_calc_logMass_regr_loss0.root',
#               f'predict/testepoch/predict_wide_H_calc_massOverfj_mass_regr_loss0.root',
#               f'predict/testepoch/predict_wide_H_calc_massOverfj_mass_regr_loss1.root',
#               f'predict/testepoch/predict_wide_H_calc_massOverfj_mass_regr_loss3.root',
#     ]

# labels = [x[37:].replace('regr_', '').replace('.root', '') for x in fns]#['mass_loss0', 'mass_loss3' , 'logMass_loss0', 'massOverfj_mass_loss0', 'massOverfj_mass_loss1', 'massOverfj_mass_loss3', ]

# res_fig, res_ax = plt.subplots()
# sen_fig, sen_ax = plt.subplots()
# logmass_fig, logmass_ax = plt.subplots()

# rms_log = {}
# mms_log = {}
# rms_lin = {}
# sen_lin = {}
# mms_lin = {}

# for lower, upper in pairwise(massrange):
#     k = f'{lower}-{upper}'
#     rms_log[k] = []
#     mms_log[k] = []
#     rms_lin[k] = []
#     sen_lin[k] = []
#     mms_lin[k] = []
#     #resplts[k] = plt.subplots()
#     #senplts[k] = plt.subplots()
#     res_fig, res_ax = plt.subplots()
#     sen_fig, sen_ax = plt.subplots()
#     for fn, label in zip(fns, labels):
#         df = pd.DataFrame()
#         f = uproot.open(fn)
#         g = f['Events']
#         for v in vs:
#             df[v] = g[v].array()
#         if 'logMass' in label:
#             df['output'] = np.exp(df['output'])
#             df['target_mass'] = np.exp(df['target_mass'])
#         elif 'massOverfj_mass' in label:
#             df['output'] = df['output']*df['fj_mass']
#             df['target_mass'] = df['target_mass']*df['fj_mass']

#         df = df[(df['output'] > 0) & (df['target_mass'] > 0) & (df['label_H_aa_bbbb']==1)]
#         cutdf = df[(df['target_mass'] > lower) & (df['target_mass'] <= upper)]
#         ratio = cutdf['output']/cutdf['target_mass']
#         log2ratio = np.clip(np.log2(ratio), a_min=-2, a_max=2)
#         ratio = np.clip(ratio, a_min=0, a_max=2)
#         histkwarg['label'] = label

#         ## resolution
#         #resplts[k][1].hist(np.log2(ratio), range=(-1,2), **histkwarg)
#         res_ax.hist(log2ratio, range=(-2,2), **histkwarg)
#         ## sensitivity
#         s_hist, s_edge = np.histogram(ratio, bins=101, range=(0,2.02))
#         binwidth = s_edge[1]-s_edge[0]
#         sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
#         #senplts[k][1].step( s_edge[:-1] , s_hist, label='loss mode {lm}')
#         #sen_ax.step( s_edge[:-1] , s_hist, label=label)
#         sen_ax.hist(ratio, range=(0,2), **histkwarg)

#         rms_log[k].append(round(np.std(log2ratio),4))
#         mms_log[k].append(round(np.mean(log2ratio),4))
#         rms_lin[k].append(round(np.std(ratio),4))
#         sen_lin[k].append(round(sensitivity,4))
#         mms_lin[k].append(round(np.mean(ratio),4))
        
#         #if lower >= 110:
#         #    logmass_ax.hist(np.clip(ratio, a_min=0, a_max=2), label=f'{lower}-{upper}:rms:{round(np.std(ratio),4)}', bins=100, histtype='step')
#         #    print(max(ratio))
#         # {'bins':100, 'histtype':'step', 'label':f'loss mode {lm}', }
#     res_ax.legend()
#     sen_ax.legend()
#     res_ax.set_title(f'log2(ratio) massrange:{lower}-{upper}')
#     sen_ax.set_title(f'sensitivity massrange:{lower}-{upper}')

#     res_fig.savefig(f'plots/testepoch/resolution_mass({lower}-{upper}).png', bbox_inches='tight')
#     print(f'save plots/testepoch/resolution_mass({lower}-{upper}).png')
#     sen_fig.savefig(f'plots/testepoch/sensitivity_mass({lower}-{upper}).png', bbox_inches='tight')
#     print(f'save plots/testepoch/sensitivity_mass({lower}-{upper}).png')
#     plt.close('all')
#     #logmass_ax.legend()
#     #logmass_fig.savefig(f'plots/testepoch/logmassallrange.png', bbox_inches='tight')

# ## write out the things
# rms_log= pd.DataFrame(rms_log)
# mms_log = pd.DataFrame(mms_log)
# rms_lin = pd.DataFrame(rms_lin)
# sen_lin = pd.DataFrame(sen_lin)
# mms_lin = pd.DataFrame(mms_lin)
# print('rms_log: ', rms_log)
# print('mss_log: ', mms_log)
# print('rms_lin: ', rms_lin)
# print('sen_lin: ', sen_lin)
# print('mms_lin: ', mms_lin)
        
