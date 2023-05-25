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
from itertools import cycle

fnslist = [[f'predict/predict_wide_H_calc_pt{ptpoint}_logMass_regr.root' for ptpoint in [150, 250,350]],
           #[f'predict/predict_wide_H_calc_pt{ptpoint}_massOverPT_regr.root' for ptpoint in [150, 250, 350]],
           ['predict/predict_wide_H_genHmassOverfj_mass.root'],
           ['predict/predict_wide_H_logGenHmassOverfj_mass.root'],
           ]

labels = ['logMass', 'massOverPT', 'genHmassOverfj_mass', 'logGenHmassOverfj_mass']
labels.remove('massOverPT')
#labels.remove('genHmassOverfj_mass')
variables = ['fj_sdmass', 'fj_mass', 'pfParticleNetMassRegressionJetTags_mass']
massrange = [0,80,95,110,135,180,99999]

lines = ["-"]#,"--","-.",":"]
linecycler = cycle(lines)
#fig, ax = plt.subplots()
pltdict = {f'{l}-{u}':plt.subplots() for l,u in pairwise(massrange)}
pltdict_log = {f'{l}-{u}':plt.subplots() for l,u in pairwise(massrange)}
rmslog_dict = {f'{l}-{u}':[] for l,u in pairwise(massrange)}
rmslin_dict = {f'{l}-{u}':[] for l,u in pairwise(massrange)}
mmslog_dict = {f'{l}-{u}':[] for l,u in pairwise(massrange)}
sensitivity_dict = {f'{l}-{u}':[] for l,u in pairwise(massrange)}

linrange = (0,2)
logrange = (-3,3)
nbins = 70


for fns, label in zip(fnslist, labels):
    hoverfj_fig, hoverfj_ax = plt.subplots()
    loghoverfj_fig, loghoverfj_ax = plt.subplots()
    df = pd.DataFrame()
    for fn in fns:
        tmpdf = pd.DataFrame()
        f = uproot.open(fn)
        g = f['Events']
        tmpdf['output'] = g['output'].array()
        tmpdf['target'] = g['target_mass'].array()
        tmpdf['pt'] = g['fj_pt'].array()
        tmpdf['fj_mass'] = g['fj_mass'].array()
        tmpdf['label'] = g['label_H_aa_bbbb'].array()
        tmpdf['genH_mass'] = g['fj_gen_H_aa_bbbb_mass_H'].array() 
        for variable in variables:
            tmpdf[variable] = g[variable].array() 
        tmpdf = tmpdf[tmpdf['label'] ==1]
        df = pd.concat([df, tmpdf], )#ignore_index=True)
        
    if label=='logMass':
        df['output'] = np.exp(df['output'])
        df['target'] = np.exp(df['target'])
    elif label=='massOverPT':
        df['output'] = df['output']*df['pt']
        df['target'] = df['target']*df['pt']
    elif label=='genHmassOverfj_mass':
        df['output'] = df['output']*df['fj_mass']
        df['target'] = df['target']*df['fj_mass']
    elif label=='logGenHmassOverfj_mass':
        df['output'] = np.exp(df['output'])*df['fj_mass']
        df['target'] = np.exp(df['target'])*df['fj_mass']

    ## selection after conversion to also catch any float math errors 
    df = df[(df['output']>0) & (df['target'] > 0)]


    linestyle = next(linecycler)
    #ax.hist(df['output']/df['target'], bins=50, histtype='step', linestyle=linestyle, label=label)
    #ax.set_title(f'resolution {label}')
    ratio = df['output']/df['target']
    if 'logGenHmassOverfj_mass' == label :
        loghoverfj_ax.hist(np.clip(np.log2(ratio), a_min=-4, a_max=4), bins=50, histtype='step', linestyle=linestyle, label=label)
    elif 'genHmassOverfj_mass' == label:
        hoverfj_ax.hist(np.clip(np.log2(ratio), a_min=-4, a_max=4), bins=50, histtype='step', linestyle=linestyle, label=label)

    ## plot by massrange 
    for lower, upper in pairwise(massrange):
        if label == 'genHmassOverfj_mass':
            name = 'massOverfj_mass'
        elif label == 'logGenHmassOverfj_mass':
            name = 'logMassOverfj_mass'
        else:
            name = label if len(label) < 15 else label[:15]
        linestyle = next(linecycler)

        print(fn)
        print(f'{lower}-{upper}')
        print('df before cut: ', df.iloc[:20])


        cut = df[(df['target'] > lower) & (df['target'] <= upper)]
        cut = cut[(cut['target']>0) & (cut['output']>0)]

        print('tmpdf after cut: ', cut.shape)

        
        key = f'{lower}-{upper}'
        ratio = np.divide(cut['output'],cut['target'])
        res = np.std(ratio)
        pltdict[key][1].hist(np.clip(ratio, a_min=linrange[0], a_max=linrange[1]), bins=nbins, range=linrange, histtype='step', label=name, linestyle=linestyle)
        pltdict[key][1].set_title(key + ' ratio')
        logres = np.std(np.log2(ratio))
        pltdict_log[key][1].hist(np.clip(np.log2(ratio), a_min=logrange[0], a_max=logrange[1]), bins=nbins, range=logrange, histtype='step', label=name, linestyle=linestyle)
        pltdict_log[key][1].set_title(f'{key} logRatio')
        
        rmslog_dict[key].append(round(logres,4))
        rmslin_dict[key].append(round(res,4))
        mmslog_dict[key].append(round(np.mean(np.log2(ratio)),4))
        s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        sensitivity_dict[key].append(round(sensitivity,4))

        
        ## add non particle net,,variables in the last loop????
        if label == 'logGenHmassOverfj_mass':
            for variable in variables:
                linestyle = next(linecycler)
                tmpcut = cut[cut[variable]>0]
                name = variable if len(variable) < 15 else variable[:15]
                ratio = np.divide(tmpcut[variable], tmpcut['target'])
                res = np.std(ratio)
                logres = np.std(np.log2(ratio))
                pltdict[key][1].hist(np.clip(ratio, a_min=linrange[0], a_max=linrange[1]), bins=nbins, range=linrange, histtype='step', label=name, linestyle=linestyle)
                pltdict_log[key][1].hist(np.clip(np.log2(ratio), a_min=logrange[0], a_max=logrange[1]), bins=nbins, range=logrange, histtype='step', label=name, linestyle=linestyle)

                rmslog_dict[key].append(round(logres,4))
                rmslin_dict[key].append(round(res,4))
                mmslog_dict[key].append(round(np.mean(np.log2(ratio)),4))
                s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
                sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                sensitivity_dict[key].append(round(sensitivity,4))
       
    

for lower,upper in pairwise(massrange):
    key = f'{lower}-{upper}'

    if key == '110-135':
        print(sensitivity_dict[key])
        print(rmslin_dict[key])


    tablelabs = labels+variables
    tablelabs = [x if len(x) < 12 else x[:12] for x in tablelabs]
    bbox = [-0.1, -0.3, 1.5, 0.2]
    tb1 = pltdict[key][1].table(
        colLabels=tablelabs,
        rowLabels=['sensitivity^2', 'RMS'],
        cellText = [sensitivity_dict[key], rmslin_dict[key]],
        bbox=bbox
    )

    tb2 = pltdict_log[key][1].table(
        colLabels=tablelabs,
        rowLabels=['MMS', 'RMS'],
        cellText=[mmslog_dict[key], rmslog_dict[key]],
        bbox=bbox
    )

    for tb in [tb1, tb2]:
        tb.auto_set_font_size(False)
        tb.set_fontsize(10)
        
    box = pltdict[key][1].get_position()
    pltdict[key][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    pltdict[key][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pltdict[key][0].savefig(f'plots/genmass/{key}.png', bbox_inches='tight')
    box = pltdict_log[key][1].get_position()
    pltdict_log[key][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    pltdict_log[key][1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pltdict_log[key][0].savefig(f'plots/genmass/log_{key}.png', bbox_inches='tight')
    


loghoverfj_ax.set_title('resolution H_calc logMassOverfj_mass')
hoverfj_ax.set_title('resolution H_calc massOverfj_mass')
loghoverfj_fig.savefig('plots/genmass/resolution_H_calc_logMassOverfj_mass.png', bbox_inches='tight')
hoverfj_fig.savefig('plots/genmass/resolution_H_calc_massOverfj_mass.png', bbox_inches='tight')

    
    
