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

density = False
lines = ["-",'-', ":",':',"--",'--',"-.",'-.']
linecycler = cycle(lines)
masspoints = [12,15,20,25,30,35,40,45,50,55,60]
oddmps = [15,25,35,45,55]
evenmps = [12,20,30,40,50,60]
fns = [f'predict/predict_wide_H_calc_central_M{mpnt}_logMass_regr.root' for mpnt in masspoints]
rootfilevars = ['output', 'target_mass', 'label_H_aa_bbbb', 'event_no']
fjvars = ['fj_sdmass', 'fj_mass', 'pfParticleNetMassRegressionJetTags_mass']
fig, ax = plt.subplots()
odd_fig, odd_ax = plt.subplots()
even_fig, even_ax = plt.subplots()
concdf = pd.DataFrame()
for fn, mp in zip(fns, masspoints):
    f = uproot.open(fn)
    g = f['Events']
    df = pd.DataFrame()
    for x in rootfilevars+fjvars:
        df[x] = g[x].array()
    df['output'] = np.exp(df['output'])
    df['target_mass'] = np.exp(df['target_mass'])
    df = df[(df['output'] >0) & (df['target_mass']>0) & (df['label_H_aa_bbbb']==1)]
    histrange = (0,225)
    concdf = pd.concat([concdf, df])
    


for pltvar in [fjvars+['output']]:
    label = pltvar[:10] if len(pltvar) > 10 else pltvar
    ax.hist(np.clip(concdf[pltvar], histrange[0], histrange[1]), bins=100, range=histrange, histtype='step', linestyle=next(linecycler), label=label, density=density) 

ax.legend(bbox_to_anchor=[1.05,1.05])
ax.set_title('distribution wideH logMass H_calc model on central data')
ax.set_xlabel('Mass (GeV)')
fig.savefig('plots/distribution_wideH_logMass_H_calc_centralData.png', bbox_inches='tight') 



## predict/predict_wide_H_calc_pt150_logMass_regr.root
## A plot of the regression to logMass, overlaid for mH-70 events in 4 mass windows: 80-90 GeV, 120-130 GeV, 170-190 GeV, 235-265
massranges = [(80,90), (120, 130), (170,190), (235,265)]
fns = [f'predict/predict_wide_H_calc_pt{ptpnt}_logMass_regr.root' for ptpnt in [150,250,350]]
df = pd.DataFrame()

for fn in fns:
    f = uproot.open(fn)
    g = f['Events']
    tmpdf = pd.DataFrame()
    for v in rootfilevars:
        tmpdf[v] = g[v].array()
    tmpdf['output'] = np.exp(tmpdf['output'])
    tmpdf['target_mass'] = np.exp(tmpdf['target_mass'])
    tmpdf = tmpdf[(tmpdf['output'] >0) & (tmpdf['target_mass']>0) & (tmpdf['label_H_aa_bbbb']==1)]
    df = pd.concat([df, tmpdf])

fig, ax = plt.subplots()
for lower, upper in massranges:
    cutdf = df[(df['output'] > lower) & (df['output'] <= upper)]
    ax.hist(cutdf['output'], bins=5, histtype='step', label=f'{lower}-{upper}')

ax.legend()
ax.set_title('Distribution of logMass H_calc over specific mass ranges')
ax.set_xlabel('Mass (GeV)')
fig.savefig('plots/distribution_H_calc_logMass_massranges.png', bbox_inches='tight')


'''
    if mp%2==0:
        even_ax.hist(np.clip(df['output'], histrange[0], histrange[1]), bins=100, range=histrange, histtype='step', linestyle=linestyle, label=f'M-{mp}')
    elif mp%2==1:
        odd_ax.hist(np.clip(df['output'], histrange[0], histrange[1]), bins=100, range=histrange, histtype='step', linestyle=linestyle, label=f'M-{mp}')
    if mp==60:
        for fjvar in fjvars:
            linestyle=next(linecycler)
            lab = fjvar[:10] if len(fjvar)>10 else fjvar 
            ax.hist(np.clip(df[fjvar], histrange[0], histrange[1]), bins=100, range=histrange, histtype='step', linestyle=linestyle, label=lab)
            even_ax.hist(np.clip(df[fjvar], histrange[0], histrange[1]), bins=100, range=histrange, histtype='step', linestyle=linestyle, label=lab)
    elif mp==55:
        for fjvar in fjvars:
            lab = fjvar[:10] if len(fjvar)>10 else fjvar
            odd_ax.hist(np.clip(df[fjvar], histrange[0], histrange[1]), bins=100, range=histrange, histtype='step', linestyle=next(linecycler), label=lab)

        
ax.legend(bbox_to_anchor=[1.05, 1.05])
ax.set_title('wideH logMass H_calc model on central data')
ax.set_xlabel('Mass (GeV)')
fig.savefig('plots/distribution_wideH_logMass_H_calc_centralData.png', bbox_inches='tight')

even_ax.legend(bbox_to_anchor=[1.05, 1.05])
even_ax.set_title('wideH logMass H_calc model on central data (even)')
even_ax.set_xlabel('Mass (GeV)')
even_fig.savefig('plots/distribution_wideH_logMass_H_calc_centralData_even.png', bbox_inches='tight')

odd_ax.legend(bbox_to_anchor=[1.05, 1.05])
odd_ax.set_title('wideH logMass H_calc model on central data (odd)')
odd_ax.set_xlabel('Mass (GeV)')
odd_fig.savefig('plots/distribution_wideH_logMass_H_calc_centralData_odd.png', bbox_inches='tight')



'''
