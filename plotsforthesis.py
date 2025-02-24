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
from itertools import cycle

a_massranges = [8, 12.5, 16.0, 21.5, 32.5, 45, 57.5, 62] 
H_massranges = [0,80,95,110,135,180,250]
runTargets = 1
runLossmodes = 1
runEpoch = 1
run34b = 1 
vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no', 'fj_pt']
#colors = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
#lines = ["-","--","-.",":"] ## do we even need this 
# linecycler = cycle(lines)
plotdir = 'thesisplots'


if runTargets: 

    mods = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT', 'ptOverMass']
    masspoints = [12,15, 20,25,30,35,40,45,50,55,60]

    rmslinplt = plt.subplots()
    senlinplt = plt.subplots()
    mmslogplt = plt.subplots()
    rmslogplt = plt.subplots()
    
    # rmslinplt[1].set_prop_cycle('color', colors)
    # senlinplt[1].set_prop_cycle('color', colors)
    # mmslogplt[1].set_prop_cycle('color', colors)
    # rmslogplt[1].set_prop_cycle('color', colors)

    for mod in mods:
        ## for storing the scores by mass range
        rmslinlist = []
        senlinlist = []
        mmsloglist = []
        rmsloglist = []
        # fns = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in masspoints]
        for masspoint in masspoints: 
            
            df = pd.DataFrame()
            #gs = [uproot.open(fn)['Events'] for fn in fns]
            fn = f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root'
            g = uproot.open(fn)['Events']
            ## add each v variable into the df 
            for v in vs:
                df[v] =  np.array(g[v].array())

            df = df[df['label_H_aa_bbbb']==1]
           
            ## convert output and target back to mass value so it makes sense whne plotting distribution
            if 'logMassOverPT' == mod:
                df['output'] = np.exp(df['output'])*df['fj_pt']
                df['target_mass'] = np.exp(df['target_mass'])*df['fj_pt']
            elif 'logMass' == mod:
                df['output'] = np.exp(df['output'])
                df['target_mass'] = np.exp(df['target_mass'])
            elif '1OverMass' == mod:
                df['output'] = 1/df['output']
                df['target_mass'] = 1/df['target_mass']
                #binrange= [0,70]
            elif 'massOverPT' == mod:
                df['output'] = df['output']*df['fj_pt']
                df['target_mass'] = df['target_mass']*df['fj_pt']
                df['output'][df['output'] < 0] = 0
                #binrange = (-2,2)
            elif 'ptOverMass' == mod:
                df['output'] = df['fj_pt']/df['output']
                df['target_mass'] = df['fj_pt']/df['target_mass']
    
            ## calculate ratios and resolutions
            df = df[df['output'] >= 0 ]
            df['ratio'] = np.divide(df['output'] , df['target_mass'])
            df['log2ratio'] = np.clip(np.log2(df['ratio']), a_min=-2, a_max=2)
            
            ## calculatate the scores for each mass range and append them to corresponding list 
            rmslinlist.append(np.std( df['ratio']    , )) 
            mmsloglist.append(np.mean(df['log2ratio'], )) 
            rmsloglist.append(np.std( df['log2ratio'], )) 
    
            s_hist, s_edge = np.histogram(df['ratio'], bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            senlinlist.append(sensitivity)

        if (mod=='mass') or (mod=='logMass'):
            print(mmsloglist)
        ## plot this mod's scores on each plot
        rmslinplt[1].plot(masspoints, rmslinlist, label=mod)
        senlinplt[1].plot(masspoints, senlinlist, label=mod)
        mmslogplt[1].plot(masspoints, mmsloglist, label=mod)
        rmslogplt[1].plot(masspoints, rmsloglist, label=mod)

    ## styling of plot
    rmslinplt[1].legend()
    senlinplt[1].legend()
    mmslogplt[1].legend()
    rmslogplt[1].legend()

    rmslinplt[1].set_xlabel('Mass of pseudoscalar [GeV]', fontsize=14, loc='right')
    senlinplt[1].set_xlabel('Mass of pseudoscalar [GeV]', fontsize=14, loc='right')
    mmslogplt[1].set_xlabel('Mass of pseudoscalar [GeV]', fontsize=14, loc='right')
    rmslogplt[1].set_xlabel('Mass of pseudoscalar [GeV]', fontsize=14, loc='right')

    rmslinplt[1].set_ylabel('Scores', fontsize=14, loc='top')
    senlinplt[1].set_ylabel('Scores', fontsize=14, loc='top')
    mmslogplt[1].set_ylabel('Scores', fontsize=14, loc='top')
    rmslogplt[1].set_ylabel('Scores', fontsize=14, loc='top')

    rmslinplt[1].set_title('a mass Linear RMS',  loc='right', fontsize =18 )
    senlinplt[1].set_title('a mass Sensitivity', loc='right', fontsize =18 )
    mmslogplt[1].set_title('a mass MMS',         loc='right', fontsize =18 )
    rmslogplt[1].set_title('a mass Log RMS',     loc='right', fontsize =18 )

    rmslinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslinplt[1].transAxes)
    senlinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=senlinplt[1].transAxes)
    mmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=mmslogplt[1].transAxes)
    rmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslogplt[1].transAxes)

    rmslinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslinplt[1].transAxes)
    senlinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=senlinplt[1].transAxes)
    mmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=mmslogplt[1].transAxes)
    rmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslogplt[1].transAxes)

    
    rmslinplt[0].savefig(f'{plotdir}/targets_overlay_rmslin.pdf', bbox_inches='tight')
    senlinplt[0].savefig(f'{plotdir}/targets_overlay_senlin.pdf', bbox_inches='tight')
    mmslogplt[0].savefig(f'{plotdir}/targets_overlay_mmslog.pdf', bbox_inches='tight')
    rmslogplt[0].savefig(f'{plotdir}/targets_overlay_rmslog.pdf', bbox_inches='tight')
    
    

    # wideH. these did not work figure out if this is trained on central then test on wide or trained and test on wide
    rmslinplt = plt.subplots()
    senlinplt = plt.subplots()
    mmslogplt = plt.subplots()
    rmslogplt = plt.subplots()

    for mod in mods: 
        fns = [f'predict/predict_wide_H_calc_pt{m}_{mod}_regr.root' for m in [150, 250, 350]]
        gs = [uproot.open(fn)['Events'] for fn in fns]
        df = pd.DataFrame()
        rmslinlist = []
        senlinlist = []
        mmsloglist = []
        rmsloglist = []
        for g in gs:
            tmpdf = pd.DataFrame()
            for v in vs:
                if (('target' in g) and (v =='target_mass')): ## some files have target in instead of target mass. need to account
                    tmpdf['target_mass'] = np.array(g['target'].array())
                else:
                    tmpdf[v] = np.array(g[v].array())
            df = df.append(tmpdf)

        df = df[df['label_H_aa_bbbb']==1]
        ## convert output and target back to mass value so it makes sense whne plotting distribution
        if 'logMassOverPT' == mod:
            df['output'] = np.exp(df['output'])*df['fj_pt']
            df['target_mass'] = np.exp(df['target_mass'])*df['fj_pt']
        elif 'logMass' == mod:
            df['output'] = np.exp(df['output'])
            df['target_mass'] = np.exp(df['target_mass'])
        elif '1OverMass' == mod:
            df['output'] = 1/df['output']
            df['target_mass'] = 1/df['target_mass']
        elif 'massOverPT' == mod:
            df['output'] = df['output']*df['fj_pt']
            df['target_mass'] = df['target_mass']*df['fj_pt']
            df['output'][df['output'] < 0] = 0
        elif 'ptOverMass' == mod:
            df['output'] = df['fj_pt']/df['output']
            df['target_mass'] = df['fj_pt']/df['target_mass']

        ## calculate ratios and resolutions
        df = df[df['output'] >= 0 ]
        
        df['ratio'] = np.divide(df['output'] , df['target_mass'])
        df['log2ratio'] = np.clip(np.log2(df['ratio']), a_min=-2, a_max=2)

        x_massrange_tick = []
        for massrange in zip(H_massranges[:-1], H_massranges[1:]):
            clippeddf = df[(df['target_mass'] > massrange[0]) & (df['target_mass'] <= massrange[1])] 

            ## calculatate the scores for each mass range and append them to corresponding list
            rmslinlist.append(np.std( clippeddf['ratio']    , ))
            mmsloglist.append(np.mean(clippeddf['log2ratio'], ))
            rmsloglist.append(np.std( clippeddf['log2ratio'], ))
              
            s_hist, s_edge = np.histogram(clippeddf['ratio'], bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            senlinlist.append(sensitivity)



        
        ## plot this mod's scores on each plot
        rmslinplt[1].plot(range(len(rmslinlist)), rmslinlist, label=mod)
        senlinplt[1].plot(range(len(rmslinlist)), senlinlist, label=mod)
        mmslogplt[1].plot(range(len(rmslinlist)), mmsloglist, label=mod)
        rmslogplt[1].plot(range(len(rmslinlist)), rmsloglist, label=mod)


    ## styling of plot
    rmslinplt[1].legend()
    senlinplt[1].legend()
    mmslogplt[1].legend()
    rmslogplt[1].legend()

    rmslinplt[1].set_xlabel('Mass of Higgs [GeV]')
    senlinplt[1].set_xlabel('Mass of Higgs [GeV]')
    mmslogplt[1].set_xlabel('Mass of Higgs [GeV]')
    rmslogplt[1].set_xlabel('Mass of Higgs [GeV]')

    rmslinplt[1].set_ylabel('Scores')
    senlinplt[1].set_ylabel('Scores')
    mmslogplt[1].set_ylabel('Scores')
    rmslogplt[1].set_ylabel('Scores')

    rmslinplt[1].set_title('wide H mass Linear RMS',  loc='right', fontsize =18 )
    senlinplt[1].set_title('wide H mass Sensitivity', loc='right', fontsize =18 )
    mmslogplt[1].set_title('wide H mass MMS',         loc='right', fontsize =18 )
    rmslogplt[1].set_title('wide H mass Log RMS',     loc='right', fontsize =18 )

    rmslinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslinplt[1].transAxes)
    senlinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=senlinplt[1].transAxes)
    mmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=mmslogplt[1].transAxes)
    rmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslogplt[1].transAxes)

    rmslinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslinplt[1].transAxes)
    senlinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=senlinplt[1].transAxes)
    mmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=mmslogplt[1].transAxes)
    rmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslogplt[1].transAxes)

    xtickslabels = ['0-80', '80-95', '95-110', '110-135', '135-180', '180<']

    rmslinplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    senlinplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    mmslogplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    rmslogplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    
    rmslinplt[0].savefig(f'{plotdir}/targets_overlay_rmslin_wideH.pdf', bbox_inches='tight')
    senlinplt[0].savefig(f'{plotdir}/targets_overlay_senlin_wideH.pdf', bbox_inches='tight')
    mmslogplt[0].savefig(f'{plotdir}/targets_overlay_mmslog_wideH.pdf', bbox_inches='tight')
    rmslogplt[0].savefig(f'{plotdir}/targets_overlay_rmslog_wideH.pdf', bbox_inches='tight')

plt.close('all')
    # rmslinplt[1]
    # senlinplt[1]
    # mmslogplt[1]
    # rmslogplt[1]

    
    
    
    ## 

if runLossmodes:
    mods = ['mass', 'logMass']
    lms = [0,1,3]
    vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass' ]
    
    rmslinplt = plt.subplots()
    senlinplt = plt.subplots()
    mmslogplt = plt.subplots()
    rmslogplt = plt.subplots()

    
    for mod in mods:
        for lm in lms:
            fn = f'predict/testLoss/predict_wide_H_calc_{mod}_regr_loss{lm}.root'
            g = uproot.open(fn)['Events']
            df = pd.DataFrame()
            rmslinlist = []
            senlinlist = []
            mmsloglist = []
            rmsloglist = []
            
            for v in vs:
                df[v] = np.array(g[v].array())
    
            df = df[df['label_H_aa_bbbb']==1]
            ## convert output and target back to mass value so it makes sense whne plotting distribution
            if 'logMass' == mod:
                df['output'] = np.exp(df['output'])
                df['target_mass'] = np.exp(df['target_mass'])
    
            ## calculate ratios and resolutions
            df = df[df['output'] >= 0 ]
            
            df['ratio'] = np.divide(df['output'] , df['target_mass'])
            df['log2ratio'] = np.clip(np.log2(df['ratio']), a_min=-2, a_max=2)

            
            x_massrange_tick = []
            for massrange in zip(H_massranges[:-1], H_massranges[1:]):
                clippeddf = df[(df['target_mass'] > massrange[0]) & (df['target_mass'] <= massrange[1])] 
    
                ## calculatate the scores for each mass range and append them to corresponding list
                rmslinlist.append(np.std( clippeddf['ratio']    , ))
                mmsloglist.append(np.mean(clippeddf['log2ratio'], ))
                rmsloglist.append(np.std( clippeddf['log2ratio'], ))
                  
                s_hist, s_edge = np.histogram(clippeddf['ratio'], bins=101, range=(0,2.02))
                binwidth = s_edge[1]-s_edge[0]
                sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                senlinlist.append(sensitivity)
        
            ## plot this mod's scores on each plot
            rmslinplt[1].plot(range(len(rmslinlist)), rmslinlist, label=f'{mod}_loss{lm}')
            senlinplt[1].plot(range(len(rmslinlist)), senlinlist, label=f'{mod}_loss{lm}')
            mmslogplt[1].plot(range(len(rmslinlist)), mmsloglist, label=f'{mod}_loss{lm}')
            rmslogplt[1].plot(range(len(rmslinlist)), rmsloglist, label=f'{mod}_loss{lm}')


    ## styling of plot
    rmslinplt[1].legend()
    senlinplt[1].legend()
    mmslogplt[1].legend()
    rmslogplt[1].legend()

    rmslinplt[1].set_xlabel('Mass of Higgs [GeV]')
    senlinplt[1].set_xlabel('Mass of Higgs [GeV]')
    mmslogplt[1].set_xlabel('Mass of Higgs [GeV]')
    rmslogplt[1].set_xlabel('Mass of Higgs [GeV]')

    rmslinplt[1].set_ylabel('Scores')
    senlinplt[1].set_ylabel('Scores')
    mmslogplt[1].set_ylabel('Scores')
    rmslogplt[1].set_ylabel('Scores')

    rmslinplt[1].set_title('wide H mass Linear RMS',  loc='right', fontsize =18 )
    senlinplt[1].set_title('wide H mass Sensitivity', loc='right', fontsize =18 )
    mmslogplt[1].set_title('wide H mass MMS',         loc='right', fontsize =18 )
    rmslogplt[1].set_title('wide H mass Log RMS',     loc='right', fontsize =18 )

    rmslinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslinplt[1].transAxes)
    senlinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=senlinplt[1].transAxes)
    mmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=mmslogplt[1].transAxes)
    rmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslogplt[1].transAxes)

    rmslinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslinplt[1].transAxes)
    senlinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=senlinplt[1].transAxes)
    mmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=mmslogplt[1].transAxes)
    rmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslogplt[1].transAxes)

    xtickslabels = ['0-80', '80-95', '95-110', '110-135', '135-180', '180<']

    rmslinplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    senlinplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    mmslogplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    rmslogplt[1].set_xticks([0,1,2,3,4,5], xtickslabels)
    
    rmslinplt[0].savefig(f'{plotdir}/lossmodes_overlay_rmslin_wideH.pdf', bbox_inches='tight')
    senlinplt[0].savefig(f'{plotdir}/lossmodes_overlay_senlin_wideH.pdf', bbox_inches='tight')
    mmslogplt[0].savefig(f'{plotdir}/lossmodes_overlay_mmslog_wideH.pdf', bbox_inches='tight')
    rmslogplt[0].savefig(f'{plotdir}/lossmodes_overlay_rmslog_wideH.pdf', bbox_inches='tight')
        
plt.close('all')

if runEpoch:
    mods = ['mass', 'logMass']
    lms = [0,3]
    vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no' ]
    epochs = [8,16,24,32,40]
    
    for mod in mods:
        for lm in lms:
            rmslinplt = plt.subplots()
            senlinplt = plt.subplots()
            mmslogplt = plt.subplots()
            rmslogplt = plt.subplots()

            for epoch in epochs: 
                rmslindifflist = []
                senlindifflist = []
                mmslogdifflist = []
                rmslogdifflist = []


                fn = f'predict/testepoch/predict_a1_calc_{mod}_regr_loss{lm}_epoch{epoch}.root'
                g = uproot.open(fn)['Events']
                df = pd.DataFrame()
                
                for v in vs:
                    df[v] = np.array(g[v].array())
        
                df = df[df['label_H_aa_bbbb']==1]

                ## convert output and target back to mass value so it makes sense whne plotting distribution
                if 'logMass' == mod:
                    df['output'] = np.exp(df['output'])
                    df['target_mass'] = np.exp(df['target_mass'])
        
                ## calculate ratios and resolutions
                df = df[df['output'] >= 0 ]
                
                df['ratio'] = np.divide(df['output'] , df['target_mass'])
                df['log2ratio'] = np.clip(np.log2(df['ratio']), a_min=-2, a_max=2)

                dftest = df[df['event_no']%2==1]
                dftrain = df[df['event_no']%2==0]
                   
                x_massrange_tick = []
                for massrange in zip(a_massranges[:-1], a_massranges[1:]):
                    clippeddftrain = dftrain[(dftrain['target_mass'] > massrange[0]) & (dftrain['target_mass'] <= massrange[1])]
                    clippeddftest = dftest[(dftest['target_mass'] > massrange[0]) & (dftest['target_mass'] <= massrange[1])] 
                    ## calculatate the scores for each mass range and append them to corresponding list
                    rmslindifflist.append(np.std(clippeddftest['ratio'])-np.std(clippeddftrain['ratio']))
                    mmslogdifflist.append(np.mean(clippeddftest['log2ratio'])-np.mean(clippeddftrain['log2ratio']))
                    rmslogdifflist.append(np.std(clippeddftest['log2ratio'])-np.std(clippeddftrain['log2ratio']))
                      
                    s_hist, s_edge = np.histogram(clippeddftest['ratio'], bins=101, range=(0,2.02))
                    binwidth = s_edge[1]-s_edge[0]
                    sensitivitytest = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                    s_hist, s_edge = np.histogram(clippeddftrain['ratio'], bins=101, range=(0,2.02))
                    binwidth = s_edge[1]-s_edge[0]
                    sensitivitytrain = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
                    senlindifflist.append(sensitivitytest-sensitivitytrain)
            
                ## plot this mod's scores on each plot
                rmslinplt[1].plot(range(len(rmslindifflist)), rmslindifflist, label=f'epoch{epoch}')
                senlinplt[1].plot(range(len(rmslindifflist)), senlindifflist, label=f'epoch{epoch}')
                mmslogplt[1].plot(range(len(rmslindifflist)), mmslogdifflist, label=f'epoch{epoch}')
                rmslogplt[1].plot(range(len(rmslindifflist)), rmslogdifflist, label=f'epoch{epoch}')
    
    
            ## styling of plot
            rmslinplt[1].legend()
            senlinplt[1].legend()
            mmslogplt[1].legend()
            rmslogplt[1].legend()
        
            rmslinplt[1].set_xlabel('Mass of Higgs [GeV]')
            senlinplt[1].set_xlabel('Mass of Higgs [GeV]')
            mmslogplt[1].set_xlabel('Mass of Higgs [GeV]')
            rmslogplt[1].set_xlabel('Mass of Higgs [GeV]')
        
            rmslinplt[1].set_ylabel('Difference in Scores')
            senlinplt[1].set_ylabel('Difference in Scores')
            mmslogplt[1].set_ylabel('Difference in Scores')
            rmslogplt[1].set_ylabel('Difference in Scores')
        
            rmslinplt[1].set_title('wide H mass Linear RMS',  loc='right', fontsize =18 )
            senlinplt[1].set_title('wide H mass Sensitivity', loc='right', fontsize =18 )
            mmslogplt[1].set_title('wide H mass MMS',         loc='right', fontsize =18 )
            rmslogplt[1].set_title('wide H mass Log RMS',     loc='right', fontsize =18 )
        
            rmslinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslinplt[1].transAxes)
            senlinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=senlinplt[1].transAxes)
            mmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=mmslogplt[1].transAxes)
            rmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslogplt[1].transAxes)
        
            rmslinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslinplt[1].transAxes)
            senlinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=senlinplt[1].transAxes)
            mmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=mmslogplt[1].transAxes)
            rmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslogplt[1].transAxes)
        
            #xtickslabels = 
        
            #rmslinplt[1].set_xticks([0,1,2,3,4,5, 6], xtickslabels)
            #senlinplt[1].set_xticks([0,1,2,3,4,5, 6], xtickslabels)
            #mmslogplt[1].set_xticks([0,1,2,3,4,5, 6], xtickslabels)
            #rmslogplt[1].set_xticks([0,1,2,3,4,5, 6], xtickslabels)
            
            rmslinplt[0].savefig(f'{plotdir}/epochs_overlay_rmslin_{mod}_loss{lm}.pdf', bbox_inches='tight')
            senlinplt[0].savefig(f'{plotdir}/epochs_overlay_senlin_{mod}_loss{lm}.pdf', bbox_inches='tight')
            mmslogplt[0].savefig(f'{plotdir}/epochs_overlay_mmslog_{mod}_loss{lm}.pdf', bbox_inches='tight')
            rmslogplt[0].savefig(f'{plotdir}/epochs_overlay_rmslog_{mod}_loss{lm}.pdf', bbox_inches='tight')
            plt.close('all')
        
        


if run34b:

    print('hello hello')
    mods = ['mass','logMass']
    loss_modes = [0, 3]
    vs = ['output', 'target_mass', 'label_H_aa_bbbb', 'fj_mass', 'event_no', 'fj_gen_H_aa_bbbb_num_b_AK8', 'fj_nbHadrons']
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

    for mod in mods:
        for lm in loss_modes:
            for num_genb in [3,4]:
                rmslinplt = plt.subplots() 
                senlinplt = plt.subplots() 
                rmslogplt = plt.subplots() 
                mmslogplt = plt.subplots()

                if num_genb == 3: num_bs = [3,34]
                if num_genb == 4: num_bs = [4,34]
                for num_b in num_bs: 
                    rmslinlist = []
                    senlinlist = []
                    mmsloglist = []
                    rmsloglist = []
                    
                    for mass in massranges:            
                        fn = f'predict/34b/predict_h125_a1_M{mass}_{mod}_{num_b}b_lm{lm}_regr.root'
                        df = pd.DataFrame()
                        f = uproot.open(fn)
                        g = f['Events']
    
                        for v in vs: 
                            df[v] = np.array(g[v].array())
                        if mod == 'logMass':
                            df['output'] = np.exp(df['output'])
                            df['target_mass'] = np.exp(df['target_mass'])
                        df = df[(df['output'] > 0) & (df['target_mass'] > 0)]
                        if num_genb == 3:
                            df = df[(df['fj_gen_H_aa_bbbb_num_b_AK8']==3) & (df['fj_nbHadrons']==3)]
                        elif num_genb == 4:
                            df = df[(df['fj_gen_H_aa_bbbb_num_b_AK8']>=4) & (df['fj_nbHadrons']>=4)]
                        else:
                            print('something went wrong in 34b')
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
                         
                    floatmassranges= [float(x) for x in massranges]
                    
                    ## plot this mod's scores on each plot
                    rmslinplt[1].plot(massranges, rmslinlist, label=f'{num_b}')
                    senlinplt[1].plot(massranges, senlinlist, label=f'{num_b}')
                    mmslogplt[1].plot(massranges, mmsloglist, label=f'{num_b}')
                    rmslogplt[1].plot(massranges, rmsloglist, label=f'{num_b}')
                ## styling of plot
                rmslinplt[1].legend()
                senlinplt[1].legend()
                mmslogplt[1].legend()
                rmslogplt[1].legend()
            
                rmslinplt[1].set_xlabel('Mass of pseudoscalar [GeV]')
                senlinplt[1].set_xlabel('Mass of pseudoscalar [GeV]')
                mmslogplt[1].set_xlabel('Mass of pseudoscalar [GeV]')
                rmslogplt[1].set_xlabel('Mass of pseudoscalar [GeV]')
        
                rmslinplt[1].set_ylabel('Scores')
                senlinplt[1].set_ylabel('Scores')
                mmslogplt[1].set_ylabel('Scores')
                rmslogplt[1].set_ylabel('Scores')
                
                rmslinplt[1].set_title(f'{mod} loss{lm} {num_genb} gen b $a$ mass Linear RMS',  loc='left', fontsize =18 , pad=21)
                senlinplt[1].set_title(f'{mod} loss{lm} {num_genb} gen b $a$ mass Sensitivity', loc='left', fontsize =18 , pad=21)
                mmslogplt[1].set_title(f'{mod} loss{lm} {num_genb} gen b $a$ mass MMS',         loc='left', fontsize =18 , pad=21)
                rmslogplt[1].set_title(f'{mod} loss{lm} {num_genb} gen b $a$ mass Log RMS',     loc='left', fontsize =18 , pad=21)
        
                rmslinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslinplt[1].transAxes)
                senlinplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=senlinplt[1].transAxes)
                mmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=mmslogplt[1].transAxes)
                rmslogplt[1].text(0, 1.02, "CMS", fontsize=18, weight='bold', transform=rmslogplt[1].transAxes)
        
                rmslinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslinplt[1].transAxes)
                senlinplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=senlinplt[1].transAxes)
                mmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=mmslogplt[1].transAxes)
                rmslogplt[1].text(0.15, 1.02, "Monte Carlo", fontsize=14, fontstyle='italic', transform=rmslogplt[1].transAxes)
            
                rmslinplt[0].savefig(f'{plotdir}/genb{num_genb}_overlay_rmslin_{mod}_loss{lm}.pdf', bbox_inches='tight')
                senlinplt[0].savefig(f'{plotdir}/genb{num_genb}_overlay_senlin_{mod}_loss{lm}.pdf', bbox_inches='tight')
                mmslogplt[0].savefig(f'{plotdir}/genb{num_genb}_overlay_mmslog_{mod}_loss{lm}.pdf', bbox_inches='tight')
                rmslogplt[0].savefig(f'{plotdir}/genb{num_genb}_overlay_rmslog_{mod}_loss{lm}.pdf', bbox_inches='tight')
            
                plt.close('all')
