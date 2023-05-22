import uproot 
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
import pandas as pd
import matplotlib as mpl
import os
import itertools 
from matplotlib.colors import LogNorm



def main():

    ## new training si this is so spaghetti. this will only do sensitivity and resolution
    fns = ['predict/predict_wide_H_genHmassOverfj_mass.root','predict/predict_wide_H_logGenHmassOverfj_mass.root'] 


    f1 = uproot.open(fns[0])
    f2 = uproot.open(fns[1])
    g1 = f1['Events']
    g2 = f2['Events'] 

    for fn in fns:
        f = uproot.open(fn)
        g = f['Events']
        df = pd.DataFrame()
        df['output'] = g['output'].array()
        df['target'] = g['target_mass'].array() 
        df['fj_mass'] = g['fj_mass'].array()
        df['genH_mass'] = g['fj_gen_H_aa_bbbb_mass_H'].array()
        df['label'] = g['label_H_aa_bbbb'].array()

        if 'predict/predict_wide_H_logGenHmassOverfj_mass.root' == fn:
            df['output'] = df['fj_mass']*np.exp(df['output'])
            df['target'] = df['fj_mass']*np.exp(df['target'])
        else:
            df['output'] = df['fj_mass']*df['output']
            df['target'] = df['fj_mass']*df['target']
        
        df = df[(df['label'] ==1) & (df['output']>0) & (df['target'] > 0)]

        dis_fig, dis_ax = plt.subplots()
        res_fig, res_ax = plt.subplots()
        sen_fig, sen_ax = plt.subplots()
        
        ## distribution
        name = fn[fn.find('wide_H')+7:].replace('.root', '')
        hist_kwargs = {'bins':50, 'histtype':'step', 'label':name}
        dis_ax.hist(df['output'], **hist_kwargs)
        
        ## reosolution
        ratio = np.array(np.divide(df['output'], df['target']))
        res_ax.hist(np.clip(np.log2(ratio), a_min=-4, a_max=4), **hist_kwargs)
        
        ## sensitivity
        s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
        binwidth = s_edge[1]-s_edge[0]
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        sen_ax.step( s_edge[:-1] , s_hist, label=name)

        massrange = [0,80,95,110,135,180,99999]
        mmsList = []
        rmsList = []
        rmsList2 = []
        sensitivityList = []
        for lower, upper in pairwise(massrange):
            print(fn)
            print(f'{lower} - {upper}')

            print('df.shape: ', df.iloc[:20])

            tmpdf = df[(df['target'] > lower) & (df['target'] <= upper)]
            tmpdf= tmpdf[ (tmpdf['output'] > 0 ) & (tmpdf['target'] > 0)]

            
            
            print('tmpdf.shape after cut: ', tmpdf.shape)


            ratio = np.divide(tmpdf['output'], tmpdf['target']).to_numpy()
            mms = np.mean(np.log2(ratio))
            rms = np.std(np.log2(ratio))
            mmsList.append(f'{mms:.4E}')
            rmsList.append(f'{rms:.4E}')

            ## sensitivity
            s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            rm2 = np.std(ratio, where=ratio<=2)
            sensitivityList.append(f'{sensitivity:.4E}')
            rmsList2.append(f'{rm2:.4E}')
        

        ## printing out csv info
        print('======================================')
        print(fn)
        print(',mass_range,MMS_logRatio,RMS_logRatio,RMS_ratio,sensitivity2')
        number = 0
        for m, rmslog, mmslog, rmslin, sens in zip(pairwise(massrange), rmsList, mmsList, rmsList2, sensitivityList):
            print(f'{number},"{m}",{mmslog},{rmslog},{rmslin},{sens}')
            number+=1

        ## add tables to resolution and sensitivity
        labels = [x for x in pairwise(massrange)]
        res_ax.table(
            colLabels=labels,
            rowLabels=['MMS', 'RMS'],
            cellText=[mmsList, rmsList],
            bbox=[0.1, -0.3, 0.9, 0.2]
        )
        sen_ax.table(
            colLabels=labels,
            rowLabels=['sensitivity^2', 'RMS'],
            cellText=[sensitivityList, rmsList2],
            bbox=[0.1, -0.3, 0.9, 0.2]
        )
        res_ax.legend()
        sen_ax.legend()
        dis_ax.set_title(f'distribution {name}')
        res_ax.set_title(f'resolution {name}')
        sen_ax.set_title(f'sensitivity {name}')
        
        dis_fig.savefig(f'plots/dist_{name}.png', bbox_inches='tight')
        res_fig.savefig(f'plots/resolution_{name}.png', bbox_inches='tight')
        sen_fig.savefig(f'plots/sensitivity_{name}.png', bbox_inches='tight')

    plt.close('all')
            

    central = False
    wideH = True

    parts = ['H_calc', 'a1', 'a2']
    mods = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT', 'ptOverMass']
    parts.remove('a2')
    mods.remove('1OverMass')
    ## centrally produced 
    mp1 = [12,20,30,40,50,60]
    mp2 = [15,25,35,45,55]
    if central:
        for part in parts:
            for mod in mods:
                fn1 = [f'predict/predict_central_{part}_M{masspoint}_{mod}_regr.root' for masspoint in mp1]
                fn2 = [f'predict/predict_central_{part}_M{masspoint}_{mod}_regr.root' for masspoint in mp2]
                titles = [f'predict {part} {mod} (even)', f'target {part} {mod} (even)', 
                          f'resolution {part} {mod} (even)', f'sig/sqrt(bg) {part} {mod} (even)']
                plotnames = (f'predict_{part}_{mod}_even', f'target_{part}_{mod}_even', 
                             f'resolution_{part}_{mod}_even', f'sensitivity_{part}_{mod}_even')
                labels = [f'M-{mp}' for mp in mp1]
                nbins = [70, 70]
                binranges = [(0, 70), (-2, 2)]
                make1DDist(fn1, titles, plotnames, labels, nbins, binranges)
                titles = [f'predict {part} {mod} (odd)', f'target {part} {mod} (odd)', 
                          f'resolution {part} {mod} (odd)', f'sig/sqrt(bg) {part} {mod} (odd)']
                plotnames = (f'predict_{part}_{mod}_odd', f'target_{part}_{mod}_odd', 
                             f'resolution_{part}_{mod}_odd', f'sensitivity_{part}_{mod}_odd')
                labels = [f'M-{mp}' for mp in mp2]
                make1DDist(fn2, titles, plotnames, labels, nbins, binranges)
    
    
        ## calculate the RMS values again to be saved to a csv 
        df = pd.DataFrame()
        for part in [0]:#parts:
            for mod in mods:
                fn1 = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mp1]
                fn2 = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mp2]
                rms, mms, rms2, sensitivity, masspoints = calc_RMS_MMS(mp1+mp2, fn1+fn2)
                df['masspoints'] = masspoints 
                df[f'rms_{mod}'] = rms
                df[f'mms_{mod}'] = mms
                df[f'rms2_{mod}'] = rms2
                df[f'sensitivity_{mod}'] = sensitivity
    
        df.to_csv('csv/RMS_MMS_masspoints.csv')

    ## wide H 
    ptpoints = [150, 250, 350]
    distranges = [(0,700), (0,300), (0,150)]
    binrangesList = [[x, (-4,4)] for x in distranges]
    parts = ['H_calc', 'a1', 'a2']
    # predict_wide_H_calc_pt150_1OverMass_regr.root
    ## TODO figure this shit out. is it all pt point in 1 plot? maybe 
    if wideH:
        for mod in mods: #for ptpnt, binrange in zip(ptpoints, binranges):
            for part, binranges in zip(['H_calc'], [[(0,700), (-4,4)]]): #zip(parts, binrangesList):
                fns = [f'predict/predict_wide_{part}_pt{ptpnt}_{mod}_regr.root' for ptpnt in ptpoints]
                titles = [f'predict {part} {mod} wideH', f'target {part} {mod} wideH',
                          f'resolution {part} {mod} wideH', f'sensitivity {part} {mod} wideH']
                plotnames = [f'predict_{part}_{mod}_wideH', f'target_{part}_{mod}_wideH',
                             f'resolution_{part}_{mod}_wideH', f'sensitivity_{part}_{mod}_wideH']
                labels = [f'pt{x}' for x in ptpoints]
                nbins = [70, 70]
                make1DDist(fns, titles, plotnames, labels, nbins, binranges)


        ## calculate the RMS values again to be saved to a csv 
        df = pd.DataFrame()
        for part in ['H_calc']:#parts:
            for mod in mods:
                fns = [f'predict/predict_wide_{part}_pt{ptpnt}_{mod}_regr.root' for ptpnt in ptpoints]
                rms, mms, rms2, sensitivity, masspoints = calc_RMS_MMS(ptpoints, fns)
                df['pt'] = masspoints 
                df[f'rms_{mod}'] = rms
                df[f'mms_{mod}'] = mms
                df[f'rms2_{mod}'] = rms2
                df[f'sensitivity_{mod}'] = sensitivity
    
        df.to_csv('csv/RMS_MMS_masspoints.csv')


        ## 2d correlations 
        for mod in mods:
            for part, distrange in zip(parts, distranges):
                fns = [f'predict/predict_wide_{part}_pt{ptpnt}_{mod}_regr.root' for ptpnt in ptpoints]
                fs = [uproot.open(fn) for fn in fns]
                gs = [f['Events'] for f in fs]
                df = pd.DataFrame()
                ## concatenate each in df 
                output = np.concatenate([g['output'].array() for g in gs])
                target = np.concatenate([g['target_mass'].array() for g in gs])
                pt = np.concatenate([g['fj_pt'].array() for g in gs])
                label_H_aa_bbbb = np.concatenate([g['label_H_aa_bbbb'].array() for g in gs])
                df['output'] = output
                df['target'] = target
                df['pt'] = pt
                df['label_H_aa_bbbb'] = label_H_aa_bbbb
                df = df[df['label_H_aa_bbbb'] ==1]
                output = df['output']
                target = df['target']
                pt = df['pt']
                output, target, binrange = returnToMass(output, target, pt, fns[0])
                fig, ax = plt.subplots()
                hist = ax.hist2d(output, target, bins=70, range=(distrange, distrange),norm=mpl.colors.LogNorm())
                fig.colorbar(hist[3], ax=ax)
                ax.set_title(f'2D Correlation {part} {mod}')
                ax.set_xlabel('prediction')
                ax.set_ylabel('target')
                plt.savefig(f'plots/correlation2D_{part}_{mod}.png')
                plt.close()

        

def make1DDist(fl, titles, plotnames, labels, nbins, binranges, enable2D=False):
    ## creates the 1d preedict, target and mass resolution plots
    ## all files in fl will go on one plot 
    ## usage
    ## fl : list of files to stack on the same plot
    ## titles : [title of predict 1d, title of target 1d, title of ratio, title of sig/sqrt(bg)]
    ## plotnames : [plotname of predict 1d, plotname of target 1d, plotname of ratio, plotname of sig/sqrt(bg)] (without file extension. This assumes the plots will go into the /plots/ directory
    ## labels : list of what goes into the legend for each filename (same length as fl)
    ## nbins : [nbins for predict and target 1d, nbins for ratio 1d]
    ## binranges : [binrange for predict and target 1d, binrange for ratio]
    ## for furure si who is confused and dumb:: The files (and other stff) that go in here are same mod different masspoints
    
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots( )
    fig4, ax4 = plt.subplots()

    ## mass vars that are in the ntuple already
    othervars = ['fj_mass', 'fj_sdmass', 'fj_corrsdmass', 'fj_sdmass_fromsubjets', 'pfParticleNetMassRegressionJetTags_mass']
    massdists = {x:plt.subplots() for x in othervars}
    massreses = {x:plt.subplots() for x in othervars}
    masssens = {x:plt.subplots() for x in othervars}
    fig_massres, ax_massres = plt.subplots()
    fig_masssen, ax_masssen = plt.subplots()

    mmsList = []
    rmsList = []

    sensitivityList = []
    rmsList2 = []

    df = pd.DataFrame()

    for fn, label in zip(fl, labels):
        f = uproot.open(fn)
        g = f['Events']
        output = g['output'].array()
        target = g['target_mass'].array()
        pt = g['fj_pt'].array()
        output, target, binrange = returnToMass(output, target, pt, fn, binranges[0])
        dfsub = pd.DataFrame()
        dfsub['output'] = output#g['output'].array()
        dfsub['target'] = target#g['target_mass'].array()
        dfsub['label_H_aa_bbbb'] = g['label_H_aa_bbbb'].array()
        dfsub['pt'] = pt#g['fj_pt'].array()
        dfsub['fj_mass'] = g['fj_mass'].array()
        dfsub['fj_sdmass'] = g['fj_sdmass'].array()
        dfsub['fj_corrsdmass'] = g['fj_corrsdmass'].array()
        dfsub['fj_sdmass_fromsubjets'] = g['fj_sdmass_fromsubjets'].array()
        dfsub['pfParticleNetMassRegressionJetTags_mass'] = g['pfParticleNetMassRegressionJetTags_mass'].array()
        dfsub = dfsub[dfsub['label_H_aa_bbbb']==1]
        df = pd.concat([df, dfsub])
        output = np.array(dfsub['output'])
        target = np.array(dfsub['target'])
        pt = np.array(dfsub['pt'])

        ## modify output and target back to mass if needed 
        #output, target, binrange = returnToMass(output, target, pt, fn, binranges[0])
        
        ## 1D dists 
        histdict = {'bins':nbins[0], 'range':binrange, 'histtype':'step', 'label':f'{label} {len(output)}'}
        ax.hist(np.clip(output, a_min=binrange[0], a_max=binrange[1]), **histdict)
        ax2.hist(np.clip(target, a_min=binrange[0], a_max=binrange[1]),**histdict)
        histdict['bins'] = nbins[1]
        histdict['range'] = binranges[1]
        condition = (output > 0 ) & (target > 0) ## exluce where target or output is zero because breaks log
        ratio = np.divide(output[condition], target[condition] )
        
        ## resolution plot
        ax3.hist(np.clip(np.log2(ratio, where=ratio>0), a_min=binranges[1][0], a_max=binranges[1][1]), **histdict)
        
        ## s/sqrt(B)    (sensitivity)
        ## Plot with 101 bins from 0 to 2.02, using the last bin as “overflow” for all events with [pred / targ] > 2.0
        ## Sum the squares of the bin values for the first 100 bins (excluding the overflow bin), multiply by 100 (the number of bins 
        ## summed), and divide by the square of the total integral (including the overflow bin).
        s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
        binwidth = s_edge[1]-s_edge[0]
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        ax4.step( s_edge[:-1] , s_hist, label=f'{label}')
        
        ## for filling the sumsqr table
        sensitivityList.append(f'{sensitivity:.4f}')
        rm2 = np.std(ratio, where=ratio<=2)
        rmsList2.append(f'{rm2:.4f}')

        ## fill table for the mms and rms
        mms = np.mean(np.log2(ratio, where=ratio>0))
        rms = np.std(np.log2(ratio, where=ratio>0))
        mmsList.append(f'{mms:.4f}')
        rmsList.append(f'{rms:.4f}')

        ## mass distributions and reosolutions 
        othervars = ['fj_mass', 'fj_sdmass', 'fj_corrsdmass', 'fj_sdmass_fromsubjets', 'pfParticleNetMassRegressionJetTags_mass']
        histdict['range'] = None
        histdict['label'] = label
        for massvar in othervars:
            #massdists = {x:plt.subplots() for x in othervars}
            #massreses = {x:plt.subplots() for xin othervars}
            #masssens = {x:plt.subplots() for x in othervars}
            ## distribution
            massdists[massvar][1].hist(dfsub[massvar], **histdict)
            massdists[massvar][1].set_title(massvar)            

            ## ratio 
            non_pNetMass = dfsub[massvar]
            condition = (non_pNetMass > 0 ) & (target > 0) ## exluce where target or output is zero because breaks log                                                                          
            ratio = np.divide(non_pNetMass[condition], target[condition] )
            massreses[massvar][1].hist(np.clip(np.log2(ratio, where=ratio>0), a_min=binranges[1][0], a_max=binranges[1][1]), **histdict)
            massreses[massvar][1].set_title(f'Resolution {massvar}/target')
            
            ## sensitivity
            s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            masssens[massvar][1].step( s_edge[:-1] , s_hist, label=f'{label}')
            masssens[massvar][1].set_title(f'Sensitivity {massvar}/target')                
            

    ## create the RMS / MMS per mass ranges 
    massrange = [0,80,95,110,135,180,99999]
    labels = []
    mmsList = []
    rmsList = []
    rmsList2 = []
    sensitivityList = []
    massvars_mmsList = {massvar:[] for massvar in othervars}
    massvars_rmsList = {massvar:[] for massvar in othervars}
    massvars_rmsList2 = {massvar:[] for massvar in othervars}
    massvars_sensitivityList = {massvar:[] for massvar in othervars}
    for lower, upper in pairwise(massrange):
        labels.append(f'{lower} - {upper}')
        tmpdf = df[(df['target'] > lower) & (df['target'] <= upper)]
        ## resolution
        tmpdf= tmpdf[ (tmpdf['output'] > 0 ) & (tmpdf['target'] > 0)]
        ratio = np.divide(tmpdf['output'], tmpdf['target']).to_numpy()
        mms = np.mean(np.log2(ratio, where=ratio>0))
        rms = np.std(np.log2(ratio, where=ratio>0))
        mmsList.append(f'{mms:.4E}')
        rmsList.append(f'{rms:.4E}')


        ## sensitivity 
        s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
        binwidth = s_edge[1]-s_edge[0]
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        rm2 = np.std(ratio, where=ratio<=2)
        sensitivityList.append(f'{sensitivity:.4E}')
        rmsList2.append(f'{rm2:.4E}')

        
        for massvar in othervars:
            ## resolution for non pNet masses 
            condition = (tmpdf[massvar] > 0) & (tmpdf['target'] > 0)
            ratio = np.divide(tmpdf[massvar][condition], tmpdf['target'][condition]).to_numpy()
            mms = np.mean(np.log2(ratio, where=ratio>0))
            rms = np.std(np.log2(ratio, where=ratio>0))

            massvars_mmsList[massvar].append(f'{mms:.4E}')
            massvars_rmsList[massvar].append(f'{rms:.4E}')

            ## sensitivity for non pNet masses 
            s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2, where=ratio>0), bins=101, range=(0,2.02))
            binwidth = s_edge[1]-s_edge[0]
            sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
            rm2 = np.std(ratio, where=ratio<=2)
            massvars_rmsList2[massvar].append(f'{rm2:.4E}')
            massvars_sensitivityList[massvar].append(f'{sensitivity:.4E}')
               


            
    for massvar in othervars:
        masssens[massvar][1].table(
            colLabels=labels,
            rowLabels=['sensitivity^2', 'RMS'],
            cellText=[massvars_sensitivityList[massvar], massvars_rmsList2[massvar]],
            bbox=[0.1, -0.3, 0.9, 0.2]
        )
        massreses[massvar][1].table(
            colLabels=labels,
            rowLabels=['MMS', 'RMS'],
            cellText=[massvars_mmsList[massvar], massvars_rmsList[massvar]],
            bbox=[0.1,-0.3, 0.9, 0.2]
        )
        massdists[massvar][1].legend()
        massreses[massvar][1].legend()
        masssens[massvar][1].legend()
    
        massdists[massvar][0].savefig(f'plots/dist_{massvar}.png', bbox_inches='tight')
        massreses[massvar][0].savefig(f'plots/resolution_{massvar}.png', bbox_inches='tight')
        masssens[massvar][0].savefig(f'plots/sensitivity_{massvar}.png', bbox_inches='tight')

        ## making the csv 
        csvdf = pd.DataFrame() 
        csvdf['mass_range'] = [x for x in pairwise(massrange)]
        csvdf['MMS_logRatio'] = massvars_mmsList[massvar]
        csvdf['RMS_logRatio'] = massvars_rmsList[massvar]
        csvdf['RMS_ratio'] = massvars_rmsList2[massvar]
        csvdf['sensitivity2'] = massvars_sensitivityList[massvar]
        csvdf.to_csv(f'csv/trend_{massvar}.csv')
    
    ax3.table(
        colLabels=labels,
        rowLabels=['MMS', 'RMS'],
        cellText = [mmsList, rmsList],
        bbox=[0.1, -0.3, 0.9, 0.2]
    )

    ax4.table(
        colLabels=labels,
        rowLabels=['sensitivity^2', 'RMS'],
        cellText = [sensitivityList, rmsList2],
        bbox=[0.1, -0.3, 0.9, 0.2]
    )

    print('=========================================================')
    print(fn)
    print(',mass_range,MMS_logRatio,RMS_logRatio,RMS_ratio,sensitivity2')
    number = 0
    for m, rmslog, mmslog, rmslin, sens in zip(pairwise(massrange), rmsList, mmsList, rmsList2, sensitivityList):
        
        print(f'{number},"{m}",{mmslog},{rmslog},{rmslin},{sens}')
        number+=1
        
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax.set_title(titles[0])
    ax2.set_title(titles[1])
    ax3.set_title(titles[2])
    ax4.set_title(titles[3])
    fig.savefig(f'plots/{plotnames[0]}.png',bbox_inches='tight')
    fig2.savefig(f'plots/{plotnames[1]}.png', bbox_inches='tight')
    fig3.savefig(f'plots/{plotnames[2]}.png', bbox_inches='tight')
    fig4.savefig(f'plots/{plotnames[3]}.png', bbox_inches='tight')
    plt.close('all')

        
def returnToMass(output, target, pt, fn, binrange=None):

    if 'logMassOverPT' in fn:
        output = np.exp(output)*pt
        target = np.exp(target)*pt
    elif ('logMass' in fn) and ('OverPT' not in fn):
        output = np.exp(output)
        target = np.exp(target)
    elif '1OverMass' in fn:
        output = 1/output
        target = 1/target
        #binrange= [0,70]
    elif 'massOverPT' in fn:
        output = output*pt
        target = target*pt
        output[output < 0] = 0
        #binrange = (-2,2)
    elif 'ptOverMass' in fn:
        output = pt/output
        target = pt/target

    # output[~np.isfinite(output)] = 0
    return output, target, binrange


## function that calculates rms and mms for each file and returns the value for plotting 
## i think it's better to receive the gen mass for each mod type and return a array of rms, mms, and mass point to use for plotting 
## it will also calculate s/sqrt(b) too because i suck
def calc_RMS_MMS(masspoints, filenames, log2=False):
    df = pd.DataFrame(masspoints, columns=['masspoints'])
    df['filenames'] = filenames
    df.sort_values(by='masspoints', ignore_index=True, inplace=True)
    mmsList = []
    rmsList = []


    rmsList2 = []
    sensitivityList = []

    #s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
    #binwidth = s_edge[1]-s_edge[0]
    #sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
    
    for fn in df['filenames']: 
        f = uproot.open(fn)
        g = f['Events']
        output = g['output'].array()
        target = g['target_mass'].array()
        pt = g['fj_pt'].array()

        output, target, binrange = returnToMass(output, target, pt, fn)
        where = {'where':output>0, 'out':np.zeros_like(output)}
        
        ratio = output/target

        mms = np.mean(np.log2(ratio, **where))
        rms = np.std(np.log2(ratio, **where))
        mmsList.append(f'{mms:.4f}')
        rmsList.append(f'{rms:.4f}')
        
        s_hist, _ = np.histogram(np.clip(ratio, a_min=0, a_max=2), bins=101, range=(0,2.02))
        sensitivity = np.sum(s_hist[:-1]*s_hist[:-1])*100/(np.square(np.sum(s_hist)))
        rms2 = np.std(ratio, where=ratio<=2)

        rmsList2.append(f'{rms2:.4f}')
        sensitivityList.append(f'{sensitivity:.4f}')

    return rmsList, mmsList, rmsList2, sensitivityList, df['masspoints'].values.tolist()

def pairwise(iterable):
    "[0,1,2,3...] -> (0,1), (1,2), (2,3)..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

if __name__ == "__main__":
    main()
