import uproot 
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
import pandas as pd
import matplotlib as mpl
import os
from matplotlib.colors import LogNorm
#import pandas 


def main():
    central = False
    wideH = True

    parts = ['H_calc', 'a1', 'a2']
    mods = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT', 'ptOverMass']
    parts.remove('a2')
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
    
        df.to_csv('RMS_MMS_masspoints.csv')

    ## wide H 
    ptpoints = [150, 250, 350]
    distranges = [(0,700), (0,300), (0,150)]
    binrangesList = [[x, (-4,4)] for x in distranges]
    parts = ['H_calc', 'a1', 'a2']
    # predict_wide_H_calc_pt150_1OverMass_regr.root
    ## TODO figure this shit out. is it all pt point in 1 plot? maybe 
    if wideH:
        for mod in mods: #for ptpnt, binrange in zip(ptpoints, binranges):
            for part, binranges in zip(parts, binrangesList):
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
        for part in parts:
            for mod in mods:
                fns = [f'predict/predict_wide_{part}_pt{ptpnt}_{mod}_regr.root' for ptpnt in ptpoints]
                rms, mms, rms2, sensitivity, masspoints = calc_RMS_MMS(ptpoints, fns)
                df['pt'] = masspoints 
                df[f'rms_{mod}'] = rms
                df[f'mms_{mod}'] = mms
                df[f'rms2_{mod}'] = rms2
                df[f'sensitivity_{mod}'] = sensitivity
    
        df.to_csv('RMS_MMS_masspoints.csv')


        ## 2d correlations 
        for mod in mods:
            for part, distrange in zip(parts, distranges):
                fns = [f'predict/predict_wide_{part}_pt{ptpnt}_{mod}_regr.root' for ptpnt in ptpoints]
                fs = [uproot.open(fn) for fn in fns]
                gs = [f['Events'] for f in fs]
                output = np.concatenate([g['output'].array() for g in gs])
                target = np.concatenate([g['target_mass'].array() for g in gs])
                pt = np.concatenate([g['fj_pt'].array() for g in gs])
                output, target, binrange = returnToMass(output, target, pt, fns[0])
                fig, ax = plt.subplots()
                hist = ax.hist2d(output, target, bins=70, range=(distrange, distrange),norm=mpl.colors.LogNorm())
                fig.colorbar(hist[3], ax=ax)
                ax.set_title(f'2D Correlation {part} {mod}')
                ax.set_xlabel('prediction')
                ax.set_ylabel('target')
                plt.savefig(f'plots/correlation2D_{part}_{mod}.png')
                plt.close()


    import sys
    sys.exit()

    fns = [f'/home/chosila/Projects/weaver/output/predict_VarHMass_20000_{x}_regr.root' for x in ['H', 'a1', 'a2']]
    names = ['H_mass_regr','a1_mass_regr','a2_mass_regr']
    histranges = [[[122,128],[122,128]], [[10,60],[10,60]], [[10,60],[10,60]]]
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # 10 - 13.5, 13.5 - 17.5, 17.5 - 22.5, 22.5 - 27.5, etc. up to 62.5, then 10 GeV bins up to 102.5
    h1Dbins = [10, 13.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 72.5, 82.5, 92.5, 102.5]

    for fn, name, histrange in zip(fns, names, histranges):
        f = uproot.open(fn)
        g = f['Events']
        thresholds = np.linspace(0,1,300)
    
        if predType == 'regr':
            fig, ax = plt.subplots()
            output = g['output'].array()
            target = g['target_mass'].array()
            ax.set_xlabel('prediction')
            ax.set_ylabel('target_mass')
            ax.set_title(name)
            loss = metrics.mean_squared_error(target, output)
            hist = ax.hist2d(output, target, bins=20, norm=mpl.colors.LogNorm(), range=histrange)
            fig.colorbar(hist[3], ax=ax)
            ax.text(0.7, 0.85, f'MSE: {loss:.3f}', transform=ax.transAxes, bbox=props)
            plt.savefig(f'plots/{name}.png')
            plt.close()

    if predType == 'bin':
        for labelstr, scorestr, name in [('label_H_aa_bbbb', 'score_label_H_aa_bbbb', 'H_aa_bbbb'), ('label_H_aa_other', 'score_label_H_aa_other', 'H_aa_other'), ('sample_isQCD', 'score_sample_isQCD', 'is_QCD')]:
            df = pd.DataFrame()
            df['lab'] = np.array(g[labelstr].array())
            df['score'] = np.array(g[scorestr].array())
            df['mass'] = np.array(g['fj_mass'].array())
            df['pt'] = np.array(g['fj_pt'].array())
            df['eta'] = np.array(g['fj_eta'].array())
            df = df[(df['mass'] < 140) & (df['mass'] > 110)]
        
        
            labels = df['lab']#g[labelstr].array()
            scores = df['score']#g[scorestr].array()
            fpr, tpr, thresholds = metrics.roc_curve(labels, scores)#roc_curve(labels, scores, thresholds)
            auc = metrics.auc(fpr, tpr)
        
            ## roc curve
            print(f'auroc {auc}')
            fig, ax = plt.subplots()
            ax.set_title(name + ' & 110 < fj_mass < 140')
            ax.set_xlabel('fpr')
            ax.set_ylabel('tpr')
            ax.plot(fpr,tpr, label=f'auc = {auc:.4f}')
            ax.legend()
            plt.savefig(f'plots/{name}_auc_masslimit.png')
        
        
            ## efficiency vs pt at a certain threahold
            threshold = 0.99
            y, edges = np.histogram(df['pt'], bins=50)
            binCenter = getBinCenter(edges)
            tpArr = []
            tpfnArr = []
            
            for i,x in enumerate(edges[:-1]):
                edgepair = (edges[i], edges[i+1])
                cutdf = df[(df['pt'] > edges[i]) & (df['pt'] < edges[i+1])]
                y_pred = cutdf['score']
                y_pred[y_pred >= threshold] = 1
                y_pred[y_pred < threshold] = 0 
                y_true = cutdf['lab']
        
                fp = np.sum((y_pred == 1) & (y_true == 0))
                tp = np.sum((y_pred == 1) & (y_true == 1))
        
                fn = np.sum((y_pred == 0) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
        
                tpArr.append(tp)
                tpfnArr.append(tp+fn)
                #tprAtPt.append(tp / (tp + fn))
                           
            #tprAtPt = np.nan_to_num(np.array(tprAtPt), nan=0.0)
            # pt efficiency plot
            log = not ('bbbb' in name)
            
            makeEfficiencyPlot(f'{name} efficiency vs pt at threshold = {threshold}',
                               'pt', 'efficiency', binCenter, tpArr, tpfnArr,
                               f'{name}_ptEfficiency', log=log)
        
        

def makeEfficiencyPlot(title, xlabel, ylabel, binCenter, tpArr, tpfnArr, pltname, log):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    width = binCenter[1]-binCenter[0]
    ax1.bar(binCenter, tpfnArr, width, alpha=0.5, label='true positives false negatives')
    ax1.bar(binCenter, tpArr, width, alpha=0.5, label='true positives')
    if log: ax1.set_yscale('log')
    ax1.legend()
    plt.savefig(f'plots/{pltname}.png')
    plt.close(fig1)

def getBinCenter(edges):
    left = edges[:-1]
    right = edges[1:]
    return (left + right)/2
        

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

    mmsList = []
    rmsList = []

    sensitivityList = []
    rmsList2 = []

    for fn, label in zip(fl, labels):
        f = uproot.open(fn)
        g = f['Events']
        output = g['output'].array()
        target = g['target_mass'].array()
        pt = g['fj_pt'].array()
        
        ## modify output and target back to mass if needed 
        output, target, binrange = returnToMass(output, target, pt, fn, binranges[0])
        
        ## 1D dists 
        histdict = {'bins':nbins[0], 'range':binrange, 'histtype':'step', 'label':label}
        ax.hist(np.clip(output, a_min=binrange[0], a_max=binrange[1]), **histdict)
        ax2.hist(np.clip(target, a_min=binrange[0], a_max=binrange[1]),**histdict)
        histdict['bins'] = nbins[1]
        histdict['range'] = binranges[1]
        condition = (output !=0 ) | (target != 0) ## exluce where target or output is zero because breaks log
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


if __name__ == "__main__":
    main()
