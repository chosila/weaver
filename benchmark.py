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

    parts = ['H_calc', 'a1', 'a2']
    mods = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT', 'ptOverMass']
    
    ## centrally produced 
    mp1 = [12,20,30,40,50,60]
    mp2 = [15,25,35,45,55]
    for part in parts:
        for mod in mods:
            fn1 = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mp1]
            fn2 = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mp2]
            titles = [f'predict {part} {mod} (even)', f'target {part} {mod} (even)', 
                      f'resolution {part} {mod} (even)', f'sig/sqrt(bg) {part} {mod} (even)']
            plotnames = (f'predict_{part}_{mod}_even', f'target_{part}_{mod}_even', 
                         f'resolution_{part}_{mod}_even', f'sOverSqrtB_{part}_{mod}_even')
            labels = [f'M-{mp}' for mp in mp1]
            nbins = [70, 70]
            binranges = [(0, 70), (-2, 2)]
            make1DDist(fn1, titles, plotnames, labels, nbins, binranges)
            titles = [f'predict {part} {mod} (odd)', f'target {part} {mod} (odd)', 
                      f'resolution {part} {mod} (odd)', f'sig/sqrt(bg) {part} {mod} (odd)']
            plotnames = (f'predict_{part}_{mod}_odd', f'target_{part}_{mod}_odd', 
                         f'resolution_{part}_{mod}_odd', f'sOverSqrtB_{part}_{mod}_odd')
            labels = [f'M-{mp}' for mp in mp2]
            make1DDist(fn2, titles, plotnames, labels, nbins, binranges)


    ## wide H 
    ptpoints = [150, 250, 350]
    binranges = [[x, (-2,2)] for x in [(0,600), (0,300), (0,125)]]
    
    ## TODO figure this shit out. is it all pt point in 1 plot? maybe 
    for ptpnt, binrange in zip(ptpoints, binranges):
        fns = [f'predict/predict_{part}_pt{ptpnt}_mass_regr.root']
        #make1DDist(fn2, titles, plotnames, labels, nbins, binranges)

    import sys
    sys.exit()

    ## wide H miniaod
    ## predict_H_calc_pt150_logMass_regr.root
    ## predict/predict_a2_pt350_mass_regr.root
    labels = [150, 250, 350]
    histranges = [(0,600), (0,300), (0,125)]
    ## 2d hist of predict 
    for part, histrange in zip(parts, histranges):
        fig1, ax1 = plt.subplots()
        fns = [f'predict/predict_{part}_pt{mp}_mass_regr.root' for mp in labels]
        hist = 0
        for fn in fns:
            f = uproot.open(fn)
            g = f['Events']
            output = g['output'].array()
            target = g['target_mass'].array()
            hist_tmp, xedge, yedge = np.histogram2d(output, target, range=[histrange, histrange], bins=50)
            hist+=hist_tmp


        c = ax1.pcolor(xedge, yedge, hist)#hist, X=xedge, Y=yedge)
        #fig1.colorbar(c, ax=ax1)
        ax1.set_title(f'{part} mass regression')
        ax1.set_ylabel('predicted mass')
        ax1.set_xlabel('target mass')
        fig1.savefig(f'plots/predict_{part}_mass_2d.png', bbox_inches='tight')
        
        c = ax1.pcolor(xedge, yedge, hist, norm=LogNorm())
        fig1.colorbar(c, ax=ax1)
        fig1.savefig(f'plots/predict_{part}_mass_2d_log.png', bbox_inches='tight')
        
        plt.close(fig1)
        
    plt.close('all')

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
    ## labels : list of what goes into the legend for each filename
    ## nbins : [nbins for predict and target 1d, nbins for ratio 1d]
    ## binranges : [binrange for predict and target 1d, binrange for ratio]
    ## for furure si who is confused and dumb:: The files (and other stff) that go in here are same mod different masspoints
    
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots( )
    fig4, ax4 = plt.subplots()

    mmsList = []
    rmsList = []

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
        ax.hist(output, **histdict)
        ax2.hist(target,**histdict)
        histdict['bins'] = nbins[1]
        histdict['range'] = binranges[1]
        ratio = np.divide(output, target, where=output/target>0)
        
        ## resolution plot
        ax3.hist(np.log2(ratio), **histdict)
        
        ## s/sqrt(B)
        ## Plot with 101 bins from 0 to 2.02, using the last bin as “overflow” for all events with [pred / targ] > 2.0
        ## Sum the squares of the bin values for the first 100 bins (excluding the overflow bin), multiply by 100 (the number of bins 
        ## summed), and divide by the square of the total integral (including the overflow bin).
        s_hist, s_edge = np.histogram(np.clip(ratio,a_min=0, a_max=2), bins=101, range=(0,2.02))
        binwidth = s_edge[1]-s_edge[0]
        sumsqr = np.sum(s_hist[:99])*100/np.square(np.sum(s_hist)*binwidth)
        ax4.step( s_edge[:-1] , s_hist, label=f'{label} :{sumsqr:.4f}')

        print(np.sum(np.ones(100)*100/np.square(np.sum(np.ones(100))*binwidth)))

        ## fill table for the mms and rms
        mms = np.mean(np.log2(output/target))
        rms = np.std(np.log2(output/target))
        mmsList.append(f'{mms:.4f}')
        rmsList.append(f'{rms:.4f}')
        
    ax3.table(
        colLabels=labels,
        rowLabels=['MMS', 'RMS'],
        cellText = [mmsList, rmsList],
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
        binrange= [0,70]
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
def calc_RMS_MMS(masspoints, filenames, log2=False):
    df = pd.DataFrame(masspoints, columns=['masspoints'])
    df['filenames'] = filenames
    df.sort_values(by='masspoints', ignore_index=True, inplace=True)
    mmsList = []
    rmsList = []

    
    for fn in df['filenames']: 
        f = uproot.open(fn)
        g = f['Events']
        output = g['output'].array()
        target = g['target_mass'].array()
        pt = g['fj_pt'].array()

        output, target, binrange = returnToMass(output, target, pt, fn)
        where = {'where':output>0, 'out':np.zeros_like(output)}

        mms = np.mean(np.log2(output/target, **where))
        rms = np.std(np.log2(output/target, **where))
        mmsList.append(f'{mms:.4f}')
        rmsList.append(f'{rms:4f}')
        
    
    return rmsList, mmsList, df['masspoints'].values.tolist()


if __name__ == "__main__":
    main()
