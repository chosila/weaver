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
    predType = 'regr'
    
    mods = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT', 'ptOverMass']
    #mods.remove('ptOverMass')
    parts = ['H_calc', 'a1', 'a2']
    parts = ['a1']
    #mods.remove('logMassOverPT')
    ## rms and mms vs masspoint figs 
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    labels = [['M-12', 'M-20', 'M-30', 'M-40', 'M-50', 'M-60'],
              ['M-15', 'M-25', 'M-35', 'M-45', 'M-55'],
              ['M-12', 'M-20', 'M-30', 'M-40', 'M-50', 'M-60'],
              ['M-15', 'M-25', 'M-35', 'M-45', 'M-55']]


    df = pd.DataFrame()
    mp1 = [12,20,30,40,50,60]
    mp2 = [15,25,35,45,55]
    ## centrally produced miniaod
    for part in [0]: #parts:
        for mod in mods:
            fn1 = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mp1]
            fn2 = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mp2]
            ## make 1d hist for the first half of the set 
            titles = (f'predict {mod} M12,20,30,40,50,60', f'predict {mod} M12,20,30,40,50,60')
            plotnames = (f'predict_a1_M12_{mod}', f'target_a1_M12_{mod}')
            make1DDist(fn1, titles, plotnames, labels[0])
            ## make 1d hist for the second half of set 
            titles =  (f'predict {mod} M15,25,35,45,55', f'predict {mod} M15,25,35,45,55')
            plotnames = (f'predict_a1_M15_{mod}', f'target_a1_M15_{mod}')
            make1DDist(fn2, titles, plotnames, labels[1])
        
            ## rms and mms vs masspoint for various mods
            rms, mms, masspoints = calc_RMS_MMS(mp1+mp2, fn1+fn2) 
            
            ax4.plot(masspoints, rms, label=mod)
            ax5.plot(masspoints, mms, label=mod) 
            df['masspoints'] = masspoints
            df[f'rms_{mod}'] = rms
            df[f'mms_{mod}'] = mms

    
        
    ## adding the only mass training to the rms plots 
    fns = [f'predict/predict_central_a1_M{x}_mass_regr.root' for x in mp1+mp2]
    rms, mms, masspoints = calc_RMS_MMS(mp1+mp2, fns)
    df[f'rms_mass'] = rms
    df[f'mms_mass'] = mms

    df.to_csv('RMS_MMS_MsPnt.csv')
    plt.close('all')

    mps = mp1+mp2
    mps.sort()
    ## mass resolution for centrally produced 

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    for histrange in [(-2,2)]:
        for mod in mods:
            
                                   
            for mps, tag in zip([mp1, mp2], ['M-12','M-15']):
                fns = [f'predict/predict_central_a1_M{masspoint}_{mod}_regr.root' for masspoint in mps]
                mmsList = []
                rmsList = []
                for fn, masspoint in zip(fns, mps):
                    print(fn)
                    f = uproot.open(fn)
                    g = f['Events'] 
                    output = g['output'].array()
                    target = g['target_mass'].array()
                    pt = g['fj_pt'].array()
                    
                    histdict = {'bins':50, 'range':histrange, 'histtype':'step', 'label':f'M-{masspoint}'}
                    #ax1.hist(np.clip(output, a_min=histrange[0], a_max=histrange[1]), **histdict)                
                    #ax2.hist(np.clip(target, a_min=histrange[0], a_max=histrange[1]), **histdict)
                    ax3.hist(np.log2(output/target, where=output/target>0), **histdict)
                    
                    mms = np.mean(np.log2(output/target, where=output/target>0))
                    mmsList.append(f'{mms:.4f}')
                    rmsList.append(np.std(np.log2(output/target, where=output/target>0)))
                    
                ax3.table(
                    colLabels=[f'M-{mp}' for mp in mps],
                    rowLabels=['MMS', 'RMS'],
                    cellText = [mmsList, rmsList],
                    bbox=[0.1, -0.3, 0.9, 0.2]
                )
                    
    
                #ax1.legend()
                #ax2.legend()
                ax3.legend()
                #ax1.set_title(f'{mod} resolution predict a1')
                #ax2.set_title(f'{mod} resolution target a1')
                ax3.set_title(f'{mod} ratio of predict to target a1 {tag}')
                # predict_a1_M12_1OverMass.png
                #fig1.savefig(f'plots/predict_a1_{mod}.png', bbox_inches='tight')
                #fig2.savefig(f'plots/target_a1_{mod}.png', bbox_inches='tight')
                fig3.savefig(f'plots/ratio_a1_{tag}_{mod}.png', bbox_inches='tight')
                #ax1.cla()
                #ax2.cla()
                ax3.cla()







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

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots() 
    fig5, ax5 = plt.subplots()

    ## mass resolution (log2(predict/target))
    for part, histrange in zip(parts, histranges):
        for mod in ['mass']: #mods: 
            #for mp in [150, 250, 350]:
            fns = [f'predict/predict_{part}_pt{mp}_{mod}_regr.root' for mp in labels]
            mmsList = []
            rmsList = []

            for fn, label in zip(fns, labels): 
                f = uproot.open(fn)
                g = f['Events']
                output = g['output'].array() #np.clip(g['output'].array(), a_min = histrange[0], a_max=histrange[1])
                target = g['target_mass'].array()#np.clip(g['target_mass'].array(), a_min = histrange[0], a_max=histrange[1])
                pt = g['fj_pt'].array()

                histdict = {'bins':50, 'range':histrange, 'histtype':'step', 'label':f'pt{label}'}
                ax1.hist(np.clip(output, a_min=histrange[0], a_max=histrange[1]), **histdict)
                ax2.hist(np.clip(target, a_min=histrange[0], a_max=histrange[1]), **histdict)
                ax3.hist(np.log2(output/target, where=output/target>0), **histdict)

                mms = np.mean(np.log2(output/target))
                mmsList.append(f'{mms:.4f}')
                rmsList.append(np.std(np.log2(output/target)))


            ax3.table(
                colLabels=labels,
                rowLabels=['MMS', 'RMS'],
                cellText = [mmsList, rmsList],
                bbox=[0.1, -0.3, 0.9, 0.2]
            )
            

         
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax1.set_title(f'predict {part}')
            ax2.set_title(f'target {part}')
            ax3.set_title('ratio of predict to target {part}')
            # predict_a1_M12_1OverMass.png
            fig1.savefig(f'plots/predict_{part}_mass.png', bbox_inches='tight')
            fig2.savefig(f'plots/target_{part}_mass.png', bbox_inches='tight')
            fig3.savefig(f'plots/ratio_{part}_mass.png', bbox_inches='tight')
            ax1.cla()
            ax2.cla()
            ax3.cla()

        #ax4.plot(label, rms)
        #ax5.plot(label, mms)
        
            


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
        

def make1DDist(fl, titles, plotnames, labels):
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots( )#height_ratios=[2,1])
    nbins = 50
    binrange = None#(2,6)
    rmsList = []
    mmsList = []
    # if '12' in fl[0]:
    #     masspoints = [12,20,30,40,50,60]
    # elif '15' in fl[0]:
    #     masspoints = [15,25,35,45,55]

    for fn, label in zip(fl, labels):
        f = uproot.open(fn)
        g = f['Events']
        output = g['output'].array()
        target = g['target_mass'].array()
        pt = g['fj_pt'].array()
        
        ## modify output and target back to mass if needed 
        output, target, binrange = returnToMass(output, target, pt, fn)
        
        histdict = {'bins':nbins, 'range':binrange, 'histtype':'step', 'label':label}
        ax.hist(output, **histdict)
        ax2.hist(target,**histdict)
        ax3.hist(np.log2(output/target, where=output/target>0), **histdict)
        
        ## fill table for the mms and rms
        mms = np.mean(np.log2(output/target))
        mmsList.append(f'{mms:.4f}')
        rmsList.append(np.std(np.log2(output/target)))
        
    ax3.table(
        colLabels=labels,
        rowLabels=['MMS', 'RMS'],
        cellText = [mmsList, rmsList],
        bbox=[0.1, -0.3, 0.9, 0.2]
    )
    
    #plt.subplots_adjust(left=-0.2, bottom=-0.2)

    ax.legend()
    ax2.legend()
    ax3.legend()
    ax.set_title(titles[0])
    ax2.set_title(titles[1])
    ax3.set_title(f'ratio {titles[0]}')
    fig.savefig(f'plots/{plotnames[0]}.png',bbox_inches='tight')
    fig2.savefig(f'plots/{plotnames[1]}.png', bbox_inches='tight')
    fig3.savefig(f'plots/ratio_{plotnames[0]}.png', bbox_inches='tight')
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
def calc_RMS_MMS(masspoints, filenames):
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
        mmsList.append(f'{mms:.4f}')
        rmsList.append(f'{np.std(np.log2(output/target, **where)):4f}')
        
    
    return rmsList, mmsList, df['masspoints'].values.tolist()


if __name__ == "__main__":
    main()
