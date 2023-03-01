import uproot 
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
import pandas as pd
import matplotlib as mpl
import os
#import pandas 

def main():
    predType = 'regr'

    ## create 1D predict and target plots for each mass point of centrally produced Ntuples
    fl2 = ['output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-15_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-25_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-35_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-45_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-55_numEvent20000.root',]
           
    fl1 = ['output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-12_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-20_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-30_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-40_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-50_numEvent20000.root',
           'output/predict_a1_AK8_HToAATo4B_GluGluH_01J_Pt150_M-60_numEvent20000.root',]


    fl3 = ['output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-12_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-20_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-30_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-40_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-60_numEvent20000.root',]

    fl4 = ['output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-15_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-25_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-35_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-45_numEvent20000.root',
           'output/predict_a2_AK8_HToAATo4B_GluGluH_01J_Pt150_M-55_numEvent20000.root'] 

    titlepairs = [('a1 prediction with masses: 12, 20, 30, 40, 50, 60', 'a1 target with massses: 12, 20, 30, 40, 50, 60'), 
                  ('a1 prediction with masses: 15, 25, 35, 45, 55',     'a1 target with masses: 15, 25, 35, 45, 55 '),
                  ('a2 prediction with masses: 12, 20, 30, 40, 50, 60', 'a2 target with masses: 12, 20, 30, 40, 50, 60'),
                  ('a2 predection with masses: 15, 25, 35, 45, 55',     'a2 target with masses: 15, 25, 35, 45, 55')]
    plotnamepairs = [('prediction_a1_12.png', 'target_a1_12.png'),
                     ('prediction_a1_15.png', 'target_a1_15.png'),
                     ('prediction_a2_12.png', 'target_a2_12.png'),
                     ('prediction_a2_15.png', 'target_a2.15.png')]

    for fl, titles, plotnames in zip([fl1,fl2,fl3,fl4], titlepairs, plotnamepairs):
        make1DDist(fl, titles, plotnames)
    ## -------------------------------- end 1D central Ntuple plots ----------------------------



    ## 1D Plot wide mass point Ntuples 
    fl = 






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
            print(log)
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


def getBinCenter(edges):
    left = edges[:-1]
    right = edges[1:]
    return (left + right)/2
        

def make1DDist(fl, titles, plotnames):
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    nbins = 50
    binrange = (0,80)

    for fn in fl:
        f = uproot.open(fn)
        g = f['Events']
        output = g['output'].array()[1::2]
        target = g['target_mass'].array()[1::2]

        split = fn.split('_')
        name = split[1] + "_" + split[7]
        print(name)
        ax.hist(output, bins=nbins, range=binrange, histtype='step', label=f'{split[7]}')
        ax2.hist(target, bins=nbins, range=binrange, histtype='step', label=f'{split[7]}')

    ax.legend()
    ax2.legend()
    ax.set_title(titles[0])
    ax2.set_title(titles[1])
    fig.savefig(f'plots/{plotnames[0]}.png',bbox_inches='tight')
    fig2.savefig(f'plots/{plotnames[1]}.png', bbox_inches='tight')
    plt.close('all')

if __name__ == "__main__":
    main()










    
#print(g['label_H_aa_other'].values())
#print(g['score_label_H_aa_other'].values())


#[b'label_H_aa_bbbb', b'score_label_H_aa_bbbb', b'label_H_aa_other', b'score_label_H_aa_other', b'sample_isQCD', b'score_sample_isQCD', b'event_no', b'label_QCD_BGen', b'label_QCD_bEnr', b'sample_min_LHE_HT', b'fj_pt', b'fj_eta', b'fj_mass', b'fj_sdmass', b'fj_corrsdmass', b'fj_sdmass_fromsubjets', b'fj_gen_pt', b'fj_genjet_pt', b'fj_genjet_mass', b'fj_genjet_sdmass', b'fj_gen_H_aa_bbbb_mass_a', b'fj_gen_H_aa_bbbb_dR_max_b', b'fj_gen_H_aa_bbbb_pt_min_b', b'pfParticleNetMassRegressionJetTags_mass', b'pfParticleNetDiscriminatorsJetTags_HbbvsQCD', b'pfParticleNetDiscriminatorsJetTags_H4qvsQCD', b'pfMassDecorrelatedParticleNetDiscriminatorsJetTags_XbbvsQCD'] 
