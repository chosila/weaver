import os
import numpy as np

pre = '''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver

'''

txt = '''
python train.py --regression-mode --data-train '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_SUSY_GluGluH_01J_HToAATo4B_Pt*50_mH-70_mA-12_wH-70_wA-70_TuneCP5_13TeV_madgraph_pythia8.root' --data-config /home/chosila/Projects/weaver/data/H_calc_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 40 --model-prefix output/testLoss/wide_H_calc_{mod}_regr_loss{loss_mode} --network-option loss_mode {loss_mode}
'''
## training script 
mods = ['mass', 'logMass', 'massOverfj_mass']
loss_modes = [0,1,2,3]
submit = open('submitall.sh', 'w+')
for mod in mods:
    for loss_mode in loss_modes:
        fn = 'train_wideH_H_calc_{mod}_loss{loss_mode}.sh'.format(mod=mod, loss_mode=loss_mode)
        submit.write(f'qsub -q hep -l nodes=1:ppn=18 {fn}\n')
        with open(fn, 'w+') as f:
            f.write(pre)
            f.write(txt.format(mod=mod, loss_mode=loss_mode))

## submit script

            
## predict script
pre = '''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
'''
txt ='''python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_SUSY_GluGluH_01J_HToAATo4B_Pt*50_mH-70_mA-12_wH-70_wA-70_TuneCP5_13TeV_madgraph_pythia8.root' --data-config data/H_calc_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testLoss/wide_H_calc_{mod}_regr_loss{loss_mode} --predict-output predict/testLoss/predict_wide_H_calc_{mod}_regr_loss{loss_mode}.root --network-option loss_mode {loss_mode}
'''

with open('predict_wideH_mod_targets.sh', 'w+') as f:
    f.write(pre)
    for mod in mods:
        for loss_mode in loss_modes:
            f.write(txt.format(mod=mod, loss_mode=loss_mode))
