import os
import numpy as np

pre = '''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver

'''

txt = '''
python train.py --regression-mode --data-train '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config /home/chosila/Projects/weaver/data/a1_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs {numepoch} --model-prefix output/testepoch/a1_calc_{mod}_regr_loss{loss_mode}_epoch{numepoch} --network-option loss_mode {loss_mode}
'''
## training script 
mods = ['mass', 'logMass']#, 'massOverfj_mass']
loss_modes = [0,3]#[0,1,2,3]
submit = open('submitall.sh', 'w+')
numepochs = [20]
for mod in mods:
    for loss_mode in loss_modes:
        for numepoch in numepochs:
            fn = 'train_a1_{mod}_loss{loss_mode}_epoch{numepoch}.sh'.format(mod=mod, loss_mode=loss_mode, numepoch=numepoch)
            submit.write(f'qsub -q hep -l nodes=1:ppn=18 {fn}\n')
            with open(fn, 'w+') as f:
                f.write(pre)
                f.write(txt.format(mod=mod, loss_mode=loss_mode, numepoch=numepoch))

## submit script

            
## predict script
pre = '''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
'''
txt ='''python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_{mod}_regr_loss{loss_mode}_epoch20_epoch-{numepoch}_state.pt --predict-output predict/testepoch/predict_a1_calc_{mod}_regr_loss{loss_mode}_epoch{numepoch}.root --network-option loss_mode {loss_mode}
'''
txt_epoch40 ='''python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_{mod}_regr_loss{loss_mode}_epoch20 --predict-output predict/testepoch/predict_a1_calc_{mod}_regr_loss{loss_mode}_epoch{numepoch}.root --network-option loss_mode {loss_mode}
'''


numepochs = [20] #[0,1,2,4,6,12,16,18,20]#[8,16,24,32,40]#[10,15,20,25,30,35,40]
with open('predict_a1_mod_targets_epoch.sh', 'w+') as f:
    f.write(pre)
    for mod in mods:
        for loss_mode in loss_modes:
            for numepoch in numepochs:
                if numepoch == 20:
                    f.write(txt_epoch40.format(mod=mod, loss_mode=loss_mode, numepoch=numepoch))
                else:
                    f.write(txt.format(mod=mod, loss_mode=loss_mode, numepoch=numepoch))

                
## predict for 2016 and 2017
pre = '''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
'''
txt ='''python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/{year}/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_{mod}_regr_loss{loss_mode}_epoch20_epoch-{numepoch}_state.pt --predict-output predict/testepoch/{year}/predict_a1_calc_{mod}_regr_loss{loss_mode}_epoch{numepoch}.root --network-option loss_mode {loss_mode}
'''
txt_epoch40 ='''python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/{year}/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_{mod}_regr_loss{loss_mode}_epoch20 --predict-output predict/testepoch/{year}/predict_a1_calc_{mod}_regr_loss{loss_mode}_epoch{numepoch}.root --network-option loss_mode {loss_mode}
'''
numepochs = [20] #[0,1,2,4,6,12,16,18,20]#[8,16,24,32,40]#[10,15,20,25,30,35,40]
with open('predict_a1_2016_2017.sh', 'w+') as f:
    f.write(pre)
    for mod in mods:
        for loss_mode in loss_modes:
            for year in [2016,2017]:
                    f.write(txt_epoch40.format(mod=mod, loss_mode=loss_mode, numepoch=numepochs[0], year=year))

