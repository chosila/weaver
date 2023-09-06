import os 


parts = ['H', 'a1', 'a2']
for part in parts:
    

    with open(f'train_GLUGLU_{part}.sh', 'w+') as txtf:
        
        txt = f'''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
python train.py --regression-mode --data-train '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*' --data-config data/{part}_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 40 --model-prefix output/GluGlu_{part}_regr '''
        txtf.write(txt)



## create modified mass target training scripts
mods = ['mass', 'logMass', 'massOverPT', 'logMassOverPT', 'ptOverMass', 'massOverfj_mass']
parts = ['H_calc', 'a1' ,'a2']
for part in parts:
    for mod in mods:
        with open(f'train_wideH_{part}_{mod}.sh' ,'w+') as f:
            txt = f'''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
python train.py --regression-mode --data-train '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_SUSY_GluGluH_01J_HToAATo4B_Pt*50_mH-70_mA-12_wH-70_wA-70_TuneCP5_13TeV_madgraph_pythia8.root' --data-config data/{part}_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 40 --model-prefix output/wide_{part}_{mod}_regr '''
            f.write(txt)


## a1 centrally produced train all mods
for part in parts:
    for mod in mods: 
        with open(f'train_central_{part}_{mod}.sh', 'w+') as f:
                txt = f'''source ~/.bashrc
    source ~/.bash_profile
    conda activate weaver
    cd /home/chosila/Projects/weaver
    python train.py --regression-mode --data-train '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_HToAATo4B_GluGluH*' --data-config data/{part}_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 40 --model-prefix output/central_{part}_{mod}_regr --export-onnx output/onnx/central_{part}_{mod}_regr.onnx'''
                f.write(txt)

        

## script to submit all training
with open('submitall.sh', 'w+') as f:
    for fn in os.listdir('.'):
        if ('train' in fn):
            print(fn)
            if ('.sh' in fn) and ('submitall' not in fn) and ('.e' not in fn) and ('.o' not in fn) and ('~' not in fn): 
                f.write(f'qsub -q hep -l nodes=1:ppn=18 {fn}\n')
                pass

        

## file to export to onnx
with open('exportToOnnx.sh', 'w+') as f:
    txt = f'''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
    '''
    f.write(txt)        
    for part in ['H_calc']: #parts:
        for mod in mods:
            
            txt = f'python train.py -c data/{part}_{mod}_regr.yaml -n networks/particle_net_pf_sv.py -m /home/chosila/Projects/weaver/output/wide_{part}_{mod}_regr_best_epoch_state.pt --export-onnx output/onnx/wide_{part}_{mod}_regr.onnx \n'
            f.write(txt)

    
## predict for wide H mass 
with open('predict_wideH_mod_targets.sh', 'w+') as f:
    f.write('''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
''')
    #mods.remove('mass')
    for mod in mods:
        for ptpoint in ['150', '250', '350']:
            for part in parts:
                txt = f"python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_SUSY_GluGluH_01J_HToAATo4B_Pt{ptpoint}_mH-70_mA-12_wH-70_wA-70_TuneCP5_13TeV_madgraph_pythia8.root' --data-config data/predict/{part}_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/wide_{part}_{mod}_regr --predict-output predict/predict_wide_{part}_pt{ptpoint}_{mod}_regr.root\n"
                f.write(txt)

## predict script for centrally produced gluglu sample
parts = ['a1' ,'H_calc', 'a2']
print(mods)
with open('predict_mod_targets.sh', 'w+') as f:
        
    f.write('''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
''')
    for part in parts:
        for masspoint in [12,15,20,25,30,35,40,45,50,55,60]:
            for mod in ['mass']:#mods:
                txt1=f"python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/AK8_HToAATo4B_GluGluH_01J_Pt150_M-{masspoint}.root' --data-config data/{part}_{mod}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/central_{part}_{mod}_regr --predict-output predict/predict_central_{part}_M{masspoint}_{mod}_regr.root\n"
                f.write(txt1)

## create the predict scripts 
# with open('predict_a_point_mass.sh','w+') as f:
#     f.write('''source ~/.bashrc
# source ~/.bash_profile
#     conda activate weaver
#     cd /home/chosila/Projects/weaver
# ''')
#     for fn in os.listdir('/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/'):
#         if 'AK8_HToAATo4B_GluGluH_01J_Pt150_M-' in fn:
#             txt1=f"python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/{fn}' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/GluGlu_a2_regr --predict-output output/predict_a1_{fn}\n"
#             txt2=f"python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/{fn}' --data-config data/a2_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/GluGlu_a2_regr --predict-output output/predict_a2_{fn}\n"
#             f.write(txt1)
#             f.write(txt2)
