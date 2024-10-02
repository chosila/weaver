import os
import glob


masspoints = [fn[fn.find('_M-')+3:].split('.root')[0] for fn in glob.glob('/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root')]
#masspoints = [float(x) for x in masspoints]
#masspoints.sort()

## create modified mass target training scripts
mods = ['mass', 'logMass']#['mass', 'logMass', 'massOverPT', 'logMassOverPT', 'ptOverMass', 'massOverfj_mass']
parts = ['a1'] #['H_calc', 'a1' ,'a2']
numhads = ['3b', '34b']
loss_modes = [0,3]

## a1 centrally produced train all mods
for lm in loss_modes:
    for numhad in numhads: 
        for part in parts:
            for mod in mods: 
                with open(f'train_h125_{part}_{mod}_{numhad}_lm{lm}.sh', 'w+') as f:
                        txt = f'''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
python train.py --regression-mode --data-train '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/34b/{part}_{mod}_{numhad}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 20 --model-prefix output/34b/h125_{part}_{mod}_{numhad}_lm{lm}_regr --network-option loss_mode {lm}
echo "h125_{part}_{mod}_{numhad}_lm{lm}_regr trained" >> trained.log 
'''

                    ## maybe we don't want this yet? do the export onnx after we've already trained
                        #--export-onnx output/onnx/34b/h125_{part}_{mod}_{numhad}_regr.onnx'''
                        f.write(txt)
    
        

## script to submit all training
with open('submitall.sh', 'w+') as f:
    for fn in os.listdir('.'):
        if ('train' in fn):
            print(fn)
            if ('.sh' in fn) and ('submitall' not in fn) and ('.e' not in fn) and ('.o' not in fn) and ('~' not in fn): 
                f.write(f'qsub -q hep -l nodes=1:ppn=18 {fn}\n')
                pass

        

# ## file to export to onnx
# with open('exportToOnnx.sh', 'w+') as f:
#     txt = f'''source ~/.bashrc
# source ~/.bash_profile
# conda activate weaver
# cd /home/chosila/Projects/weaver
#     '''
#     f.write(txt)        
#     for part in ['H_calc']: #parts:
#         for mod in mods:
            
#             txt = f'python train.py -c data/{part}_{mod}_regr.yaml -n networks/particle_net_pf_sv.py -m /home/chosila/Projects/weaver/output/wide_{part}_{mod}_regr_best_epoch_state.pt --export-onnx output/onnx/wide_{part}_{mod}_regr.onnx \n'
#             f.write(txt)


## predict script for centrally produced gluglu sample
#parts = #['a1' ,'H_calc', 'a2']
print(mods)
with open('predict_mod_targets.sh', 'w+') as f:
        
    f.write('''source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
''')
    for lm in loss_modes:
        for part in parts:
            for masspoint in masspoints:
                for mod in mods:
                    for numhad in numhads + ['4b']:
                        txt1=f"python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-{masspoint}.root' --data-config data/34b/{part}_{mod}_{numhad}_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/34b/h125_{part}_{mod}_{numhad}_lm{lm}_regr --predict-output predict/34b/predict_h125_{part}_M{float(masspoint)}_{mod}_{numhad}_lm{lm}_regr.root --network-option loss_mode {lm}\n"
                        f.write(txt1)

