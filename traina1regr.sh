

#python train.py --regression-mode --data-train '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/20000/AK8_HToAATo4B_Pt150_mH-70_mA-1*' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 40 --model-prefix output/VarHMass_20000_a1_regr

python train.py --regression-mode --data-train 'https://baylorhep.slack.com/archives/C013B0LRAEA/p1712261642970739' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 40 --model-prefix output/VarHMass_20000_a1_regr
