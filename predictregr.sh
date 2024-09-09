python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/20000/*.root' --data-config data/H_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/VarHMass_20000_H_regr --predict-output output/predict_VarHMass_20000_H_regr.root
python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/20000/*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/VarHMass_20000_a1_regr --predict-output output/predict_VarHMass_20000_a1_regr.root
python train.py --predict --regression-mode --data-test '/home/chosila/Projects/CMSSW_10_6_32/src/DeepNTuples/Ntuples/20000/*.root' --data-config data/a2_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/VarHMass_20000_a2_regr --predict-output output/predict_VarHMass_20000_a2_regr.root






