source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2016/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_mass_regr_loss0_epoch20 --predict-output predict/testepoch/2016/predict_a1_calc_mass_regr_loss0_epoch20.root --network-option loss_mode 0
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_mass_regr_loss0_epoch20 --predict-output predict/testepoch/2017/predict_a1_calc_mass_regr_loss0_epoch20.root --network-option loss_mode 0
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_mass_regr_loss0_epoch20_epoch-20_state.pt --predict-output predict/testepoch/2017/predict_a1_calc_mass_regr_loss0_epoch20.root --network-option loss_mode 0
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2016/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_mass_regr_loss3_epoch20 --predict-output predict/testepoch/2016/predict_a1_calc_mass_regr_loss3_epoch20.root --network-option loss_mode 3
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_mass_regr_loss3_epoch20 --predict-output predict/testepoch/2017/predict_a1_calc_mass_regr_loss3_epoch20.root --network-option loss_mode 3
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_mass_regr_loss3_epoch20_epoch-20_state.pt --predict-output predict/testepoch/2017/predict_a1_calc_mass_regr_loss3_epoch20.root --network-option loss_mode 3
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2016/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_logMass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_logMass_regr_loss0_epoch20 --predict-output predict/testepoch/2016/predict_a1_calc_logMass_regr_loss0_epoch20.root --network-option loss_mode 0
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_logMass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_logMass_regr_loss0_epoch20 --predict-output predict/testepoch/2017/predict_a1_calc_logMass_regr_loss0_epoch20.root --network-option loss_mode 0
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_logMass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_logMass_regr_loss0_epoch20_epoch-20_state.pt --predict-output predict/testepoch/2017/predict_a1_calc_logMass_regr_loss0_epoch20.root --network-option loss_mode 0
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2016/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_logMass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_logMass_regr_loss3_epoch20 --predict-output predict/testepoch/2016/predict_a1_calc_logMass_regr_loss3_epoch20.root --network-option loss_mode 3
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_logMass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_logMass_regr_loss3_epoch20 --predict-output predict/testepoch/2017/predict_a1_calc_logMass_regr_loss3_epoch20.root --network-option loss_mode 3
python train.py --predict --regression-mode --data-test '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2017/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config data/a1_logMass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --model-prefix output/testepoch/a1_calc_logMass_regr_loss3_epoch20_epoch-20_state.pt --predict-output predict/testepoch/2017/predict_a1_calc_logMass_regr_loss3_epoch20.root --network-option loss_mode 3
