source ~/.bashrc
source ~/.bash_profile
conda activate weaver
cd /home/chosila/Projects/weaver


python train.py --regression-mode --data-train '/cms/data/store/user/abrinke1/Trees/HtoAAto4B/ParticleNet/Ntuples/2024_04_01/2018/AK8_HToAATo4B_GluGluH_01J_Pt150_M-*.root' --data-config /home/chosila/Projects/weaver/data/a1_mass_regr.yaml --network-config networks/particle_net_pf_sv_mass_regression.py --num-epochs 20 --model-prefix output/testepoch/a1_calc_mass_regr_loss0_epoch20 --network-option loss_mode 0
