qsub -q hep -l nodes=1:ppn=18 train_a1_mass_loss0_epoch20.sh
qsub -q hep -l nodes=1:ppn=18 train_a1_mass_loss3_epoch20.sh
qsub -q hep -l nodes=1:ppn=18 train_a1_logMass_loss0_epoch20.sh
qsub -q hep -l nodes=1:ppn=18 train_a1_logMass_loss3_epoch20.sh
