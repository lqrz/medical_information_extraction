#!/bin/bash
declare -a lr_train=(0.1 0.01)
declare -a lr_tune=(0.01 0.001)

for i in "${lr_train[@]}"
do
	for j in "${lr_tune[@]}"
	do
		echo "lr_train $i lr_tune $j"
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window 7 --epochs 20 --plot --lrtrain "$i" --lrtune "$j" --minibatch 32 --configini cnn_w2v_c_m.cfg --earlystop 2 --logger earlystop_lrtrain_"$i"_lrtune_"$j".log
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window 7 --epochs 20 --plot --lrtrain "$i" --lrtune "$j" --minibatch 32 --configini cnn_w2v_c_m.cfg --picklelists --logger lrtrain_"$i"_lrtune_"$j".log
	done
done