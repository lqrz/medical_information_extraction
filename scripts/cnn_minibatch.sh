#!/bin/bash
declare -a minibatch_size=(32 64 128 256)

for i in "${minibatch_size[@]}"
do
	echo "Training cnn minibatch_size: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window 7 --epochs 20 --plot --lrtrain 0.01 --lrtune 0.01 --configini cnn_w2v_c_m.cfg --earlystop 2 --minibatch "$i" --logger earlystop_minibatch_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window 7 --epochs 20 --plot --lrtrain 0.01 --lrtune 0.01 --configini cnn_w2v_c_m.cfg --picklelists --minibatch "$i" --logger minibatch_"$i".log
done
