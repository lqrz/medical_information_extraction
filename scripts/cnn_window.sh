#!/bin/bash
#declare -a window_size=(1 3 5 7)
#
#for i in "${window_size[@]}"
#do
#	echo "Training last_tag window_size: $i"
#	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$i" --epochs 20 --plot --lrtrain 0.01 --lrtune 0.01 --minibatch 256 --configini cnn_w2v_c_m.cfg --earlystop 2 --logger earlystop_window_"$i".log
#	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$i" --epochs 20 --plot --lrtrain 0.01 --lrtune 0.01 --minibatch 256 --configini cnn_w2v_c_m.cfg --picklelists --logger window_"$i".log
#done

declare i=7
sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$i" --epochs 20 --plot --lrtrain 0.01 --lrtune 0.01 --minibatch 256 --configini cnn_w2v_c_m.cfg --earlystop 2 --logger earlystop_window_"$i".log
sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$i" --epochs 20 --plot --lrtrain 0.01 --lrtune 0.01 --minibatch 256 --configini cnn_w2v_c_m.cfg --picklelists --logger window_"$i".log