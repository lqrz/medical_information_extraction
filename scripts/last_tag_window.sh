#!/bin/bash

declare -a window=(1 3 5 7)

for i in "${window[@]}"
do
	echo "Training last_tag window: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window "$i" --tagdim 10 --token --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --earlystop 2 --logger earlystop_window_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window "$i" --tagdim 10 --token --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --picklelists --logger window_"$i".log
done
