#!/bin/bash
declare -a update=('token' 'sentence')

for i in "${update[@]}"
do
	echo "Training last_tag update: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window 5 --tagdim 10 --"$i" --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --earlystop 2 --logger earlystop_update_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window 5 --tagdim 10 --"$i" --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --picklelists --logger update_"$i".log
done