#!/bin/bash
declare -a arr_dir=('--unidirectional' '--bidirectional')
declare -a arr_type=('normal' 'lstm' 'gru')

for j in "${arr_type[@]}"
do
	for i in "${arr_dir[@]}"
	do
		echo "Training type: $j direction: $i"
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_rnn --window 1 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --minibatch 128 --rnntype "$j" --earlystop 2 "$i" --logger earlystop_"$j"_"$i".log
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_rnn --window 1 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --minibatch 128 --rnntype "$j" --picklelists "$i" --logger "$j"_"$i".log
	done
done
