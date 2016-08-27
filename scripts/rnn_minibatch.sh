#!/bin/bash
declare -a arr_type=('normal' 'lstm' 'gru')
declare -a arr_minibatch=(32 64 128 256)

for j in "${arr_type[@]}"
do
	for i in "${arr_minibatch[@]}"
	do
		echo "Training type: $j minibatch_size: $i"
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_rnn --window 1 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --unidirectional --minibatch "$i" --rnntype "$j" --earlystop 2 --logger earlystop_"$j"_minibatch_size_"$i".log
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_rnn --window 1 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --unidirectional --minibatch "$i" --rnntype "$j" --picklelists --logger "$j"_minibatch_size_"$i".log
	done
done
