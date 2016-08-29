#!/bin/bash
declare -a arr_type=('gru')
declare -a grad_clip=(5 10 30 50)

for j in "${arr_type[@]}"
do
	for i in "${grad_clip[@]}"
	do
		echo "Training type: $j grad_clip: $i"
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_rnn --window 1 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --unidirectional --minibatch 128 --rnntype "$j" --earlystop 2 --gradclip "$i" --logger earlystop_"$j"_grad_clip_"$i".log
		stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_rnn --window 1 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --unidirectional --minibatch 128 --rnntype "$j" --picklelists --gradclip "$i" --logger "$j"_grad_clip_"$i".log
	done
done
