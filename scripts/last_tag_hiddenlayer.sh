#!/bin/bash
declare -a hidden_layer_size=(50 500 1500 3000)
#declare -a window=(1 3 5 7)

#for j in "${arr_type[@]}"
#do
for i in "${hidden_layer_size[@]}"
do
	echo "Training last_tag hidden_layer_size: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window 5 --hidden "$i" --tagdim 10 --token --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --earlystop 2 --logger earlystop_hidden_layer_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window 5 --hidden "$i" --tagdim 10 --token --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --picklelists --logger hidden_layer_"$i".log
done
#done
