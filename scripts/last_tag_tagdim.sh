#!/bin/bash
declare -a tag_dim=(1 5 10 20 50 100 250)
#declare -a window=(1 3 5 7)

#for j in "${arr_type[@]}"
#do
for i in "${tag_dim[@]}"
do
	echo "Training last_tag tag_dim: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window 5 --tagdim "$i" --token --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --earlystop 2 --logger earlystop_tagdim_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window 5 --tagdim "$i" --token --epochs 30 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --picklelists --logger tagdim_"$i".log
done
#done
