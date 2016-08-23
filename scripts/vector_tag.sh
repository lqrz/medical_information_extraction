#!/bin/bash
declare -a arr=(500 1500 3000)
for i in "${arr[@]}"
do
	sudo python SOTA/neural_network/train_neural_network.py --net vector_tag --window 5 --epochs 15 --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --regularization --tagdim 100 --plot --hidden "$i"
done
