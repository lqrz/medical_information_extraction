#!/bin/bash
declare -a hidden_layer_size=(50 500 1500 3000)

declare window=5
declare token_update="True"
declare sentence_update="False"
declare two_layers="True"
declare tagdim=10
declare epochs=15

#for j in "${arr_type[@]}"
#do
for i in "${hidden_layer_size[@]}"
do
	echo "Training last_tag hidden_layer_size: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window "$window" --hidden "$i" --tagdim "$tagdim" --token --epochs "$epochs" --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --earlystop 2 --logger earlystop_hidden_layer_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net last_tag --window "$window" --hidden "$i" --tagdim "$tagdim" --token --epochs "$epochs" --w2vvectorscache googlenews_representations_train_True_valid_True_test_False.p --plot --lrtrain 0.01 --picklelists --logger hidden_layer_"$i".log
	mkdir trained_models/last_tag/hidden_layer_comparison
	sudo mv trained_models/last_tag/valid_cost_list-"$window"_"$sentence_update"_"$token_update"_"$two_layers"_"$tagdim".p trained_models/last_tag/hidden_layer_comparison/valid_cost_list-"$window"_"$sentence_update"_"$token_update"_"layer"_"$i"_"$tagdim".p
	sudo mv trained_models/last_tag/valid_cross_entropy_list-"$window"_"$sentence_update"_"$token_update"_"$two_layers"_"$tagdim".p trained_models/last_tag/hidden_layer_comparison/valid_cross_entropy_list-"$window"_"$sentence_update"_"$token_update"_"layer"_"$i"_"$tagdim".p
	sudo mv trained_models/last_tag/train_cost_list-"$window"_"$sentence_update"_"$token_update"_"$two_layers"_"$tagdim".p trained_models/last_tag/hidden_layer_comparison/train_cost_list-"$window"_"$sentence_update"_"$token_update"_"layer"_"$i"_"$tagdim".p
	sudo mv trained_models/last_tag/train_cross_entropy_list-"$window"_"$sentence_update"_"$token_update"_"$two_layers"_"$tagdim".p trained_models/last_tag/hidden_layer_comparison/train_cross_entropy_list-"$window"_"$sentence_update"_"$token_update"_"layer"_"$i"_"$tagdim".p
	sudo mv trained_models/last_tag/valid_f1_score_list-"$window"_"$sentence_update"_"$token_update"_"$two_layers"_"$tagdim".p trained_models/last_tag/hidden_layer_comparison/valid_f1_score_list-"$window"_"$sentence_update"_"$token_update"_"layer"_"$i"_"$tagdim".p

done
#done
