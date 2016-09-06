#!/bin/bash
declare -a hidden_layer_size=(50 500 1200 3000)

declare window=7
declare epochs=20
declare minibatch=32
declare lrtrain=0.01
declare lrtune=0.01

#for j in "${arr_type[@]}"
#do
for i in "${hidden_layer_size[@]}"
do
	echo "Training cnn hidden_layer_size: $i"
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$window" --epochs "$epochs" --minibatch "$minibatch" --plot --lrtrain "$lrtrain" --lrtune "$lrtune" --configini cnn_w2v_c_m.cfg --hidden "$i" --earlystop 2 --logger earlystop_hidden_layer_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$window" --epochs "$epochs" --minibatch "$minibatch" --plot --lrtrain "$lrtrain" --lrtune "$lrtune" --configini cnn_w2v_c_m.cfg --hidden "$i" --picklelists --logger hidden_layer_"$i".log
	mkdir trained_models/tf_cnn/hidden_layer_comparison
	sudo mv trained_models/tf_cnn/valid_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/hidden_layer_comparison/valid_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None_"$i".p
	sudo mv trained_models/tf_cnn/valid_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/hidden_layer_comparison/valid_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None_"$i".p
	sudo mv trained_models/tf_cnn/train_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/hidden_layer_comparison/train_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None_"$i".p
	sudo mv trained_models/tf_cnn/train_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/hidden_layer_comparison/train_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None_"$i".p
	sudo mv trained_models/tf_cnn/valid_f1_score_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/hidden_layer_comparison/valid_f1_score_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None_"$i".p

done
#done
