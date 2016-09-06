#!/bin/bash

function print_region_size {
				echo "[w2v_embeddings]" >> $2
				echo "source: googlenews_representations_train_True_valid_True_test_False.p"  >> $2
				echo "dim: 300"  >> $2
				echo "embedding_item: word"  >> $2
				echo "learning: tune"  >> $2
				echo "" >> $2

				echo "[pos_embeddings]"  >> $2
				echo "source: word_final_embeddings.p"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: word"  >> $2
				echo "learning: tune"  >> $2
				echo "" >> $2

				echo "[ner_embeddings]"  >> $2
				echo "source: probabilistic"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: word"  >> $2
				echo "learning: train"  >> $2
				echo "" >> $2

				echo "[sent_nr_embeddings]"  >> $2
				echo "source: probabilistic"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: tag"  >> $2
				echo "learning: train"  >> $2
				echo "" >> $2

				echo "[tense_embeddings]"  >> $2
				echo "source: probabilistic"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: tag"  >> $2
				echo "learning: train"  >> $2
				echo "" >> $2
				
				echo "[feature_w2v_c_m]"  >> $2
				echo "name: w2v_c_m"  >> $2
				echo "n_filters: 400"  >> $2
				echo "region_sizes: $1"  >> $2
           }

function delete_file {
	sudo rm $1
}

#declare -a region_sizes=("3 5 7" "2 4 7" "5 6 7" "2 3 4 5 6 7")
declare -a region_sizes=("3 5 7")

declare window=7
declare minibatch=32
declare lrtrain=0.01
declare lrtune=0.01
declare epochs=20
declare filters=400

for i in "${region_sizes[@]}"
do
	echo "region_sizes $i"
	delete_file data/params/neural_network/tmp.cfg
	print_region_size "$i" data/params/neural_network/tmp.cfg
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$window" --epochs "$epochs" --plot --lrtrain "$lrtrain" --lrtune "$lrtune" --minibatch "$minibatch" --configini cnn_w2v_c_m.cfg --earlystop 2 --logger earlystop_region_sizes_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window "$window" --epochs "$epochs" --plot --lrtrain "$lrtrain" --lrtune "$lrtune" --minibatch "$minibatch" --configini cnn_w2v_c_m.cfg --picklelists --logger region_sizes_"$i".log
	sudo mv trained_models/tf_cnn/valid_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/valid_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_"$filters"_regions_"$i".p
	sudo mv trained_models/tf_cnn/valid_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/valid_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_"$filters"_regions_"$i".p
	sudo mv trained_models/tf_cnn/train_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/train_cost_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_"$filters"_regions_"$i".p
	sudo mv trained_models/tf_cnn/train_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/train_cross_entropy_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_"$filters"_regions_"$i".p
	sudo mv trained_models/tf_cnn/valid_f1_score_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_None_regions_None.p trained_models/tf_cnn/valid_f1_score_list-"$window"_"$minibatch"_lrtrain_"$lrtrain"_lrtune_"$lrtune"_filters_"$filters"_regions_"$i".p
done