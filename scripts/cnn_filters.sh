#!/bin/bash

function print_region_size {
				echo "[w2v_embeddings]" >> $2
				echo "source: googlenews_representations_train_True_valid_True_test_False.p"  >> $2
				echo "dim: 300"  >> $2
				echo "embedding_item: word"  >> $2
				echo "learning: tune"  >> $2

				echo "[pos_embeddings]"  >> $2
				echo "source: word_final_embeddings.p"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: word"  >> $2
				echo "learning: tune"  >> $2

				echo "[ner_embeddings]"  >> $2
				echo "source: probabilistic"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: word"  >> $2
				echo "learning: train"  >> $2

				echo "[sent_nr_embeddings]"  >> $2
				echo "source: probabilistic"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: tag"  >> $2
				echo "learning: train"  >> $2

				echo "[tense_embeddings]"  >> $2
				echo "source: probabilistic"  >> $2
				echo "dim: 100"  >> $2
				echo "embedding_item: tag"  >> $2
				echo "learning: train"  >> $2

				echo "[feature_w2v_c_m]"  >> $2
				echo "name: w2v_c_m"  >> $2
				echo "n_filters: $1"  >> $2
				echo "region_sizes: 1 3 5 7"  >> $2
           }

function delete_file {
	sudo rm $1
}

declare -a n_filters=(10 25 50 100 400)

for i in "${n_filters[@]}"
do
	echo "Training last_tag n_filters: $i"
	delete_file data/params/neural_network/tmp.cfg
	print_region_size "$i" data/params/neural_network/tmp.cfg
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window 7 --epochs 20 --minibatch 256 --plot --lrtrain 0.01 --lrtune 0.01 --configini tmp.cfg --earlystop 2 --logger earlystop_filters_"$i".log
	stdbuf -oL sudo python SOTA/neural_network/train_neural_network.py --net tf_cnn --window 7 --epochs 20 --minibatch 256 --plot --lrtrain 0.01 --lrtune 0.01 --configini tmp.cfg --picklelists --logger filters_"$i".log
done