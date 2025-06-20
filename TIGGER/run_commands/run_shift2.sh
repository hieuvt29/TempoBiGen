
# run shift data with 2 weeks, 10 different units

export num_epochs=1
export model_name='best'
export dataset_name="shift2.0_10"
export data_path="../DATA/processed_data/$dataset_name/all_unittypes_refined.csv"
export config_path="results/${dataset_name}"
export sample_rate=10

export embedding_dim=256
export nb_lstm_units=512
export nb_layers=3
export time_emb_dim=64
export LR=0.001
export seed=1

export num_of_sampled_graphs=1
export eval_time_window=86400 # 1 day

CUDA_VISIBLE_DEVICES=0,1 time python train_transductive.py --dataset_name $dataset_name --data_path $data_path \
    --embedding_dim $embedding_dim --nb_lstm_units $nb_lstm_units \
    --nb_layers $nb_layers --time_emb_dim $time_emb_dim \
    --config_path "${config_path}" --num_epochs $num_epochs \
    --l_w 20 --batch_size 512 --patience 50 --lr $LR --seed $seed  &&

python test_transductive.py --gpu_num 2 --dataset_name $dataset_name --data_path $data_path \
    --embedding_dim $embedding_dim --nb_lstm_units $nb_lstm_units \
    --nb_layers $nb_layers --time_emb_dim $time_emb_dim \
    --config_path $config_path --model_name $model_name --random_walk_sampling_rate $sample_rate \
    --num_of_sampled_graphs $num_of_sampled_graphs --l_w 20 --batch_size 512  &&

python graph_generation_from_sampled_random_walks.py --data_path $data_path --config_path $config_path \
    --num_of_sampled_graphs $num_of_sampled_graphs \
    --time_window $eval_time_window --topk_edge_sampling 1 --l_w 20 --model_name $model_name &&


for ((i=0; i<$num_of_sampled_graphs; i++)); do
    python graph_metrics.py \
        --op results_$model_name/original_graphs.pkl \
        --sp results_$model_name/sampled_graph_${i}.pkl \
        --config_path $config_path &&

    python ../convert_gen_graph_to_dataobj.py --time_window $eval_time_window \
        --opath $config_path/results_$model_name/original_graphs.pkl \
        --gpath $config_path/results_$model_name/sampled_graph_${i}.pkl \
        --label_path $config_path/graph_label_to_id.pkl 
done
