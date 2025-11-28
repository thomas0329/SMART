random_port=$(shuf -i 1000-9999 -n 1)
log_dir="/home/thomas/Desktop/SMART/logs"
mkdir -p "$log_dir"
run_id=$(date +%Y%m%d_%H%M%S)
log_file="${log_dir}/run_${run_id}.log"
for dataset in  'c19'
do
for seed in 1 42 3407
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 --master_port $random_port \
    main_pretrain.py --dataset $dataset --seed $seed >> "$log_file" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 --master_port $random_port \
    main_finetune.py --dataset $dataset --seed $seed >> "$log_file" 2>&1
done
done
