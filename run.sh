random_port=$(shuf -i 1000-9999 -n 1)
for dataset in 'c19'
do
for seed in 1 42 3407
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 --master_port $random_port \
    main_pretrain.py --dataset $dataset --seed $seed
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 --master_port $random_port \
    main_finetune.py --dataset $dataset --seed $seed
done
done
