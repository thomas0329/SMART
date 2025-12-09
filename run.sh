random_port=$(shuf -i 1000-9999 -n 1)
for dataset in c12 c19
do
for seed in 1 42 3407
do
python main_pretrain.py --dataset $dataset --seed $seed --local-rank 0 --batch_size 256 --epochs 100
python main_finetune.py --dataset $dataset --seed $seed --batch_size 256 --freeze_epochs 5
done
done
