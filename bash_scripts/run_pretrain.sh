# pretrains i3d model for mixamo to kinetics
# mixamo -> kinetics
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../pretrain_main.py \
    --source_dataset /data/datasets/mixamo_datasets/mixamo14_larger \
    --val_dataset /data/datasets/mixamo_datasets/mixamo14_larger \
    --name mixamo-larger-pretrain \
    --project CO2A_2 \
    --epochs 100 \
    --bottleneck_size 256 \
    --optimizer sgd \
    --lr 0.01 \
    --weight_decay 1e-9 \
    --scheduler cosine \
    --n_frames 16 \
    --n_clips 4 \
    --frame_size 224 \
    --imagenet_pretrained \
    --aggregation mlp_weights \
    --batch_size 10 \
    --num_workers 8 \
    --gpus 0 1 2 3 \
    --distributed_backend ddp \
    --wandb \
    --layers 2 \
    --layers_ca 1
