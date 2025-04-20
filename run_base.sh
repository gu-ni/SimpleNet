JSON_PATH="/workspace/meta_files/base_classes.json"
SAVE_CKPT_DIR="/workspace/MegaInspection/SimpleNet/checkpoints/base"

# 실행
python main.py \
    --gpu 0 \
    --seed 0 \
    --log_group simplenet_continual \
    --log_project ContinualAD \
    --results_path results \
    --run_name base_model \
    --save_checkpoint $SAVE_CKPT_DIR \
    net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 50 \
    --embedding_size 256 \
    --gan_epochs 1 \
    --noise_std 0.015 \
    --dsc_hidden 1024 \
    --dsc_layers 2 \
    --dsc_margin .5 \
    --pre_proj 1 \
    dataset \
    --json_path $JSON_PATH \
    --batch_size 8 \
    --num_workers 8 \
    --resize 336 \
    --imagesize 336