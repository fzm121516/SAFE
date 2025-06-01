
GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL="SAFE"
RESUME_PATH="./checkpoint"

eval_datasets=(
    "/root/zgp2/fanzheming/LVLM/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test" \
)
for eval_dataset in "${eval_datasets[@]}"
do
    python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
        --input_size 256 \
        --transform_mode 'crop' \
        --model $MODEL \
        --eval_data_path $eval_dataset \
        --batch_size 32 \
        --num_workers 1 \
        --output_dir $RESUME_PATH \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True
done