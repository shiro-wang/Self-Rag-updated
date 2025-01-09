export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=1B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --use_flash_attn \
    --use_slow_tokenizer \
    --tokenizer_name meta-llama/Llama-3.2-1B-Instruct \
    --train_file ../../../../dataspace/P76124574/SELF-RAG/training_data/full_output_1005.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir ../../../../dataspace/P76124574/SELF-RAG/model/generator/self_rag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
    --logging_name 1b_tuning