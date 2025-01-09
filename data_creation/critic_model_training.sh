export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONWARNINGS="ignore"
torchrun --nproc_per_node=1 train_special_tokens.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --data_path ../../../../dataspace/P76124574/SELF-RAG/training_data/gpt4_reward_all_0813_train.json \
  --bf16  True \
  --output_dir ../model/critic_model \
  --num_train_epochs 3  \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --fsdp "full_shard auto_wrap"