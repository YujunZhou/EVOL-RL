#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
# ------------------------------------------------------------

DATE=$(date +%m%d)

N_SAMPLES_PER_PROMPT=16
N_VOTES_PER_PROMPT=32
DATA_TRAIN_BATCH_SIZE=256
MINI_BATCH_SIZE=16 # Actual mini batch size is MINI_BATCH_SIZE * NN_SAMPLES_PER_PROMPT
ADVANTAGE="rloo"
TASK="MATH-TTT"
DATA_LOCAL_DIR=/root/workdir/verl/data/
BACKBONE="Qwen2.5-Math-1.5B"

MODEL="${TASK}-${BACKBONE}"
BACKBONE_PATH="/cpfs04/user/shengli/model/${BACKBONE}"

EXPERIMENT="TTRL-Maj@${N_VOTES_PER_PROMPT}-Roll@${N_SAMPLES_PER_PROMPT}-Temp@1.0-Batch@${DATA_TRAIN_BATCH_SIZE}"
EXP="${DATE}-${TASK}-${EXPERIMENT}-${BACKBONE}-PRIME-${ADVANTAGE}"
# ------------------------------------------------------------
python3 -m recipe.prime.main_prime \
    reward_model.reward_manager=ttrl \
    reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
    reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
    reward_model.reward_kwargs.mode="train" \
    reward_model.model.path=$BACKBONE_PATH \
    reward_model.micro_batch_size_per_gpu=4 \
    reward_model.model.update=after \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=256 \
    data.train_files=["$DATA_LOCAL_DIR/MATH-TTT/train.parquet"] \
    data.val_files=["$DATA_LOCAL_DIR/AIME-TTT/test.parquet","$DATA_LOCAL_DIR/MATH-TTT/test.parquet","$DATA_LOCAL_DIR/AIME25-TTT/test.parquet","$DATA_LOCAL_DIR/AMC-TTT/test.parquet"] \
    data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.path=$BACKBONE_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.do_vote=True \
    actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
    actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=9e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BACKBONE_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.00 \
    algorithm.adv_estimator=$ADVANTAGE \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="TTRL-verl-dev" \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    trainer.total_epochs=40 $@