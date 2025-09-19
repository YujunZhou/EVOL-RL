export export RAY_TMPDIR=/mnt/petrelfs/shengli/tmp/ray

MODEL_PATH=Skywork/Skywork-OR1-Math-7B
DATA_PATH=/mnt/petrelfs/shengli/project/verl/data/r1_bench
ROLLOUT_N=64

cat << 'EOT'
# Eval Data Process
python3 -m recipe.r1.data_process \
    --local_dir $DATA_PATH \
    --tasks aime2024

# Generation
MP_NUM_THREADS=8 srun  --partition=MoE  --mpi=pmi2  --job-name=test  -c 100  --gres=gpu:8 --nodes=1  --ntasks-per-node=1  --kill-on-bad-exit=1  --quotatype=reserved python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=$ROLLOUT_N \
    data.output_path=$DATA_PATH/test-output-$MODEL_PATH-$ROLLOUT_N.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=32768 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536


# Evaluation
MP_NUM_THREADS=8 srun  --partition=MoE  --mpi=pmi2  --job-name=test  -c 100  --gres=gpu:0 --nodes=1  --ntasks-per-node=1  --kill-on-bad-exit=1  --quotatype=reserved python3 -m recipe.r1.main_eval \
    data.path=$DATA_PATH/test-output-$MODEL_PATH-$ROLLOUT_N.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=maj_reward_func \
    custom_reward_function.do_majority_vote=True \
    custom_reward_function.majority_n=64 \

MP_NUM_THREADS=8 srun  --partition=MoE  --mpi=pmi2  --job-name=test  -c 100  --gres=gpu:0 --nodes=1  --ntasks-per-node=1  --kill-on-bad-exit=1  --quotatype=reserved python3 -m recipe.r1.main_eval \
    data.path=$DATA_PATH/test-output-$MODEL_PATH-$ROLLOUT_N.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=maj_reward_func \
    custom_reward_function.do_majority_vote=True \
    custom_reward_function.majority_n=32 \

MP_NUM_THREADS=8 srun  --partition=MoE  --mpi=pmi2  --job-name=test  -c 100  --gres=gpu:0 --nodes=1  --ntasks-per-node=1  --kill-on-bad-exit=1  --quotatype=reserved python3 -m recipe.r1.main_eval \
    data.path=$DATA_PATH/test-output-$MODEL_PATH-$ROLLOUT_N.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=maj_reward_func \
    custom_reward_function.do_majority_vote=True \
    custom_reward_function.majority_n=16 \
EOT

MP_NUM_THREADS=8 srun  --partition=MoE  --mpi=pmi2  --job-name=test  -c 100  --gres=gpu:0 --nodes=1  --ntasks-per-node=1  --kill-on-bad-exit=1  --quotatype=reserved python3 -m recipe.r1.main_eval \
    data.path=$DATA_PATH/test-output-$MODEL_PATH-$ROLLOUT_N.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=avg_reward_func \
    custom_reward_function.do_majority_vote=True \
    custom_reward_function.majority_n=32 \