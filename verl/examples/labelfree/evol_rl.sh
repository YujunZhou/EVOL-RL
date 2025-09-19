#!/bin/bash

# === Semantic Novelty TTRL Training Script ===
# Usage: ./evol_rl.sh [--task TASK] [--backbone BACKBONE] [--clip-high] [--temp TEMP]
#
# Parameters:
#   --task      Task name (default: AIME-TTT)
#               Options: AIME-TTT, MATH-TTT, AIME25, etc.
#   --backbone  Backbone model (default: Qwen3-4B-Base)
#               Options: Qwen3-4B-Base, Qwen3-8B-Base
#   --clip-high[=VAL]
#               Clip ratio control:
#               - Not specified: use default 0.2
#               - Specified without value: use 0.28
#               - Specified with value: use that value
#   --ent       Entropy regularization coefficient (float), e.g. 0.000/0.001/0.003 (default: 0.000)
#   --temp      Temperature parameter (default: 1.0)
#               Controls generation randomness, typically range 0.1-2.0
#   -h, --help  Show help information
#
# Examples:
#   ./evol_rl.sh                                    # Use default parameters
#   ./evol_rl.sh --task MATH                   # Specify task
#   ./evol_rl.sh --task AIME --backbone Qwen3-4B-Base  # Specify task and model
#   ./evol_rl.sh --clip-high                       # High clip ratio mode
#   ./evol_rl.sh --temp 0.8                        # Set temperature parameter
#
# Note: This script uses semantic novelty functionality, requires starting local vLLM embedding service
# =======================

#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --clip-high)
            CLIP_HIGH="true"
            CLIP_SPECIFIED="true"
            CLIP_MODE="high"
            if [[ -n "$2" && "$2" != --* ]]; then
              CLIP_VALUE="$2"
              shift 2
            else
              shift 1
            fi
            ;;
        --clip-high=*)
            CLIP_HIGH="true"
            CLIP_SPECIFIED="true"
            CLIP_MODE="high"
            CLIP_VALUE="${1#--clip-high=}"
            shift 1
            ;;
        --ent)
            # Read entropy regularization coefficient; use default if no value provided
            if [[ -z "$2" || "$2" == --* ]]; then
              shift 1
            else
              ENT="$2"
              shift 2
            fi
            ;;
        --ent=*)
            ENT="${1#--ent=}"
            shift 1
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --temp=*)
            TEMP="${1#--temp=}"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [--task TASK] [--backbone BACKBONE] [--clip-high[=VAL]] [--ent COEFF] [--temp TEMP]"
            echo "  --task      Task name (default: AIME)"
            echo "  --backbone  Backbone model (default: Qwen3-4B-Base)"
            echo "  --clip-high[=VAL] set clip ratio: not specified=0.2; flag only=0.28; with value use that value"
            echo "  --ent       Entropy regularization coefficient (float), e.g. 0.000/0.001/0.005 (default: 0.000)"
            echo "  --temp      Temperature parameter (float), e.g. 0.6/0.8/1.0 (default: 1.0)"
            echo "  -h, --help  Show help information"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Set default values
TASK=${TASK:-"AIME"}
BACKBONE=${BACKBONE:-"Qwen3-4B-Base"}
CLIP_HIGH=${CLIP_HIGH:-"false"}
CLIP_SPECIFIED=${CLIP_SPECIFIED:-"false"}
CLIP_VALUE=${CLIP_VALUE:-""}
CLIP_MODE=${CLIP_MODE:-""}
ENT=${ENT:-"0.000"}
TEMP=${TEMP:-"1.0"}

## ENT as numerical coefficient, no longer perform boolean normalization



# Set entropy coefficient (numerical) based on --ent
ENTROPY_COEFF=$ENT
RAW_TASK="$TASK"
if [ "$RAW_TASK" = "math_train" ]; then
  TASK="MATH-TTT"
else
  TASK="$TASK-TTT"
fi

# === Using Remote vLLM Embedding API ===
# Remote vLLM API configuration
export VLLM_API_URL=""
export VLLM_MODEL_NAME="Qwen/Qwen3-Embedding-4B"

# Check remote vLLM API configuration
if [ -z "$VLLM_API_URL" ] || [ -z "$VLLM_MODEL_NAME" ]; then
    echo "❌ Error: Please set remote vLLM API configuration first"
    echo "Please check the following environment variables:"
    echo "- VLLM_API_URL"
    echo "- VLLM_MODEL_NAME"
    exit 1
fi

pkill -f "python.*main_ppo" || true
pkill -f "python.*main_dapo" || true
pkill -f "multiprocessing.spawn" || true
pkill -f "test_three_datasets.sh" || true
pkill -f "python.*scripts.model_merger" || true
ray stop --force 2>/dev/null || true
sleep 2
echo "========================="

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

ADVANTAGE="grpo"

echo "=== Basic Configuration Information ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"
echo "Advantage estimator: $ADVANTAGE"
echo "====================================="

# Set K value
K=12
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * $K))
# Pre-calculate required values to avoid type errors - use arithmetic expansion to ensure numerical type
MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKEN_LEN2=$((MAX_TOKEN_LEN * 2))
if [ "$K" -gt 13 ]; then
  N=4
else
  N=16
fi

# Set EPISODE
EPISODE=3
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=64 # Reduce candidates to balance computational overhead
N_SAMPLES_PER_PROMPT=32 # Keep training sample count
MINI_BATCH_SIZE=1 # Actual mini batch size is MINI_BATCH_SIZE * N_SAMPLES_PER_PROMPT - increase mini batch
MICRO_BATCH_SIZE=2 # Increase micro batch to better utilize GPU

DATA_LOCAL_DIR="data"
# Parse backbone model path and safe name (avoid directory names containing slashes)
CHAT_TEMPLATE=""
if [[ "$BACKBONE" == *"/"* ]]; then
  BACKBONE_PATH="$BACKBONE"
  BACKBONE_NAME="${BACKBONE##*/}"
else
  BACKBONE_PATH="Qwen/${BACKBONE}"
  BACKBONE_NAME="$BACKBONE"
fi

echo "Parsed model path: $BACKBONE_PATH"
echo "Parsed model name: $BACKBONE_NAME"

MODEL="${TASK}-${BACKBONE_NAME}"

# === Semantic Novelty Parameter Configuration ===
USE_SEMANTIC_NOVELTY=true  # Enable semantic novelty
EMBEDDING_DIM=2560  # Modified to local vLLM model dimension

EXPERIMENT="EVOL-RL"

# Set clip_ratio_high value and experiment name suffix
if [ "$CLIP_SPECIFIED" = "true" ]; then
  if [ -n "$CLIP_VALUE" ]; then
    CLIP_RATIO_HIGH=$CLIP_VALUE
  else
    CLIP_RATIO_HIGH=0.28
  fi
  if [ "$CLIP_HIGH" = "true" ]; then
    EXPERIMENT="${EXPERIMENT}-ClipHigh"
  fi
else
  CLIP_RATIO_HIGH=0.2
fi

# if RAW_TASK is math_train, use our preprocessed parquet; else follow original logic
if [ "$RAW_TASK" = "math_train" ]; then
  TRAIN_FILES="math_train_ttrl.parquet"
else
  if [[ "$TASK" == *"AIME"* ]]; then
    TRAIN_FILES="train-simplerl-16.parquet"
  else
    TRAIN_FILES="train-simplerl.parquet"
  fi
fi

# === Remote vLLM API Parameters ===
VLLM_API_URL_VAR="$VLLM_API_URL"  # Remote vLLM API address
VLLM_MODEL_NAME_VAR="$VLLM_MODEL_NAME"  # Remote vLLM model name

MAX_RETRIES=5
RETRY_DELAY=2.0

# Set WANDB_PROJECT based on TASK
if [ "$RAW_TASK" = "math_train" ]; then
  WANDB_PROJECT="TTRL_MATH_TRAIN"
  EXPERIMENT="${EXPERIMENT}-MATH_TRAIN"
elif [ "$TASK" = "AIME-TTT" ]; then
  WANDB_PROJECT="TTRL-AIME24"
else
  WANDB_PROJECT="TTRL-MATH500"
fi


if [ "$CLIP_HIGH" = "true" ]; then
  EXPERIMENT="${EXPERIMENT}-ClipHigh"
fi

# Always include specific entropy coefficient in experiment name for ablation tracking
EXPERIMENT="${EXPERIMENT}-Ent${ENTROPY_COEFF}"


LOG_NAME="${EXPERIMENT}-${MODEL}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${EXPERIMENT}"



echo "=== Semantic Novelty TTRL Training Configuration ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE"
echo "Advantage estimator: $ADVANTAGE"
echo "Use semantic novelty: $USE_SEMANTIC_NOVELTY"
echo "Remote vLLM API address: $VLLM_API_URL_VAR"
echo "Remote vLLM model name: $VLLM_MODEL_NAME_VAR"
echo "Embedding dimension: $EMBEDDING_DIM"
# Print entropy regularization switch (based on whether coefficient is 0)
if [[ "$ENTROPY_COEFF" != "0" && "$ENTROPY_COEFF" != "0.0" && "$ENTROPY_COEFF" != "0.00" && "$ENTROPY_COEFF" != "0.000" ]]; then
  ENT_ENABLED="true"
else
  ENT_ENABLED="false"
fi
echo "Enable entropy regularization: $ENT_ENABLED"
echo "Entropy coefficient: $ENTROPY_COEFF"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $LOG_NAME"
echo "==============================="

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=semantic_ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  +reward_model.reward_kwargs.use_semantic_novelty=$USE_SEMANTIC_NOVELTY \
  +reward_model.reward_kwargs.embedding_dim=$EMBEDDING_DIM \
  +reward_model.reward_kwargs.vllm_api_url=$VLLM_API_URL_VAR \
  +reward_model.reward_kwargs.vllm_model_name=$VLLM_MODEL_NAME_VAR \
  +reward_model.reward_kwargs.max_retries=$MAX_RETRIES \
  +reward_model.reward_kwargs.retry_delay=$RETRY_DELAY \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/$TRAIN_FILES"] \
  data.val_files=["$DATA_LOCAL_DIR/AIME-TTT/test-simplerl.parquet","$DATA_LOCAL_DIR/MATH-TTT/test-simplerl.parquet","$DATA_LOCAL_DIR/AIME25/test-simplerl.parquet","$DATA_LOCAL_DIR/GPQA-TTT/test-simplerl.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_TOKEN_LEN2)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_TOKEN_LEN)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_TOKEN_LEN2)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=60 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "=== Training Completed ==="
echo "Output directory: $OUTPUT_DIR"
echo "Project name: $WANDB_PROJECT"
echo "Experiment name: $LOG_NAME"
echo "========================"

# === Automatic Evaluation Module ===
echo ""
echo "🚀 Starting automatic evaluation process..."
echo "========================================="

# 1. Detect latest checkpoint
echo "📁 Detecting latest checkpoint..."
LATEST_CHECKPOINT=""
if [ -d "$OUTPUT_DIR" ]; then
    # Find latest global_step directory
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "global_step_*" -type d | sort -V | tail -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "✅ Found latest checkpoint: $LATEST_CHECKPOINT"
    else
        echo "❌ No checkpoint found, skipping evaluation"
        exit 0
    fi
else
    echo "❌ Output directory does not exist, skipping evaluation"
    exit 0
fi

# 2. Build model merge parameters
echo "🔧 Building model merge parameters..."
ACTOR_DIR="${LATEST_CHECKPOINT}/actor"
HF_MODEL_PATH="$BACKBONE_PATH"

# Build target directory name
TARGET_MODEL_NAME="${MODEL}-${EXPERIMENT}"
TARGET_DIR="models/${TARGET_MODEL_NAME}"
# Remove slashes from model name when uploading to HF
SANITIZED_TARGET_MODEL_NAME="${TARGET_MODEL_NAME//\//-}"
HF_UPLOAD_PATH="username/${SANITIZED_TARGET_MODEL_NAME}"

echo "Model merge configuration:"
echo "  - Local directory: $ACTOR_DIR"
echo "  - HF model path: $HF_MODEL_PATH"
echo "  - Target directory: $TARGET_DIR"
echo "  - HF upload path: $HF_UPLOAD_PATH"

# 3. Execute model merge
echo ""
echo "🔄 Starting model merge..."
python -m scripts.model_merger \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --hf_model_path "$HF_MODEL_PATH" \
    --target_dir "$TARGET_DIR" 
# python -m scripts.model_merger \
#     --backend fsdp \
#     --local_dir "$ACTOR_DIR" \
#     --hf_model_path "$HF_MODEL_PATH" \
#     --target_dir "$TARGET_DIR" \
#     --hf_upload_path "$HF_UPLOAD_PATH"
MERGE_OK=0
if [ -d "$TARGET_DIR" ]; then
  if compgen -G "$TARGET_DIR/*.safetensors" > /dev/null || [ -f "$TARGET_DIR/tokenizer.json" ]; then
    MERGE_OK=1
  fi
fi
if [ $MERGE_OK -eq 0 ]; then
    echo "✅ Model merge successful"
else
    echo "❌ Model merge failed, skipping evaluation"
    exit 1
fi

# 4. Execute automatic evaluation
echo ""
echo "🧪 Starting automatic evaluation..."
echo "Evaluating model: $TARGET_DIR"

# Call test script
./test_three_datasets.sh --model_path "$TARGET_DIR" --backbone "$BACKBONE"

if [ $? -eq 0 ]; then
    echo "✅ Automatic evaluation completed"
else
    echo "❌ Automatic evaluation failed"
    exit 1
fi

