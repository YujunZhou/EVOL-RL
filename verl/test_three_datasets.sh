#!/bin/bash

# === TTRL Batch Model Evaluation Script ===
# Usage: ./test_three_datasets.sh [--model_path MODEL_PATH] [--backbone BACKBONE] [--datasets DATASETS] [--batch_mode]
#
# Single model mode parameters:
#   --model_path Trained model path (required)
#   --backbone   Backbone model name (default: auto-detect)
#   --datasets   List of datasets to test, comma-separated (default: AIME-TTT,MATH-TTT,AIME25,AMC-TTT,GPQA-TTT)
#   -h, --help   Show help information
#
# Batch mode parameters:
#   --batch_mode Enable batch evaluation mode, evaluate predefined model list
#   --datasets   List of datasets to test, comma-separated (default: AIME-TTT,MATH-TTT,AIME25,AMC-TTT,GPQA-TTT)
#
# Default test datasets:
#   - AIME-TTT/test-simplerl.parquet
#   - MATH-TTT/test-simplerl.parquet
#   - AIME25/test-simplerl.parquet
#   - AMC-TTT/test-simplerl.parquet
#   - GPQA-TTT/test-simplerl.parquet
#
# Features:
#   - Complete evaluation following training validation approach
#   - Generate 32 samples, calculate pass@1, maj@16, pass@16 metrics
#   - Use temperature=0 to ensure deterministic results
#   - Support cumulative results: multiple runs will merge different dataset results
#   - Support batch evaluation of multiple models
#   - Auto-detect backbone type
#
# Examples:
#   # Single model evaluation
#   ./test_three_datasets.sh --model_path checkpoints/my_model
#   ./test_three_datasets.sh --model_path checkpoints/my_model --backbone Qwen3-4B-Base
#
#   # Batch evaluation of all predefined models
#   ./test_three_datasets.sh --batch_mode
#   ./test_three_datasets.sh --batch_mode --datasets "GSM8K,MATH,CodeContests"
# =======================

#export VLLM_ATTENTION_BACKEND=XFORMERS
# ps aux | grep gpu_new.py | grep -v taijirun | awk '{print $2}' | xargs kill

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1


PREDEFINED_MODELS=(
""
)

PREDEFINED_MODELS2=(
    ""
)

PREDEFINED_MODELS3=(
    ""
)


# Function to auto-detect backbone
detect_backbone() {
    local model_path="$1"
    if [[ "$model_path" == *"Qwen3-8B-Base"* ]]; then
        echo "Qwen3-8B-Base"
    elif [[ "$model_path" == *"Qwen3-4B-Base"* ]]; then
        echo "Qwen3-4B-Base"
    else
        echo "Qwen3-4B-Base"  # Default value
    fi
}

# Parse command line arguments
MODEL_PATH=""
DATASETS="AIME-TTT,MATH-TTT,AIME25,AMC-TTT,GPQA-TTT"  # Default datasets
BATCH_MODE=false
PREDEFINED_SET=1  # Select predefined model set: 1=PREDEFINED_MODELS(default), 2=PREDEFINED_MODELS2, 3=PREDEFINED_MODELS3

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --batch_mode)
            BATCH_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--model_path MODEL_PATH] [--backbone BACKBONE] [--datasets DATASETS] [--batch_mode] [--set {1|2|3}]"
            echo ""
            echo "Single model mode:"
            echo "  --model_path Trained model path (required)"
            echo "  --backbone   Backbone model name (default: auto-detect)"
            echo "  --datasets   List of datasets to test, comma-separated (default: AIME-TTT,MATH-TTT,AIME25,AMC-TTT,GPQA-TTT)"
            echo ""
            echo "Batch mode:"
            echo "  --batch_mode Enable batch evaluation mode, evaluate predefined model list"
            echo "  --datasets   List of datasets to test, comma-separated (default: AIME-TTT,MATH-TTT,AIME25,AMC-TTT,GPQA-TTT)"
            echo "  --set Select predefined model set: 1=PREDEFINED_MODELS(default), 2=PREDEFINED_MODELS2, 3=PREDEFINED_MODELS3"
            echo ""
            echo "Examples:"
            echo "  # Single model evaluation"
            echo "  $0 --model_path checkpoints/my_model"
            echo "  $0 --model_path checkpoints/my_model --backbone Qwen3-4B-Base"
            echo ""
            echo "  # Batch evaluation"
            echo "  $0 --batch_mode --set 2"
            echo "  $0 --batch_mode --datasets \"GSM8K,MATH,CodeContests\""
            exit 0
            ;;
        --set)
            PREDEFINED_SET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check parameters
if [ "$BATCH_MODE" = false ]; then
    # Single model mode: check required parameters
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Single model mode must specify --model_path parameter"
        echo "Usage: $0 --model_path MODEL_PATH [--backbone BACKBONE]"
        echo "Or use batch mode: $0 --batch_mode"
        exit 1
    fi

    # Auto-detect backbone (if not specified)
    if [ -z "$BACKBONE" ]; then
        BACKBONE=$(detect_backbone "$MODEL_PATH")
        echo "Auto-detected backbone: $BACKBONE"
    fi
else
    # Batch mode: display models to be evaluated
    echo "=== Batch Evaluation Mode ==="

    # Select model array based on PREDEFINED_SET
    if [ "$PREDEFINED_SET" = "2" ]; then
        SELECTED_MODELS=("${PREDEFINED_MODELS2[@]}")
    elif [ "$PREDEFINED_SET" = "3" ]; then
        SELECTED_MODELS=("${PREDEFINED_MODELS3[@]}")
    else
        SELECTED_MODELS=("${PREDEFINED_MODELS[@]}")
    fi

    echo "Using predefined model set: $PREDEFINED_SET (${#SELECTED_MODELS[@]} models total)"
    for i in "${!SELECTED_MODELS[@]}"; do
        model="${SELECTED_MODELS[$i]}"
        backbone=$(detect_backbone "$model")
        echo "  $((i+1)). $model (backbone: $backbone)"
    done
    echo "====================="
fi

# ------------------------------------------------------------

# Function to check if dataset has been evaluated
check_dataset_evaluated() {
    local results_file="$1"
    local dataset="$2"

    if [ ! -f "$results_file" ]; then
        return 1  # File doesn't exist, not evaluated
    fi

    # Use Python to check if the dataset is included in the JSON file
    python3 -c "
import json
import sys

try:
    with open('$results_file', 'r', encoding='utf-8') as f:
        data = json.load(f)

    datasets = data.get('datasets', {})
    if '$dataset' in datasets:
        print('True')
    else:
        print('False')
except Exception as e:
    print('False')
" 2>/dev/null | grep -q "True"

    return $?
}

# Clean up local cache for predefined models (Hugging Face cache)
cleanup_predefined_model_cache() {
    local repo_id="$1"

    # Only clean up remote repository IDs; return directly if it's a local path
    if [ -d "$repo_id" ] || [[ "$repo_id" == /* || "$repo_id" == ./* ]]; then
        return 0
    fi

    echo "\n🧹 Cleaning up model cache: $repo_id"

    # Try huggingface-cli first (if available), delete cache by repository ID
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli delete-cache --yes --repo-id "$repo_id" 2>/dev/null || true
    fi

    # Fallback: directly delete HF local cache directory
    local HF_CACHE_ROOT="$HF_HOME/hub"
    local MODEL_CACHE_DIR="${HF_CACHE_ROOT}/models--${repo_id//\//--}"
    if [ -d "$MODEL_CACHE_DIR" ]; then
        echo "Deleting directory: $MODEL_CACHE_DIR"
        rm -rf "$MODEL_CACHE_DIR" || true
    else
        echo "Cache directory not found (may have been cleaned or using different cache path): $MODEL_CACHE_DIR"
    fi
}

# Single model evaluation function
evaluate_single_model() {
    local model_path="$1"
    local backbone="$2"
    local datasets="$3"

    # Extract model name as result file name
    local model_name=$(basename "$model_path")
    # If model path contains checkpoints directory, extract more meaningful name
    if [[ "$model_path" == *"checkpoints"* ]]; then
        # Extract model name from checkpoints path
        model_name=$(echo "$model_path" | sed 's/.*checkpoints\///' | sed 's/\/.*$//')
    fi

    # Convert dataset string to array
    IFS=',' read -ra dataset_array <<< "$datasets"

    # Check already evaluated datasets
    local OUTPUT_DIR="eval_results"
    local RESULTS_FILE="$OUTPUT_DIR/${model_name}_results.json"
    local datasets_to_evaluate=()

    local skipped_datasets=()

    echo ""
    echo "🎯 Starting model evaluation: $model_name"
    echo "Model path: $model_path"
    echo "Backbone model: $backbone"
    echo "Test datasets: $datasets"
    echo "Data directory: ./data"
    echo "Sample count: 32 (calculate pass@1, maj@16, pass@16)"
    echo "Support cumulative results: Yes"
    echo "========================="

    # Check if each dataset has already been evaluated
    echo "📋 Checking already evaluated datasets..."
    for dataset in "${dataset_array[@]}"; do
        if check_dataset_evaluated "$RESULTS_FILE" "$dataset"; then
            skipped_datasets+=("$dataset")
            echo "  ⏭️  Skipping already evaluated dataset: $dataset"
        else
            datasets_to_evaluate+=("$dataset")
            echo "  ✅ Need to evaluate dataset: $dataset"
        fi
    done

    # If no datasets need evaluation, return directly
    if [ ${#datasets_to_evaluate[@]} -eq 0 ]; then
        echo ""
        echo "🎉 All datasets have been evaluated, skipping model: $model_name"
        echo "Already evaluated datasets: ${skipped_datasets[*]}"
        return 0
    fi

    # Rebuild dataset string for evaluation
    local datasets_string=$(IFS=','; echo "${datasets_to_evaluate[*]}")
    echo ""
    echo "📊 Evaluation statistics:"
    echo "  Total datasets: ${#dataset_array[@]}"
    echo "  Already evaluated: ${#skipped_datasets[@]} (${skipped_datasets[*]})"
    echo "  To be evaluated: ${#datasets_to_evaluate[@]} (${datasets_string})"
    echo "========================="

        # Set parameters based on model
        local K=12

    local MAX_PROMPT_LENGTH=1024
    local MAX_RESPONSE_LENGTH=$((1024 * K))
    local MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    local MAX_TOKEN_LEN2=$((MAX_TOKEN_LEN * 2))

    # Follow training validation settings
    local N_SAMPLES=32  # Generate 32 samples for metric calculation

    local DATA_LOCAL_DIR="./data"
    local OUTPUT_DIR="eval_results"
    mkdir -p "$OUTPUT_DIR"

    echo "Results will be saved as: $OUTPUT_DIR/${model_name}_results.json"

    # Use training validation approach for fast evaluation
    local LOG_FILE="$OUTPUT_DIR/${model_name}_evaluation.log"
    local RESULTS_FILE="$OUTPUT_DIR/${model_name}_results.json"

    echo "Running fast evaluation..."

    # Use existing training data file
    local TRAIN_FILE="$DATA_LOCAL_DIR/AIME-TTT/train-simplerl.parquet"

    # Check if training file exists
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "❌ Training file does not exist: $TRAIN_FILE"
        return 1
    fi

    echo "Using training file: $TRAIN_FILE"

    python -m verl.trainer.main_ppo \
      reward_model.reward_manager=ttrl \
      reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES \
      reward_model.reward_kwargs.n_votes_per_prompt=$N_SAMPLES \
      reward_model.reward_kwargs.mode="eval" \
      data.train_files=["$TRAIN_FILE"] \
      data.val_files=[$(printf "\"%s/test-simplerl.parquet\"," "${datasets_to_evaluate[@]/#/$DATA_LOCAL_DIR/}" | sed 's/,$//')]  \
      data.max_prompt_length=$MAX_PROMPT_LENGTH \
      data.max_response_length=$MAX_RESPONSE_LENGTH \
      data.train_batch_size=1 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      actor_rollout_ref.model.path=$model_path \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=1 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.entropy_coeff=0.000 \
      actor_rollout_ref.actor.optim.lr=5e-7 \
      actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
      actor_rollout_ref.actor.optim.warmup_style='cosine' \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_TOKEN_LEN2)) \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.temperature=1.0 \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
      actor_rollout_ref.rollout.do_vote=True \
      actor_rollout_ref.rollout.n_vote=$N_SAMPLES \
      actor_rollout_ref.rollout.n=$N_SAMPLES \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.val_kwargs.n=$N_SAMPLES \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
      actor_rollout_ref.rollout.val_kwargs.top_k=20 \
      actor_rollout_ref.rollout.max_model_len=$((MAX_TOKEN_LEN)) \
      actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_TOKEN_LEN2)) \
      critic.optim.lr=9e-6 \
      critic.model.use_remove_padding=True \
      critic.model.path=$model_path \
      critic.model.enable_gradient_checkpointing=True \
      critic.ppo_micro_batch_size_per_gpu=2 \
      critic.model.fsdp_config.param_offload=False \
      critic.model.fsdp_config.optimizer_offload=False \
      algorithm.kl_ctrl.kl_coef=0.00 \
      algorithm.adv_estimator=grpo \
      trainer.logger=['console'] \
      trainer.resume_mode=auto \
      trainer.project_name="TTRL-FastEval" \
      trainer.experiment_name="FastEval-${model_name}" \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.save_freq=2000000 \
      trainer.test_freq=5 \
      trainer.max_actor_ckpt_to_keep=0 \
      trainer.max_critic_ckpt_to_keep=0 \
      trainer.default_local_dir=$OUTPUT_DIR \
      trainer.total_epochs=0 2>&1 | tee "$LOG_FILE"

    # Evaluation completed (no need to clean up temporary files)

    # Check if evaluation was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✅ Evaluation completed successfully!"

        # Parse and display results, also save to JSON file
        if [ -f "$LOG_FILE" ]; then
            echo "Parsing results..."
            python "$(dirname "$0")/fast_eval_metrics.py" "$LOG_FILE" --save-json "$RESULTS_FILE" --model-name "$model_name"

            if [ -f "$RESULTS_FILE" ]; then
                echo ""
                echo "✅ Results saved to: $RESULTS_FILE"
            fi
        else
            echo "Warning: Log file does not exist, cannot parse results"
        fi
        return 0
    else
        echo ""
        echo "❌ Evaluation failed, please check log: $LOG_FILE"
        return 1
    fi
}

# Main execution logic
if [ "$BATCH_MODE" = true ]; then
    # Batch mode: evaluate all predefined models
    echo ""
    echo "🚀 Starting batch evaluation of ${#SELECTED_MODELS[@]} models..."
    echo "Datasets: $DATASETS"
    echo ""

    TOTAL_MODELS=${#SELECTED_MODELS[@]}
    SUCCESS_COUNT=0
    SKIPPED_COUNT=0
    FAILED_MODELS=()

    for i in "${!SELECTED_MODELS[@]}"; do
        model="${SELECTED_MODELS[$i]}"
        backbone=$(detect_backbone "$model")

        echo "[$((i+1))/$TOTAL_MODELS] Evaluating model: $model"

        if evaluate_single_model "$model" "$backbone" "$DATASETS"; then
            # Check if all datasets were already evaluated (skipped)
            local model_name=$(basename "$model")
            if [[ "$model" == *"checkpoints"* ]]; then
                model_name=$(echo "$model" | sed 's/.*checkpoints\///' | sed 's/\/.*$//')
            fi
            local RESULTS_FILE="eval_results/${model_name}_results.json"

            # Check if all datasets have been evaluated
            local all_evaluated=true
            IFS=',' read -ra dataset_array <<< "$DATASETS"
            for dataset in "${dataset_array[@]}"; do
                if ! check_dataset_evaluated "$RESULTS_FILE" "$dataset"; then
                    all_evaluated=false
                    break
                fi
            done

            if [ "$all_evaluated" = true ]; then
                SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                echo "⏭️  Model $model already fully evaluated, skipping"
            else
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                echo "✅ Model $model evaluation successful"
            fi
        else
            FAILED_MODELS+=("$model")
            echo "❌ Model $model evaluation failed"
        fi

        # After each model evaluation, clean up its local cache to avoid disk space issues
        cleanup_predefined_model_cache "$model"

        echo ""
        echo "Progress: $((i+1))/$TOTAL_MODELS completed"
        echo "Success: $SUCCESS_COUNT, Skipped: $SKIPPED_COUNT, Failed: ${#FAILED_MODELS[@]}"
        echo "=================================="
    done

    echo ""
    echo "🎉 Batch evaluation completed!"
    echo "Total models: $TOTAL_MODELS"
    echo "Successful evaluations: $SUCCESS_COUNT"
    echo "Skipped evaluations: $SKIPPED_COUNT (all datasets already evaluated)"
    echo "Failed evaluations: ${#FAILED_MODELS[@]}"

    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "❌ Failed models:"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "  - $failed_model"
        done
    fi

    echo ""
    echo "📁 All results saved in: eval_results/"
    echo "You can view individual model JSON result files"

else
    # Single model mode: evaluate specified model
    if evaluate_single_model "$MODEL_PATH" "$BACKBONE" "$DATASETS"; then
        echo ""
        echo "=== Fast Test Completed ==="
        echo "Results saved to: eval_results/"
        echo ""
        echo "Metric explanation (verl metrics → pass@k style):"
        echo "- pass@1 ≈ mean@32: Average success rate of single attempt"
        echo "- maj@16 = maj@16/mean: Majority voting success rate of 16 attempts"
        echo "- pass@16 ≈ best@16/mean: Best result success rate of 16 attempts"
        echo "- best@32: Best success rate of 32 attempts (reference)"
        echo "======================"
    else
        exit 1
    fi
fi

