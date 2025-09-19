# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics, test_time_train_metrics)


class TTRLRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", compute_score=None, n_votes_per_prompt=1, n_samples_per_prompt=1, mode="eval", eval_n_samples=1) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        assert n_votes_per_prompt >= n_samples_per_prompt, f"For TTRL settings, n_votes_per_prompt {n_votes_per_prompt} should be greater than or equal to n_samples_per_prompt {n_samples_per_prompt}"

        print(f"TTRLRewardManager initialized with n_votes_per_prompt {n_votes_per_prompt}, n_samples_per_prompt {n_samples_per_prompt}, eval_n_samples {eval_n_samples}")


    def _data_source_to_task(self, data_source):
        # Standardize
        ds = str(data_source)
        if ds in ["MATH-TTT", "AIME-TTT", "AMC-TTT", "AIME25"]:
            return "math"
        if ds in ["GPQA-TTT"]:
            return "gpqa"
        if ds in ["BBEH", "bbeh", "BigBench-Extra-Hard"]:
            return "bbeh"

        dsl = ds.lower()
        # Keyword matching (more robust)
        if any(key in dsl for key in ["gpqa"]):
            return "gpqa"
        if any(key in dsl for key in ["aime", "math", "amc", "aime25"]):
            return "math"
        if "bbeh" in dsl or "bigbench" in dsl:
            return "bbeh"

        raise NotImplementedError(f"Data source {data_source} is not supported for TTRLRewardManager")

    def _compute_strategy_entropy(self, data_items):
        """
        Calculate strategy entropy Ĥ_ttrl(π_θ) = -1/N ∑(i=1 to N) (1/|y*|) ∑(t=1 to |y*|) log π_θ(y_t*|y_<t)

        Note: Returns token-normalized negative log-likelihood (similar to cross-entropy) for easy comparison across different sequence lengths

        Args:
            data_items: List of data items from DataProto

        Returns:
            float: Average normalized negative log-likelihood value (averaged by token)
        """
        try:
            if not data_items:
                return 0.0

            total_neg_log_likelihood = 0.0
            total_sequences = 0
            first_success = True

            for data_item in data_items:
                try:
                    # Check if log probability data exists
                    if hasattr(data_item, 'batch') and "old_log_probs" in data_item.batch:
                        # Get response length
                        prompt_length = data_item.batch["prompts"].shape[-1]
                        attention_mask = data_item.batch.get("attention_mask", None)

                        if attention_mask is not None and len(attention_mask) > prompt_length:
                            response_length = attention_mask[prompt_length:].sum().item()

                            if response_length > 0:
                                old_log_probs = data_item.batch["old_log_probs"]

                                # Fix slicing logic: old_log_probs only contains response part
                                if isinstance(old_log_probs, torch.Tensor) and old_log_probs.numel() > 0:
                                    log_probs_length = old_log_probs.shape[-1]

                                    # Choose slicing strategy based on actual length of old_log_probs
                                    if log_probs_length == response_length:
                                        # Case 1: old_log_probs only contains response part, use directly
                                        response_log_probs = old_log_probs
                                    elif log_probs_length == prompt_length + response_length:
                                        # Case 2: old_log_probs contains prompt+response, need to slice
                                        response_log_probs = old_log_probs[prompt_length:prompt_length+response_length]
                                    elif log_probs_length > response_length:
                                        # Case 3: Take last response_length tokens (fallback strategy)
                                        response_log_probs = old_log_probs[-response_length:]
                                    else:
                                        # Case 4: Insufficient length, skip
                                        continue

                                    # Calculate total log probability of sequence (sum over all tokens)
                                    if response_log_probs.numel() > 0:
                                        sequence_log_prob = torch.sum(response_log_probs).item()
                                        # Calculate normalized negative log-likelihood (averaged by token)
                                        normalized_neg_log_likelihood = -sequence_log_prob / response_length
                                        total_neg_log_likelihood += normalized_neg_log_likelihood
                                        total_sequences += 1

                                        # Print debug info on first success
                                        if first_success:
                                            print(f"    Strategy entropy calculation enabled: old_log_probs shape={old_log_probs.shape}, response_length={response_length}")
                                            first_success = False
                except Exception as e:
                    # Single data item processing failed, skip
                    continue

            # Calculate average negative log-likelihood
            if total_sequences > 0:
                return total_neg_log_likelihood / total_sequences
            else:
                return 0.0

        except Exception as e:
            # Overall calculation failed, return 0
            return 0.0

    def compute_post_ttrl_metrics(self, data: DataProto):
        """
        Compute post TTRL metrics for the given data.
        """
        assert len(data) % self.n_samples_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_samples_per_prompt}"
        prompt_num = len(data) // self.n_samples_per_prompt

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
                group_vote_rewards = []
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []
                task = None

                for i in range(self.n_samples_per_prompt):
                    data_item = data[prompt_i * self.n_samples_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]
                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    vote_reward = data_item.batch["acc"]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_vote_rewards.append(vote_reward)
                    group_extra_info.append(extra_info)
                
                post_ttrl_metrics = post_test_time_train_metrics(group_pred_outputs, group_labels, group_vote_rewards, task=task, extra_info=group_extra_info)
                for k, v in post_ttrl_metrics.items():
                    post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _compute_ttrl_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            assert len(data) % self.n_votes_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_votes_per_prompt}"
            
            prompt_num = len(data) // self.n_votes_per_prompt

            reward_tensor = torch.zeros_like(data.batch["responses"][:prompt_num*self.n_samples_per_prompt], dtype=torch.float32)

            already_print_data_sources = {}

            all_ttrl_metrics = defaultdict(list)

            scores = [0.0 for _ in range(len(data))]
            
            for prompt_i in range(prompt_num):
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []

                task = None

                for i in range(self.n_votes_per_prompt):
                    data_item = data[prompt_i * self.n_votes_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]

                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_extra_info.append(extra_info)
                rewards, ttrl_metrics = test_time_train_metrics(group_pred_outputs, group_labels, task=task, extra_info=group_extra_info)

                # === Calculate strategy entropy ===
                current_group_data = data[prompt_i * self.n_votes_per_prompt:(prompt_i + 1) * self.n_votes_per_prompt]
                strategy_entropy = self._compute_strategy_entropy(current_group_data)
                ttrl_metrics["neg_log_likelihood"] = strategy_entropy
                if strategy_entropy > 0:
                    print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")

                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)

                for i in range(self.n_votes_per_prompt):
                    if i < self.n_samples_per_prompt:
                        reward_tensor[prompt_i * self.n_samples_per_prompt + i, valid_response_length - 1] = rewards[i]
                    scores[prompt_i * self.n_votes_per_prompt + i] = rewards[i]

                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                        print("[score]", rewards[i])

            data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v


            return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print_data_sources = {}

            # Group by task to avoid inconsistency errors from mixed tasks
            task_groups = {}
            # Record valid response length for each sample to facilitate reward backfill
            sample_valid_resp_len = {}

            for i in range(len(data)):
                data_item = data[i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]
                sample_valid_resp_len[i] = int(valid_response_length)

                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]

                # Print a few samples
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)

                task_key = self._data_source_to_task(data_source)
                if task_key not in task_groups:
                    task_groups[task_key] = {"indices": [], "outputs": [], "labels": [], "extra": []}
                task_groups[task_key]["indices"].append(i)
                task_groups[task_key]["outputs"].append(response_str)
                task_groups[task_key]["labels"].append(ground_truth)
                task_groups[task_key]["extra"].append(extra_info)

            # Call verification function separately by task and backfill results to corresponding sample positions
            for task_key, group in task_groups.items():
                rewards, verify_extra_info = auto_verify(task_key, group["outputs"], group["labels"], extra_info=group["extra"])
                # Aggregate extra information
                for k, v in verify_extra_info.items():
                    if isinstance(v, list):
                        reward_extra_info[k] += v
                # Backfill reward to corresponding sample's last token position
                for idx_in_group, sample_idx in enumerate(group["indices"]):
                    valid_len = sample_valid_resp_len[sample_idx]
                    reward_tensor[sample_idx, valid_len - 1] = rewards[idx_in_group]

            # Compute TTRL metrics
            all_ttrl_metrics = defaultdict(list)
            prompt_num = len(data) // self.eval_n_samples
            for prompt_i in range(prompt_num):
                group_pred_outputs_ttrl = []
                group_labels_ttrl = []
                group_extra_info_ttrl = []

                task = None

                for i in range(self.eval_n_samples):
                    data_item = data[prompt_i * self.eval_n_samples + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]



                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels_ttrl.append(ground_truth)
                    group_pred_outputs_ttrl.append(response_str)
                    group_extra_info_ttrl.append(extra_info)

                _, ttrl_metrics = test_time_train_metrics(group_pred_outputs_ttrl, group_labels_ttrl, task=task, extra_info=group_extra_info_ttrl)
                
                # === Calculate strategy entropy ===
                current_group_data = data[prompt_i * self.eval_n_samples:(prompt_i + 1) * self.eval_n_samples]
                strategy_entropy = self._compute_strategy_entropy(current_group_data)
                ttrl_metrics["neg_log_likelihood"] = strategy_entropy
                if strategy_entropy > 0:
                    print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")
                
                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v


            
            return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict=False):

        if self.mode == "train":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for TTRLRewardManager")

        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "ttrl_info": ttrl_info,
                }
        else:
            return reward_tensor