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
import numpy as np

def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        from recipe.r1.tasks import math

        return math.compute_score(solution_str, ground_truth)
    elif data_source == "Idavidrein/gpqa":
        from recipe.r1.tasks import gpqa

        return gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["livecodebench/code_generation_lite", "livecodebench/code_generation"]:
        from recipe.r1.tasks import livecodebench

        return livecodebench.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
def maj_reward_func(data_source, solution_str_list, ground_truth, extra_info=None):
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        from verl.utils.reward_score.ttrl.qwen.qwen_eval import majority_vote

        return majority_vote(solution_str_list, ground_truth, task="math")

def avg_reward_func(data_source, solution_str_list, ground_truth, extra_info=None):
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        from verl.utils.reward_score.ttrl.auto_verify import auto_verify
        n_prompts = len(solution_str_list) // len(ground_truth)
        ground_truth_list = [ground_truth] * n_prompts
        return  np.mean(auto_verify("math", 1, solution_str_list, ground_truth_list))