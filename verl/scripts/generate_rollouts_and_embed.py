#!/usr/bin/env python3
"""
生成 AIME25 样本题目的多样化 rollouts，并调用 vLLM 嵌入服务保存嵌入结果。

功能概述：
- 从 AIME25/test-simplerl.parquet 采样 4 道题（可配置）。
- 对每道题用 vLLM 以 8 卡张量并行生成 16 个 rollouts。
- 调用 http://localhost:2341/embed（与 semantic_novelty 保持一致）获取嵌入。
- 将 rollouts 与 embeddings 一起保存为 .npz，便于后续可视化比较。

使用示例：
python scripts/generate_rollouts_and_embed.py \
  --models default \
  --dataset /apdcephfs_us/share_300814644/user/yujun/ttrl/TTRL/verl/data/AIME25/test-simplerl.parquet \
  --num-questions 4 \
  --samples-per-question 16 \
  --tensor-parallel-size 8 \
  --embed-url http://localhost:2341 \
  --outdir /apdcephfs_us/share_300814644/user/yujun/ttrl/TTRL/verl/eval_results/embeddings

依赖：vllm, pandas, pyarrow(or fastparquet), numpy, requests
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


def _sanitize_model_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name.strip())
    return s.strip("._-") or "model"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_questions(dataset_path: str, num_questions: int, seed: int) -> Tuple[List[str], List[str]]:
    """从 parquet/json 读取题目；优先 parquet。返回 (ids, prompts)。"""
    rnd = random.Random(seed)

    def _normalize(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cols = set(c.lower() for c in df.columns)
        # 可能的列名候选
        prompt_col = None
        for cand in ["prompt", "question", "input", "instruction", "text"]:
            if cand in cols:
                # 找原列名
                for c in df.columns:
                    if c.lower() == cand:
                        prompt_col = c
                        break
                if prompt_col:
                    break
        if prompt_col is None:
            raise ValueError(f"未找到题目字段，可用列: {list(df.columns)}")

        id_col = None
        for cand in ["id", "idx", "index"]:
            if cand in cols:
                for c in df.columns:
                    if c.lower() == cand:
                        id_col = c
                        break
                if id_col:
                    break

        ids = [str(x) if id_col is not None else str(i) for i, x in enumerate(df[id_col] if id_col else range(len(df)))]
        prompts = [str(x) for x in df[prompt_col].tolist()]
        return ids, prompts

    ids: List[str]
    prompts: List[str]

    if dataset_path.lower().endswith(".parquet") and os.path.exists(dataset_path):
        df = pd.read_parquet(dataset_path)
        ids, prompts = _normalize(df)
    else:
        # 尝试旁路 JSON（例如 data/AIME25/test.json）
        json_path = os.path.splitext(dataset_path)[0] + ".json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"未找到数据文件: {dataset_path} 或 {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        ids, prompts = _normalize(df)

    if len(prompts) < num_questions:
        raise ValueError(f"数据量不足：需要 {num_questions} 道题，实际 {len(prompts)}")

    idxs = list(range(len(prompts)))
    rnd.shuffle(idxs)
    idxs = idxs[:num_questions]

    return [ids[i] for i in idxs], [prompts[i] for i in idxs]


def generate_rollouts_vllm(model_name: str, prompts: List[str], samples_per_question: int, tp_size: int,
                           max_new_tokens: int = 512, temperature: float = 0.8, top_p: float = 0.95,
                           repetition_penalty: float = 1.0) -> List[List[str]]:
    """使用 vLLM 以张量并行生成多样化 rollouts。"""
    from vllm import LLM, SamplingParams

    print(f"[vLLM] 准备加载模型: {model_name} tp={tp_size}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=False,
        max_model_len=8192,
        swap_space=2,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp",
    )
    sampling = SamplingParams(
        n=samples_per_question,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
    )

    rollouts: List[List[str]] = []
    for pi, prompt in enumerate(prompts):
        print(f"[vLLM] 生成第 {pi+1}/{len(prompts)} 题的 {samples_per_question} 个 rollouts ...")
        outputs = llm.generate([prompt], sampling)
        # vLLM 对单个输入返回一个 RequestOutput，其中 outputs 为 n 个候选
        if not outputs:
            rollouts.append([""] * samples_per_question)
            continue
        texts = [o.text for o in outputs[0].outputs]
        # 若少于 n，补空串占位
        while len(texts) < samples_per_question:
            texts.append("")
        rollouts.append(texts[:samples_per_question])
    return rollouts


def embed_texts_embed_api(texts: List[str], embed_url: str, batch_size: int = 128, timeout: int = 600) -> np.ndarray:
    """调用 /embed API 分批获取文本嵌入，返回 (N, D) 数组。"""
    url = embed_url.rstrip("/") + "/embed"
    headers = {"Content-Type": "application/json"}
    all_embs: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        print(f"[embed] {start}..{start+len(chunk)-1} / {len(texts)-1}")
        payload = {"texts": chunk}
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embs = data.get("embeddings", [])
        if not embs:
            raise RuntimeError("嵌入服务返回空列表")
        all_embs.append(np.asarray(embs, dtype=np.float32))

    out = np.concatenate(all_embs, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="为 AIME25 题目生成 rollouts 并保存嵌入")
    parser.add_argument("--models", type=str, default="default",
                        help="模型名（逗号分隔）或 'default' 使用内置列表")
    parser.add_argument("--dataset", type=str,
                        default="/apdcephfs_us/share_300814644/user/yujun/ttrl/TTRL/verl/data/AIME25/test-simplerl.parquet",
                        help="数据集路径（parquet 优先，自动回退到同名 .json）")
    parser.add_argument("--num-questions", type=int, default=4)
    parser.add_argument("--samples-per-question", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--embed-url", type=str, default="http://30.159.163.76:2341",
                        help="嵌入服务地址，默认固定为 http://30.159.163.76:2341")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str,
                        default="/apdcephfs_us/share_300814644/user/yujun/ttrl/TTRL/verl/eval_results/embeddings")

    args = parser.parse_args()

    if args.models.strip().lower() == "default":
        model_list = [
            "yujunzhou/MATH-TTT-Qwen3-4B-Base-TTRL",
            "yujunzhou/AIME-TTT-Qwen3-4B-Base-TTRL",
            "yujunzhou/AIME-TTT-Qwen3-4B-Base-Semantic-ClipHigh-Ent",
            "yujunzhou/MATH-TTT-Qwen3-4B-Base-Semantic-ClipHigh-Ent-Simple-v2"
        ]
    else:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_list:
            raise ValueError("--models 不能为空")

    print(f"将使用模型：{model_list}")
    _ensure_dir(args.outdir)

    # 采样题目
    q_ids, q_prompts = load_questions(args.dataset, args.num_questions, args.seed)
    print(f"采样到 {len(q_prompts)} 道题: {q_ids}")

    for mi, model_name in enumerate(model_list):
        print("=" * 80)
        print(f"[{mi+1}/{len(model_list)}] 模型 {model_name} 生成 rollouts")
        rollouts = generate_rollouts_vllm(
            model_name=model_name,
            prompts=q_prompts,
            samples_per_question=args.samples_per_question,
            tp_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        # 展平后送入嵌入接口
        flat_texts: List[str] = []
        index_map: List[Tuple[int, int]] = []  # (q_idx, s_idx)
        for qi, outs in enumerate(rollouts):
            for si, t in enumerate(outs):
                flat_texts.append(t)
                index_map.append((qi, si))

        print(f"调用嵌入服务: {args.embed_url}，共 {len(flat_texts)} 条")
        embs = embed_texts_embed_api(flat_texts, args.embed_url, batch_size=128)
        if embs.ndim != 2:
            raise RuntimeError(f"嵌入形状异常: {embs.shape}")
        dim = embs.shape[1]
        # 还原成 [Q, S, D]
        Q = len(q_prompts)
        S = args.samples_per_question
        emb_3d = np.zeros((Q, S, dim), dtype=np.float32)
        for (qi, si), vec in zip(index_map, embs):
            emb_3d[qi, si] = vec

        # 保存（文件名仅到模型名为止）
        fname = f"{_sanitize_model_name(model_name)}.npz"
        out_path = os.path.join(args.outdir, fname)
        meta = {
            "model": model_name,
            "embed_url": args.embed_url,
            "embed_model_name": "Qwen/Qwen3-Embedding-4B",
            "dataset": os.path.abspath(args.dataset),
            "question_ids": q_ids,
            "num_questions": Q,
            "samples_per_question": S,
            "embedding_dim": int(dim),
            "tp_size": int(args.tensor_parallel_size),
            "gen_params": {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
            },
        }
        np.savez_compressed(
            out_path,
            model=np.array(meta["model"]).astype(object),
            embed_url=np.array(meta["embed_url"]).astype(object),
            dataset=np.array(meta["dataset"]).astype(object),
            question_ids=np.array(meta["question_ids"]).astype(object),
            prompts=np.array(q_prompts).astype(object),
            rollouts=np.array(rollouts).astype(object),
            embeddings=emb_3d,
            meta=json.dumps(meta, ensure_ascii=False),
        )
        print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()


