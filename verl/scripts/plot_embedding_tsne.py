#!/usr/bin/env python3
"""
载入两个模型的 embeddings .npz，使用 t-SNE 降维并绘制分布对比图。

输入：两个 .npz 文件路径（由 generate_rollouts_and_embed.py 生成）。
输出：保存 PNG（默认与输入同目录），并可选显示。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def load_npz(path: str) -> Tuple[np.ndarray, dict, list]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(f"文件缺少 'embeddings': {path}")
    embs = data["embeddings"]  # 形状 [Q, S, D]
    meta_raw = data.get("meta", None)
    meta = json.loads(str(meta_raw)) if meta_raw is not None else {}
    prompts = data.get("prompts", None)
    prompts_list = prompts.tolist() if prompts is not None else []
    return embs, meta, prompts_list


def prepare_points(embs: np.ndarray) -> np.ndarray:
    if embs.ndim == 3:
        Q, S, D = embs.shape
        return embs.reshape(Q * S, D)
    elif embs.ndim == 2:
        return embs
    else:
        raise ValueError(f"不支持的嵌入形状: {embs.shape}")


def run_tsne(x: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=seed)
    y = tsne.fit_transform(x)
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="比较两个模型 embeddings 的 t-SNE 分布")
    parser.add_argument("--npz-a", type=str, required=True, help="模型 A 的 npz 路径")
    parser.add_argument("--npz-b", type=str, required=True, help="模型 B 的 npz 路径")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--per-question-2x2", action="store_true", help="按题目分别作图，2x2 子图（前4题）")

    args = parser.parse_args()

    embs_a, meta_a, prompts_a = load_npz(args.npz_a)
    embs_b, meta_b, prompts_b = load_npz(args.npz_b)

    # 分题目 2x2 子图模式
    if args.per_question_2x2:
        if embs_a.ndim != 3 or embs_b.ndim != 3:
            raise ValueError("per-question-2x2 需要 [Q,S,D] 形状的 embeddings")
        Q = embs_a.shape[0]
        if Q < 4:
            raise ValueError(f"需要至少 4 道题，当前 Q={Q}")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
        axes = axes.flatten()
        colors = np.array([[0.2, 0.5, 0.9, 0.7], [0.9, 0.3, 0.2, 0.7]])
        for i in range(4):
            Ea = embs_a[i]
            Eb = embs_b[i]
            if Ea.shape[1] != Eb.shape[1]:
                d = min(Ea.shape[1], Eb.shape[1])
                Ea = Ea[:, :d]
                Eb = Eb[:, :d]
            X = np.concatenate([Ea, Eb], axis=0)
            labels = np.array([0] * len(Ea) + [1] * len(Eb))
            Y = run_tsne(X, perplexity=args.perplexity, seed=args.seed)
            ax = axes[i]
            ax.scatter(Y[labels == 0, 0], Y[labels == 0, 1], s=8, c=[colors[0]], label=meta_a.get("model", "A"), edgecolors='none')
            ax.scatter(Y[labels == 1, 0], Y[labels == 1, 1], s=8, c=[colors[1]], label=meta_b.get("model", "B"), edgecolors='none')
            title = f"Q{i}"
            try:
                if prompts_a and i < len(prompts_a):
                    title = f"Q{i}: {str(prompts_a[i])[:48]}"
            except Exception:
                pass
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        # 统一图例
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper right')
        fig.suptitle("t-SNE per question (A vs B)")
        plt.tight_layout(rect=[0, 0, 0.94, 0.96])

        if args.out:
            out_path = args.out
        else:
            base_dir = os.path.dirname(os.path.commonpath([os.path.abspath(args.npz_a), os.path.abspath(args.npz_b)]))
            out_path = os.path.join(base_dir, "tsne_compare_per_q_2x2.png")
        plt.savefig(out_path)
        print(f"已保存图像: {out_path}")
        if args.show:
            plt.show()
        return

    # 原单图模式
    X_a = prepare_points(embs_a)
    X_b = prepare_points(embs_b)

    # 对齐维度（通常相同）
    if X_a.shape[1] != X_b.shape[1]:
        d = min(X_a.shape[1], X_b.shape[1])
        X_a = X_a[:, :d]
        X_b = X_b[:, :d]

    X = np.concatenate([X_a, X_b], axis=0)
    labels = np.array([0] * len(X_a) + [1] * len(X_b))

    print(f"t-SNE 输入: {X.shape}, A={len(X_a)} B={len(X_b)}")
    Y = run_tsne(X, perplexity=args.perplexity, seed=args.seed)

    # 绘图
    plt.figure(figsize=(8, 6), dpi=150)
    colors = np.array([[0.2, 0.5, 0.9, 0.7], [0.9, 0.3, 0.2, 0.7]])
    ms = 8
    plt.scatter(Y[labels == 0, 0], Y[labels == 0, 1], s=ms, c=[colors[0]], label=meta_a.get("model", "A"), edgecolors='none')
    plt.scatter(Y[labels == 1, 0], Y[labels == 1, 1], s=ms, c=[colors[1]], label=meta_b.get("model", "B"), edgecolors='none')
    plt.legend()
    plt.title("t-SNE of Embeddings (A vs B)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()

    # 输出路径
    if args.out:
        out_path = args.out
    else:
        base_dir = os.path.dirname(os.path.commonpath([os.path.abspath(args.npz_a), os.path.abspath(args.npz_b)]))
        out_path = os.path.join(base_dir, "tsne_compare.png")
    plt.savefig(out_path)
    print(f"已保存图像: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


