"""
Лабораторная работа № 9. Вариант 17.
Тип графа: дерево (неориентированное).
n = 1000, 5000, 10000.
Задача: центр дерева.
Доп. И3: визуализация малого графа и дерева обхода (Mermaid).
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT = Path("/home/workdir/artifacts/lr9")
CODE = OUT / "code"
DIAG = OUT / "diagrams"
CODE.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)


class MatrixGraph:
    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1
        self.matrix[v][u] = 1

    def has_edge(self, u, v):
        return self.matrix[u][v] == 1

    def neighbors(self, v):
        row = self.matrix[v]
        return [u for u in range(self.n) if row[u]]


class ListGraph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def has_edge(self, u, v):
        return v in self.adj[u]

    def neighbors(self, v):
        return self.adj[v]


def bfs(graph, start):
    n = graph.n
    dist = [-1] * n
    parent = [-1] * n
    order = []
    dist[start] = 0
    q = deque([start])
    while q:
        v = q.popleft()
        order.append(v)
        for u in graph.neighbors(v):
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                parent[u] = v
                q.append(u)
    return order, dist, parent


def dfs(graph, start):
    visited = [False] * graph.n
    order, stack = [], [start]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for u in reversed(list(graph.neighbors(v))):
            if not visited[u]:
                stack.append(u)
    return order


def generate_tree(cls, n, seed=42):
    """Случайное дерево: Prüfer-подобное / последовательное присоединение."""
    rng = random.Random(seed + n)
    g = cls(n)
    # строим дерево: каждая новая вершина соединяется со случайной уже существующей
    for v in range(1, n):
        u = rng.randrange(v)
        g.add_edge(u, v)
    return g


def tree_center(graph):
    """
    Центр дерева: 1 или 2 вершины с минимальной эксцентриситетом.
    Алгоритм: два BFS — концы диаметра, затем середина диаметра.
    """
    # 1) BFS из 0 → самая дальняя a
    _, dist0, _ = bfs(graph, 0)
    a = max(range(graph.n), key=lambda v: dist0[v])
    # 2) BFS из a → самая дальняя b, parent для пути
    _, dist_a, parent = bfs(graph, a)
    b = max(range(graph.n), key=lambda v: dist_a[v])
    diameter = dist_a[b]
    # 3) путь a → b
    path = []
    v = b
    while v != -1:
        path.append(v)
        v = parent[v]
    path.reverse()
    # центр — середина пути
    mid = diameter // 2
    if diameter % 2 == 0:
        centers = [path[mid]]
    else:
        centers = [path[mid], path[mid + 1]]
    return sorted(centers), diameter, path


def deep_sizeof(obj, seen=None):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, list):
        size += sum(deep_sizeof(x, seen) for x in obj)
    return size


def main():
    # --- Малый граф для проверки и И3 ---
    print("=== Малый граф (дерево, 10 вершин) ===")
    small = ListGraph(10)
    edges_small = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7), (3, 8), (8, 9)]
    for u, v in edges_small:
        small.add_edge(u, v)
    order_bfs, dist, parent = bfs(small, 0)
    order_dfs = dfs(small, 0)
    centers, diam, path = tree_center(small)
    print(f"  BFS order: {order_bfs}")
    print(f"  DFS order: {order_dfs}")
    print(f"  diameter={diam}, path={path}, center={centers}")

    # Mermaid for И3
    mermaid_lines = ["graph TD"]
    for u, v in edges_small:
        mermaid_lines.append(f"    {u}(({u})) --- {v}(({v}))")
    # highlight center
    for c in centers:
        mermaid_lines.append(f"    style {c} fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px")
    # BFS tree edges (parent)
    mermaid_lines.append("")
    mermaid_lines.append("%% BFS tree from 0 (parent pointers):")
    for v, p in enumerate(parent):
        if p != -1:
            mermaid_lines.append(f"%% parent[{v}] = {p}")

    mermaid_path = DIAG / "small_tree.mmd"
    mermaid_path.write_text("\n".join(mermaid_lines), encoding="utf-8")

    # --- Бенчмарк на деревьях ---
    SIZES = [1000, 5000, 10000]
    results = []
    print(f"\n{'n':>6} {'матрица,с':>12} {'списки,с':>12} {'мат,МБ':>10} {'сп,МБ':>10} "
          f"{'центр':>12} {'диам':>6}")
    for n in SIZES:
        gm = generate_tree(MatrixGraph, n)
        gl = generate_tree(ListGraph, n)

        t0 = time.perf_counter()
        bfs(gm, 0)
        t_m = time.perf_counter() - t0

        t0 = time.perf_counter()
        bfs(gl, 0)
        t_l = time.perf_counter() - t0

        mem_m = deep_sizeof(gm.matrix) / (1024 * 1024)
        mem_l = deep_sizeof(gl.adj) / (1024 * 1024)

        t0 = time.perf_counter()
        centers, diam, _ = tree_center(gl)
        t_c = time.perf_counter() - t0

        row = {
            "n": n, "t_matrix": t_m, "t_list": t_l,
            "mem_matrix_mb": mem_m, "mem_list_mb": mem_l,
            "centers": centers, "diameter": diam, "t_center": t_c,
        }
        results.append(row)
        print(f"{n:>6} {t_m:>12.4f} {t_l:>12.4f} {mem_m:>10.2f} {mem_l:>10.2f} "
              f"{str(centers):>12} {diam:>6}")

    # Also verify center on small
    small_info = {
        "edges": edges_small,
        "bfs_order": order_bfs,
        "dfs_order": order_dfs,
        "centers": centers if n == 10 else tree_center(small)[0],
        "diameter": tree_center(small)[1],
        "path": tree_center(small)[2],
        "dist_from_0": dist,
        "parent": parent,
    }
    # fix small_info centers
    c0, d0, p0 = tree_center(small)
    small_info["centers"] = c0
    small_info["diameter"] = d0
    small_info["path"] = p0

    out = {"benchmark": results, "small": small_info}
    with open(CODE / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("results.json saved")

    # Graphs
    font_path = "/usr/share/fonts/SlidesCarnival/google/Tinos/Tinos-Regular.ttf"
    font_bold = "/usr/share/fonts/SlidesCarnival/google/Tinos/Tinos-Bold.ttf"
    prop = fm.FontProperties(fname=font_path) if Path(font_path).exists() else None
    prop_b = fm.FontProperties(fname=font_bold) if Path(font_bold).exists() else None

    ns = [r["n"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ns, [r["t_matrix"] for r in results], "o-", label="Матрица", color="#c62828")
    axes[0].plot(ns, [r["t_list"] for r in results], "s-", label="Списки", color="#1565c0")
    axes[0].set_xlabel("n", fontproperties=prop)
    axes[0].set_ylabel("Время BFS, с", fontproperties=prop)
    axes[0].set_title("Время обхода дерева", fontproperties=prop_b)
    axes[0].legend(prop=prop, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_yscale("log")

    axes[1].plot(ns, [r["mem_matrix_mb"] for r in results], "o-", label="Матрица", color="#c62828")
    axes[1].plot(ns, [r["mem_list_mb"] for r in results], "s-", label="Списки", color="#1565c0")
    axes[1].set_xlabel("n", fontproperties=prop)
    axes[1].set_ylabel("Память, МБ", fontproperties=prop)
    axes[1].set_title("Расход памяти", fontproperties=prop_b)
    axes[1].legend(prop=prop, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_yscale("log")

    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            if prop:
                label.set_fontproperties(prop)
            label.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(DIAG / "tree_bench.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("tree_bench.png saved")


if __name__ == "__main__":
    main()
