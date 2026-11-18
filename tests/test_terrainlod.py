# -*- coding: utf-8 -*-
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import terrainlod as mod  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")


def _load():
    with open(os.path.join(DATA, "heightmap.json"), encoding="utf-8") as f:
        d = json.load(f)
    return d["heights"]


def _all_nodes(root):
    stack = [root]
    out = []
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def _walk(n):
    yield n
    for c in (n.children or []):
        yield from _walk(c)


def _cells(leaves):
    """叶子覆盖的格子 -> 叶子 映射；重复覆盖即抛错。"""
    owner = {}
    for n in leaves:
        for j in range(n.y, n.y + n.ext_h):
            for i in range(n.x, n.x + n.ext_w):
                key = (i, j)
                assert key not in owner, "叶子之间存在重叠"
                owner[key] = n
    return owner


def _boundary_keys_equal(leaves):
    """缝合后，每条层级差边界上各方顶点键必须完全一致（无 T 型接点）。"""
    for nodes, vertical, boundary, keys in mod.find_seams(leaves):
        target = set(keys)
        for n in nodes:
            got = set()
            for i, j in n.mesh_vertex_keys():
                if (vertical and i == boundary) or (
                        not vertical and j == boundary):
                    got.add((i, j))
            if got != target:
                return False
    return True


# ---------- 骨架原有测试 ----------

def test_root_covers_all():
    h = _load()
    root = mod.build_quadtree(h, 3)
    leaves = mod.select_lod(root, 10 ** 9)
    assert leaves and all(hasattr(n, "x") and hasattr(n, "size") for n in leaves)


def test_error_monotone():
    h = _load()
    root = mod.build_quadtree(h, 3)
    for n in _walk(root):
        for c in (n.children or []):
            assert n.error >= c.error - 1e-12


def test_budget_respected():
    h = _load()
    root = mod.build_quadtree(h, 3)
    tri = 20
    leaves = mod.select_lod(root, tri)
    total = sum(2 * (n.size * n.size) for n in leaves)
    assert total <= tri


def test_deterministic():
    h = _load()
    a = mod.select_lod(mod.build_quadtree(h, 3), 100)
    b = mod.select_lod(mod.build_quadtree(h, 3), 100)
    assert [(n.x, n.y, n.size) for n in a] == \
           [(n.x, n.y, n.size) for n in b]


def test_stitch_returns_verts():
    h = _load()
    root = mod.build_quadtree(h, 3)
    leaves = mod.select_lod(root, 100)
    verts, extra = mod.stitch_seams(leaves, h)
    assert verts and isinstance(extra, int) and extra >= 0


# ---------- 覆盖性与无重叠 ----------

def test_full_coverage_no_overlap_square():
    h = _load()
    rows, cols = len(h), len(h[0])
    root = mod.build_quadtree(h, 3)
    for budget in (8, 20, 100, 1000, 10 ** 9):
        leaves = mod.select_lod(root, budget)
        owner = _cells(leaves)
        assert len(owner) == rows * cols  # 无空洞
        # 无重叠已在 _cells 中断言


def test_full_refinement_reaches_cells():
    h = _load()
    rows, cols = len(h), len(h[0])
    root = mod.build_quadtree(h, 6)  # 足够深，全部切成单格
    leaves = mod.select_lod(root, 10 ** 9)
    owner = _cells(leaves)
    assert len(owner) == rows * cols
    assert all(n.ext_w == 1 and n.ext_h == 1 for n in leaves)


def test_rectangular_partition_consistency():
    # 任意预算下，叶子宽高之和都等于地形尺寸（划分恒等式）
    h = [[(i * 3 + j * 5) % 7 for i in range(10)] for j in range(7)]
    root = mod.build_quadtree(h, 6)
    for budget in (8, 40, 200, 5000):
        leaves = mod.select_lod(root, budget)
        _cells(leaves)  # 重叠检查
        # 每一行被叶子宽度完整铺满
        for row in range(7):
            row_leaves = [n for n in leaves
                          if n.y <= row < n.y + n.ext_h]
            assert sum(n.ext_w for n in row_leaves) == 10


# ---------- 误差严格单调 ----------

def test_error_strictly_monotone():
    h = _load()
    root = mod.build_quadtree(h, 4)
    for n in _all_nodes(root):
        for c in n.children:
            assert n.error > c.error


def test_error_non_negative_and_finite():
    h = _load()
    root = mod.build_quadtree(h, 4)
    for n in _all_nodes(root):
        assert math.isfinite(n.error) and n.error >= 0.0


# ---------- 预算 ----------

def test_budget_never_exceeded_many_budgets():
    h = _load()
    root = mod.build_quadtree(h, 4)
    floor = root.tri_cost()
    # 低于根节点成本的预算由 test_tiny_budget_degrades_to_root 专门覆盖
    for budget in range(floor, 400, 7):
        leaves = mod.select_lod(root, budget)
        total = sum(n.tri_cost() for n in leaves)
        assert total <= budget


def test_tiny_budget_degrades_to_root():
    h = _load()
    root = mod.build_quadtree(h, 4)
    leaves = mod.select_lod(root, 1)  # 连根的 8 三角形都买不起
    assert leaves == [root]
    assert sum(n.tri_cost() for n in leaves) == root.tri_cost()


def test_budget_monotone_refinement():
    # 预算增大，叶子集合只能更细：每个新叶子必须落在旧叶子内部
    h = _load()
    root = mod.build_quadtree(h, 5)
    prev = {id(root): root}
    for budget in (8, 24, 64, 200, 800, 4000, 10 ** 9):
        leaves = mod.select_lod(root, budget)
        cur_owner = _cells(leaves)
        for cell, node in cur_owner.items():
            old = prev.get(cell)
            if old is not None:
                # 同一格子的新叶子深度不能变浅
                assert node.depth >= old.depth
        prev = cur_owner


def test_budget_used_fully_when_refinement_available():
    # 存在可负担的细分时，不允许预算没用完就停止（无退化）
    h = _load()
    root = mod.build_quadtree(h, 5)
    for budget in (64, 200, 800):
        leaves = mod.select_lod(root, budget)
        used = sum(n.tri_cost() for n in leaves)
        frontier = set(id(n) for n in leaves)
        cheapest = min(
            (sum(c.tri_cost() for c in n.children) - n.tri_cost()
             for n in leaves if n.children),
            default=None)
        if cheapest is not None:
            assert used + cheapest > budget or used == budget


def test_deterministic_across_budgets_and_calls():
    h = _load()
    for budget in (20, 100, 999):
        a = mod.select_lod(mod.build_quadtree(h, 4), budget)
        b = mod.select_lod(mod.build_quadtree(h, 4), budget)
        assert [(n.x, n.y, n.size, n.depth) for n in a] == \
               [(n.x, n.y, n.size, n.depth) for n in b]
        # 缝合结果同样确定
        va, ea = mod.stitch_seams(a, h)
        vb, eb = mod.stitch_seams(b, h)
        assert va == vb and ea == eb


# ---------- 接缝 / T 型接点 ----------

def test_seams_welded_no_t_junction():
    h = _load()
    root = mod.build_quadtree(h, 4)
    for budget in (40, 100, 300, 1000):
        leaves = mod.select_lod(root, budget)
        mod.stitch_seams(leaves, h)
        assert _boundary_keys_equal(leaves)


def test_stitch_extra_count_is_insertion_count():
    h = _load()
    root = mod.build_quadtree(h, 4)
    leaves = mod.select_lod(root, 100)
    _, extra = mod.stitch_seams(leaves, h)
    counted = sum(len(n.stitched_keys) for n in leaves)
    assert extra == counted
    # 全细化（无层级差边界）时不应有任何补点
    deep_leaves = mod.select_lod(mod.build_quadtree(h, 6), 10 ** 9)
    _, extra_deep = mod.stitch_seams(deep_leaves, h)
    assert extra_deep == 0
    assert mod.find_seams(deep_leaves) == []


def test_stitch_vertices_are_grounded_on_heightmap():
    h = _load()
    root = mod.build_quadtree(h, 4)
    leaves = mod.select_lod(root, 100)
    verts, _ = mod.stitch_seams(leaves, h)
    rows, cols = len(h), len(h[0])
    seen = set()
    for i, j, z in verts:
        assert 0 <= i <= cols and 0 <= j <= rows
        assert (i, j) not in seen  # 全局去重
        seen.add((i, j))
        ci, cj = min(i, cols - 1), min(j, rows - 1)
        assert z == h[cj][ci]


def test_stitch_repeatable_resets_state():
    h = _load()
    root = mod.build_quadtree(h, 4)
    leaves = mod.select_lod(root, 100)
    mod.stitch_seams(leaves, h)
    v1, e1 = mod.stitch_seams(leaves, h)
    v2, e2 = mod.stitch_seams(leaves, h)
    assert v1 == v2 and e1 == e2


# ---------- 边界情况 ----------

def test_non_power_of_two_sizes():
    for cols, rows in ((5, 5), (10, 7), (7, 10), (3, 1), (6, 9)):
        h = [[(i * i + 3 * j) % 11 for i in range(cols)]
             for j in range(rows)]
        root = mod.build_quadtree(h, 8)
        for budget in (8, 100, 10 ** 9):
            leaves = mod.select_lod(root, budget)
            owner = _cells(leaves)
            assert len(owner) == rows * cols
            verts, extra = mod.stitch_seams(leaves, h)
            assert verts and extra >= 0
            assert _boundary_keys_equal(leaves)


def test_single_cell_map():
    h = [[42]]
    root = mod.build_quadtree(h, 6)
    leaves = mod.select_lod(root, 10 ** 9)
    assert len(leaves) == 1
    assert leaves[0].ext_w == 1 and leaves[0].ext_h == 1
    verts, extra = mod.stitch_seams(leaves, h)
    assert len(verts) == 4 and extra == 0


def test_flat_terrain():
    flat = [[2.5] * 9 for _ in range(9)]
    root = mod.build_quadtree(flat, 6)
    # 全平时原始偏差为 0，但深度小量仍保证严格单调
    for n in _all_nodes(root):
        for c in n.children:
            assert n.error > c.error
    # 误差按深度有序 -> 选择仍然确定且完整覆盖
    a = mod.select_lod(root, 500)
    b = mod.select_lod(root, 500)
    assert [(n.x, n.y, n.size) for n in a] == \
           [(n.x, n.y, n.size) for n in b]
    assert len(_cells(a)) == 81
    verts, extra = mod.stitch_seams(a, flat)
    assert all(z == 2.5 for _, _, z in verts)
    assert _boundary_keys_equal(a)


def test_extreme_view_parameters():
    h = _load()
    near = {"distance": 1e-3, "fov": 1e-3, "screen_height": 4320}
    far = {"distance": 1e9, "fov": math.pi, "screen_height": 320}
    root_near = mod.build_quadtree(h, 4, view=near)
    root_far = mod.build_quadtree(h, 4, view=far)
    near_leaves = mod.select_lod(root_near, 10 ** 9, sse_threshold=1.0)
    far_leaves = mod.select_lod(root_far, 10 ** 9, sse_threshold=1.0)
    # 极近视角误差被放大 -> 更细；极远视角冻结在根附近
    assert len(near_leaves) >= len(far_leaves)
    assert len(_cells(near_leaves)) == 64
    assert len(_cells(far_leaves)) == 64
    # 极端参数下两次结果一致
    again = mod.select_lod(
        mod.build_quadtree(h, 4, view=far), 10 ** 9, sse_threshold=1.0)
    assert [(n.x, n.y, n.size) for n in far_leaves] == \
           [(n.x, n.y, n.size) for n in again]
    # 无阈值时视角不改变预算选择
    plain = mod.select_lod(mod.build_quadtree(h, 4), 10 ** 9)
    viewed = mod.select_lod(root_near, 10 ** 9)
    assert [(n.x, n.y, n.size) for n in plain] == \
           [(n.x, n.y, n.size) for n in viewed]


def test_dict_heightmap_input_accepted():
    d = {"size": 8, "heights": _load()}
    root = mod.build_quadtree(d, 3)
    leaves = mod.select_lod(root, 100)
    assert len(_cells(leaves)) == 64


# ---------- 规模 / 性能 ----------

def test_256_performance_and_scale():
    import random
    random.seed(42)
    n = 256
    h = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            h[j][i] = (50.0 * math.sin(i / 13.0) * math.cos(j / 17.0)
                       + 20.0 * math.sin((i + j) / 7.0)
                       + 5.0 * random.random())

    t0 = time.perf_counter()
    root = mod.build_quadtree(h, 8)
    leaves = mod.select_lod(root, 20000)
    verts, extra = mod.stitch_seams(leaves, h)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0
    tri = sum(n.tri_cost() for n in leaves)
    assert tri <= 20000
    assert len(_cells(leaves)) == n * n
    assert _boundary_keys_equal(leaves)

    # 预算翻倍：叶子明显增多，且新预算同样被充分使用
    leaves2 = mod.select_lod(root, 40000)
    tri2 = sum(n.tri_cost() for n in leaves2)
    assert tri2 <= 40000
    assert len(leaves2) > len(leaves)
    assert tri2 > tri
    # 用不完的剩余额度必须连最便宜的一次细分都买不起
    cheapest = min(
        (sum(c.tri_cost() for c in nd.children) - nd.tri_cost()
         for nd in leaves2 if nd.children),
        default=40001)
    assert tri2 + cheapest > 40000
    assert isinstance(extra, int) and extra >= 0
    assert len(verts) > 0
