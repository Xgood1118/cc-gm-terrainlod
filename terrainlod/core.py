# -*- coding: utf-8 -*-
"""terrainlod

分块式地形四叉树 LOD 选择器（仅使用 Python 标准库）。

核心模型
========
- 高度图是一张 ``rows * cols`` 的规则网格，行列数都不必是 2 的幂。
- 每个四叉树节点覆盖高度图上的一个半开矩形 *格子* 区域
  ``[x, x+ext_w) x [y, y+ext_h)``。叶子集合构成整块地形格子的
  一个无重叠、无空洞的划分。
- 节点用一张固定分辨率的粗网格（每侧最多 ``MESH_QUADS = 2`` 个
  四边形，即 2x2 个四边形 / 8 个三角形）近似其覆盖区域；``size``
  为该网格每侧四边形数（被地图边缘裁剪时取宽高方向的较大值），
  因此一个节点的三角形成本恰为 ``2 * size * size``，与测试中的
  统计口径一致。
- 网格顶点跨越格子边界取整数格子坐标：区域 ``[x, x+w)`` 的网格
  顶点横坐标为 ``x .. x+w``，于是相邻节点共享同一条边界上的顶点
  坐标，缝合时可以按坐标一一焊接。地形外缘的越界坐标在取高度时
  钳制到最后一个采样点。
- 细分一个节点 = 用 4 个（边缘处可能只有 2 个）覆盖更小区域的
  子节点替换它，叶子始终是整块地形的一个划分。

几何误差
========
节点的几何误差为覆盖区内每个高度采样点相对节点角点双线性插值
曲面的最大高度偏差；再与子节点误差取最大值、并叠加一个与深度
相关的极小量 ``ERROR_EPS``，保证 *父节点误差严格大于子节点*，
全平地形也不例外。

若通过 ``view`` 提供视角参数，还会为每个节点计算屏幕空间误差
SSE，``select_lod`` 可通过 ``sse_threshold`` 提前冻结足够远的
节点（极端视角参数下同样有确定结果）。
"""

from __future__ import annotations

import heapq
import math

__all__ = ["QuadNode", "build_quadtree", "select_lod", "stitch_seams",
           "find_seams"]

#: 单个节点网格每侧最多多少个四边形（分块 LOD 的块分辨率）。
MESH_QUADS = 2

#: 误差聚合时附加的微小量，保证误差序列严格单调。
ERROR_EPS = 1e-9


class QuadNode:
    """四叉树节点。

    公开属性：
      x, y    节点覆盖区左上角在高度图中的格子坐标
      size    节点网格每侧的四边形数（三角形成本 = 2*size*size）
      error   聚合后的几何误差（父节点严格大于子节点）
      ext_w, ext_h  覆盖区宽高（高度图格子数）
      depth   节点深度，根为 0
      sse     屏幕空间误差（未提供视角时为 0.0）
      order   构建时分配的确定性遍历序号
      children 子节点列表（叶子为空列表）
      stitched_keys 缝合阶段补入本节点网格的边界顶点键
    """

    def __init__(self, x, y, size, error, *, ext_w=None, ext_h=None,
                 depth=0, parent=None, order=0):
        self.x = x
        self.y = y
        self.size = size
        self.error = error
        self.ext_w = ext_w if ext_w is not None else size
        self.ext_h = ext_h if ext_h is not None else size
        self.depth = depth
        self.parent = parent
        self.sse = 0.0
        self.order = order
        self.children = []
        self.stitched_keys = []

    def is_leaf(self):
        """是否为四叉树叶子（未再细分）。"""
        return not self.children

    def tri_cost(self):
        """本节点作为 LOD 叶子时贡献的三角形数。"""
        return 2 * self.size * self.size

    def base_mesh_keys(self):
        """本节点粗网格自带的顶点键（不含缝合补点）。"""
        return _node_mesh_keys(self)

    def mesh_vertex_keys(self):
        """本节点网格的全部顶点键（含缝合补入的边界顶点），有序。"""
        return _node_mesh_keys(self) + list(self.stitched_keys)


def _as_rows(heightmap):
    """兼容 list[list[float]] 与 {"heights": [...]} 两种输入。"""
    if isinstance(heightmap, dict):
        rows = heightmap["heights"]
    else:
        rows = heightmap
    if not rows or not rows[0]:
        raise ValueError("heightmap 不能为空")
    return rows


def _sample_indices(origin, extent, quads):
    """网格顶点在高度图上的整数格子下标。

    顶点跨越格子边界：覆盖 ``[origin, origin+extent)`` 的网格，其
    顶点横坐标取 ``origin .. origin+extent``，共 ``quads+1`` 个
    互不相同的下标（extent >= quads 保证步长 >= 1）。
    """
    return tuple(origin + round(k * extent / quads)
                 for k in range(quads + 1))


def _node_mesh_keys(node):
    qw = min(MESH_QUADS, node.ext_w)
    qh = min(MESH_QUADS, node.ext_h)
    xs = _sample_indices(node.x, node.ext_w, qw)
    ys = _sample_indices(node.y, node.ext_h, qh)
    return [(i, j) for j in ys for i in xs]


def _raw_deviation(rows, x, y, w, h):
    """覆盖区内所有高度点相对角点双线性插值面的最大绝对偏差。"""
    h00 = rows[y][x]
    h10 = rows[y][x + w - 1]
    h01 = rows[y + h - 1][x]
    h11 = rows[y + h - 1][x + w - 1]
    wm1 = w - 1
    hm1 = h - 1
    worst = 0.0
    for j in range(y, y + h):
        v = 0.0 if hm1 == 0 else (j - y) / hm1
        row = rows[j]
        inv_v = 1.0 - v
        for i in range(x, x + w):
            u = 0.0 if wm1 == 0 else (i - x) / wm1
            top = h00 + (h10 - h00) * u
            bot = h01 + (h11 - h01) * u
            approx = top * inv_v + bot * v
            dev = abs(row[i] - approx)
            if dev > worst:
                worst = dev
    return worst


def _build_node(rows, x, y, w, h, depth, max_depth, parent,
                view, counter):
    qw = min(MESH_QUADS, w)
    qh = min(MESH_QUADS, h)
    node = QuadNode(x, y, max(qw, qh), 0.0, ext_w=w, ext_h=h,
                    depth=depth, parent=parent, order=counter[0])
    counter[0] += 1

    raw = _raw_deviation(rows, x, y, w, h)

    # 只有区域还能继续切分时才递归，杜绝零尺寸子节点。
    if depth < max_depth and (w > 1 or h > 1):
        w2 = (w + 1) // 2
        h2 = (h + 1) // 2
        spans_x = [(x, w2)]
        if w - w2 > 0:
            spans_x.append((x + w2, w - w2))
        spans_y = [(y, h2)]
        if h - h2 > 0:
            spans_y.append((y + h2, h - h2))
        for cy, ch in spans_y:
            for cx, cw in spans_x:
                child = _build_node(rows, cx, cy, cw, ch, depth + 1,
                                    max_depth, node, view, counter)
                node.children.append(child)

    child_worst = max((c.error for c in node.children), default=0.0)
    # 自底向上聚合 + 深度小量：父节点误差严格大于任一子节点。
    node.error = max(raw, child_worst) + ERROR_EPS

    if view is not None:
        dist = max(float(view.get("distance", 1.0)), 1e-9)
        screen = float(view.get("screen_height", 1080))
        fov = float(view.get("fov", math.pi / 3))
        denom = 2.0 * math.tan(max(fov, 1e-9) / 2.0) * dist
        node.sse = raw * screen / denom
    return node


def build_quadtree(heightmap, max_depth=6, *, view=None):
    """按几何误差构建四叉树并返回根节点。

    heightmap : 二维高度序列（list[list[float]]），行列数任意；
                也接受带 ``heights`` 键的 dict。
    max_depth : 最大细分深度。
    view      : 可选视角参数 ``{"distance", "fov", "screen_height"}``，
                提供后每个节点会计算屏幕空间误差 ``sse``。
    """
    rows = _as_rows(heightmap)
    cols = len(rows[0])
    total_rows = len(rows)
    counter = [0]
    return _build_node(rows, 0, 0, cols, total_rows, 0, max_depth,
                       None, view, counter)


def select_lod(root, tri_budget, *, sse_threshold=0.0):
    """在三角形预算内选出确定性的叶子节点集合。

    策略：从根节点开始，始终细分当前误差最大、且 SSE 达到阈值、
    且细分后总成本仍不超预算的节点；不满足任一条件的节点降级保留
    为叶子。预算连根节点都无法容纳时，返回根节点（表示下限）。

    返回的叶子按构建序号排序，同一预算下两次调用结果完全一致。
    """
    budget = int(tri_budget)
    heap = [(-root.error, root.order, root)]
    frontier = {id(root): root}
    total = root.tri_cost()

    while heap:
        _, _, node = heapq.heappop(heap)
        if id(node) not in frontier:
            continue  # 已被细分，旧条目忽略。
        if not node.children:
            continue  # 到达四叉树最大深度，无法再细分。
        if sse_threshold > 0.0 and node.sse < sse_threshold:
            continue  # 视角下足够远，冻结为叶子。
        delta = sum(c.tri_cost() for c in node.children) - node.tri_cost()
        if total + delta > budget:
            continue  # 预算不足，降级保留（后续细分只会更贵）。
        total += delta
        del frontier[id(node)]
        for child in node.children:
            frontier[id(child)] = child
            heapq.heappush(heap, (-child.error, child.order, child))

    return sorted(frontier.values(), key=lambda n: n.order)


def find_seams(leaves):
    """枚举叶子之间的层级差边界，返回确定性的接缝列表。

    每个接缝为 ``(nodes, vertical, boundary, keys)``：
      nodes    : 共享该边界的全部叶子（深度不完全相同）
      vertical : True 表示竖直边界（共享列坐标），False 表示水平
      boundary : 共享边界的格子坐标（列号或行号）
      keys     : 边界两侧网格顶点键的并集（缝合后双方必须一致）
    同深度相邻不产生接缝，地形外缘不产生接缝。
    """
    leaves = sorted(leaves, key=lambda n: n.order)
    grid_cols = max(n.x + n.ext_w for n in leaves)
    grid_rows = max(n.y + n.ext_h for n in leaves)

    # 格子 -> 叶子 的占用索引，构建复杂度 O(地形格子数)。
    owner = {}
    for node in leaves:
        for j in range(node.y, node.y + node.ext_h):
            for i in range(node.x, node.x + node.ext_w):
                owner[(i, j)] = node

    records = {}
    for node in leaves:
        x, y = node.x, node.y
        w, h = node.ext_w, node.ext_h
        edges = (
            (True, x, x - 1, range(y, y + h)),       # 左边界
            (True, x + w, x + w, range(y, y + h)),   # 右边界
            (False, y, y - 1, range(x, x + w)),      # 上边界
            (False, y + h, y + h, range(x, x + w)),  # 下边界
        )
        for vertical, boundary, neighbor_coord, span in edges:
            if neighbor_coord < 0 or neighbor_coord >= (
                    grid_cols if vertical else grid_rows):
                continue  # 地形外缘。
            neighbors = []
            seen = set()
            for c in span:
                nb = owner.get(
                    (neighbor_coord, c) if vertical else (c, neighbor_coord))
                if nb is not None and id(nb) not in seen:
                    seen.add(id(nb))
                    neighbors.append(nb)
            if not neighbors:
                continue
            if all(nb.depth == node.depth for nb in neighbors):
                continue  # 同深度相邻，无层级差。
            token = (vertical, boundary)
            entry = records.setdefault(token, set())
            entry.add(id(node))
            for nb in neighbors:
                entry.add(id(nb))

    index = {id(n): n for n in leaves}
    seams = []
    for (vertical, boundary), ids in records.items():
        nodes = [index[k] for k in sorted(ids, key=lambda k: index[k].order)]
        keys = set()
        for node in nodes:
            for i, j in node.base_mesh_keys():
                if vertical and i == boundary:
                    keys.add((i, j))
                elif not vertical and j == boundary:
                    keys.add((i, j))
        seams.append((nodes, vertical, boundary, tuple(sorted(keys))))
    seams.sort(key=lambda s: (not s[1], s[2],
                              tuple(n.order for n in s[0])))
    return seams


def stitch_seams(leaves, heightmap):
    """缝合不同层级叶子之间的接缝。

    对每条层级差边界，把对侧存在而本侧网格缺失的边界顶点补入对应
    叶子的网格——边界顶点一一对应后，T 型接点被焊接消除，裂缝不再
    产生。

    返回 ``(verts, extra)``：
      verts : 全局去重后的全部网格顶点 ``(i, j, height)``，
              按 (j, i) 确定性排序；地形外缘越界坐标钳制取高度
      extra : 缝合额外引入的顶点数（int），即向各叶子网格插入
              边界顶点的总次数，每个 T 型接点对应一次插入
    """
    rows = _as_rows(heightmap)
    leaves = sorted(leaves, key=lambda n: n.order)
    grid_cols = len(rows[0])
    grid_rows = len(rows)

    # 重置上一次缝合可能留下的状态，保证可重复调用。
    for node in leaves:
        node.stitched_keys = []

    extra = 0
    for nodes, _vertical, _boundary, keys in find_seams(leaves):
        for node in nodes:
            have = set(node.base_mesh_keys()) | set(node.stitched_keys)
            for key in keys:
                if key not in have:
                    node.stitched_keys.append(key)
                    have.add(key)
                    extra += 1

    all_keys = set()
    for node in leaves:
        all_keys.update(node.base_mesh_keys())
        all_keys.update(node.stitched_keys)

    def _height(i, j):
        ci = min(max(i, 0), grid_cols - 1)
        cj = min(max(j, 0), grid_rows - 1)
        return rows[cj][ci]

    verts = [(i, j, _height(i, j))
             for i, j in sorted(all_keys, key=lambda k: (k[1], k[0]))]
    return verts, extra
