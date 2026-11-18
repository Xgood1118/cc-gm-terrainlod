# -*- coding: utf-8 -*-
"""terrainlod"""

from __future__ import annotations

__all__ = ["QuadNode", "build_quadtree", "select_lod", "stitch_seams"]


class QuadNode:
    """QuadNode"""

    def __init__(self, x, y, size, error):
        """四叉树节点"""
        raise NotImplementedError("QuadNode.__init__ 尚未实现")

    def children(self):
        """返回子节点列表"""
        raise NotImplementedError("QuadNode.children 尚未实现")


def build_quadtree(heightmap, max_depth=6):
    """按几何误差建四叉树"""
    raise NotImplementedError("build_quadtree 尚未实现")


def select_lod(root, tri_budget):
    """在预算内选出叶子集合"""
    raise NotImplementedError("select_lod 尚未实现")


def stitch_seams(leaves, heightmap):
    """缝合不同层级的接缝，返回 (顶点, 额外顶点数)"""
    raise NotImplementedError("stitch_seams 尚未实现")
