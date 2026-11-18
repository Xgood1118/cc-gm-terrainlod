# -*- coding: utf-8 -*-
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import terrainlod as mod  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load():
    d = json.load(open(os.path.join(DATA, "heightmap.json"), encoding="utf-8"))
    return d["heights"]


def test_root_covers_all():
    h = _load()
    root = mod.build_quadtree(h, 3)
    leaves = mod.select_lod(root, 10 ** 9)
    assert leaves and all(hasattr(n, "x") and hasattr(n, "size") for n in leaves)


def test_error_monotone():
    h = _load()
    root = mod.build_quadtree(h, 3)

    def walk(n):
        yield n
        for c in (n.children or []):
            yield from walk(c)

    for n in walk(root):
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
    assert [(n.x, n.y, n.size) for n in a] == [(n.x, n.y, n.size) for n in b]


def test_stitch_returns_verts():
    h = _load()
    root = mod.build_quadtree(h, 3)
    leaves = mod.select_lod(root, 100)
    verts, extra = mod.stitch_seams(leaves, h)
    assert verts and isinstance(extra, int) and extra >= 0
