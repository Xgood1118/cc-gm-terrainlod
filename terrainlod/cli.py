# -*- coding: utf-8 -*-
import argparse
import json

import terrainlod as m


def main():
    ap = argparse.ArgumentParser(description="terrainlod 四叉树 LOD 选择器")
    ap.add_argument("path", help="高度图 JSON 路径")
    ap.add_argument("--out", default=None, help="选择结果输出路径")
    ap.add_argument("--budget", type=int, default=20000,
                    help="三角形预算（默认 20000）")
    ap.add_argument("--depth", type=int, default=3,
                    help="四叉树最大深度（默认 3）")
    a = ap.parse_args()

    with open(a.path, encoding="utf-8") as f:
        data = json.load(f)
    heights = data["heights"] if isinstance(data, dict) else data

    root = m.build_quadtree(heights, a.depth)
    leaves = m.select_lod(root, a.budget)
    verts, extra = m.stitch_seams(leaves, heights)

    triangles = sum(n.tri_cost() for n in leaves)
    result = {
        "budget": a.budget,
        "leaf_count": len(leaves),
        "triangles": triangles,
        "budget_usage": round(triangles / a.budget, 6),
        "stitch_extra_vertices": extra,
        "total_vertices": len(verts),
        "leaves": [
            {"x": n.x, "y": n.y, "size": n.size,
             "ext_w": n.ext_w, "ext_h": n.ext_h,
             "depth": n.depth, "error": n.error}
            for n in leaves
        ],
    }

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(
        {k: v for k, v in result.items() if k != "leaves"},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
