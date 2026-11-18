# -*- coding: utf-8 -*-
import argparse
import json
import terrainlod as m


def main():
    ap = argparse.ArgumentParser(description="terrainlod")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.path, encoding="utf-8"))
    root = m.build_quadtree(d["heights"], 3)
    leaves = m.select_lod(root, 20000)
    print(json.dumps({"leaves": len(leaves)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
