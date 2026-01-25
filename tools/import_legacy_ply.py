#!/usr/bin/env python3
"""Import a legacy GraphDeco PLY (per-vertex f_dc_/f_rest_ or compact) and write a compact PLY
compatible with this repository's `GaussianModel` and `SHStorage`.

Usage:
    tools/import_legacy_ply.py --input path/to/legacy.ply --out_model_path /path/to/model --iteration 0

The script will create: /path/to/model/point_cloud/iteration_{iteration}/point_cloud.ply
in the compact format used by this repo.
"""
import os
import argparse
from plyfile import PlyData
from scene.gaussian_model import GaussianModel


def detect_max_sh_degree(ply_path):
    ply = PlyData.read(ply_path)
    props = [p.name for p in ply.elements[0].properties]
    # compact format
    if 'sh_degrees' in props:
        # read max degree
        sh_degs = PlyData.read(ply_path).elements[0]['sh_degrees']
        return int(max(sh_degs))
    # legacy dense format
    f_rest = [n for n in props if n.startswith('f_rest_')]
    if len(f_rest) >= 0:
        K = 1 + (len(f_rest) // 3)
        return int((K ** 0.5) - 1)
    # default fallback
    return 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to legacy PLY')
    parser.add_argument('--out_model_path', required=True, help='Destination model path')
    parser.add_argument('--iteration', default=0, type=int)
    parser.add_argument('--sh_degree', default=None, type=int, help='Optional: override inferred max SH degree')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    sh_deg = args.sh_degree if args.sh_degree is not None else detect_max_sh_degree(args.input)

    gm = GaussianModel(sh_deg)
    print(f"Loading PLY {args.input} into GaussianModel(max_sh_degree={sh_deg})")
    gm.load_ply(args.input)

    out_dir = os.path.join(args.out_model_path, 'point_cloud', f'iteration_{args.iteration}')
    os.makedirs(out_dir, exist_ok=True)
    out_ply = os.path.join(out_dir, 'point_cloud.ply')
    print(f"Saving compact PLY to {out_ply}")
    gm.save_ply(out_ply)


if __name__ == '__main__':
    main()
