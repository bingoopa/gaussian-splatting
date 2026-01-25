#!/usr/bin/env python3
"""Test utility: load a trained PLY (compact or legacy), render the test views and report PSNR/SSIM.

Usage:
	python3 scene/test.py --ply /path/to/point_cloud.ply --source_path /path/to/dataset --model_path /tmp/model_root --iteration 0

The script will copy the provided PLY into the model layout under
`{model_path}/point_cloud/iteration_{iteration}/point_cloud.ply` and then use the existing
Scene and renderer to produce renders and compute metrics against the ground-truth images
found under `source_path`.
"""

import os
import shutil
import argparse
import torch
from types import SimpleNamespace
from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render as grender
from utils.image_utils import psnr
from utils.loss_utils import ssim


def prepare_model_ply(input_ply: str, model_path: str, iteration: int):
	out_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
	os.makedirs(out_dir, exist_ok=True)
	out_ply = os.path.join(out_dir, "point_cloud.ply")
	shutil.copyfile(input_ply, out_ply)
	return out_ply


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ply', required=True, help='Path to input PLY (compact or legacy formats)')
	parser.add_argument('--source_path', required=True, help='Path to original dataset (Colmap/Blender folder)')
	parser.add_argument('--model_path', required=True, help='Path where the model layout will be created')
	parser.add_argument('--iteration', type=int, default=0)
	parser.add_argument('--sh_degree', type=int, default=3, help='SH degree to initialize GaussianModel with')
	parser.add_argument('--white_background', action='store_true')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# copy provided PLY into model layout
	prepare_model_ply(args.ply, args.model_path, args.iteration)

	# create GaussianModel and Scene
	gauss = GaussianModel(args.sh_degree)

	ds_args = SimpleNamespace()
	ds_args.model_path = os.path.abspath(args.model_path)
	ds_args.source_path = os.path.abspath(args.source_path)
	ds_args.images = None
	ds_args.white_background = args.white_background
	ds_args.eval = True
	ds_args.resolution = -1
	ds_args.data_device = 'cuda' if torch.cuda.is_available() else 'cpu'

	print(f"Loading scene from source {ds_args.source_path} using model at {ds_args.model_path} (iter {args.iteration})")
	scene = Scene(ds_args, gauss, load_iteration=args.iteration, shuffle=False)

	pipeline = SimpleNamespace()
	pipeline.convert_SHs_python = False
	pipeline.compute_cov3D_python = False
	pipeline.debug = False

	bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
	background = torch.tensor(bg_color, dtype=torch.float32, device=device)

	cameras = scene.getTestCameras()
	if len(cameras) == 0:
		print("No test cameras found for this scene. Make sure --source_path points to the dataset and it contains images.")
		return

	psnrs = []
	ssims = []

	with torch.no_grad():
		for view in cameras:
			out = grender(view, gauss, pipeline, background)
			rendering = out["render"]
			gt = view.original_image[0:3, :, :].to(rendering.device)
			# ensure same shape
			if rendering.shape != gt.shape:
				# simple center-crop or resize could be applied, but here we assert equality
				min_h = min(rendering.shape[1], gt.shape[1])
				min_w = min(rendering.shape[2], gt.shape[2])
				rendering = rendering[:, :min_h, :min_w]
				gt = gt[:, :min_h, :min_w]

			s = ssim(rendering.unsqueeze(0), gt.unsqueeze(0)).item()
			p = psnr(rendering.unsqueeze(0), gt.unsqueeze(0)).item()
			ssims.append(s)
			psnrs.append(p)
			print(f"View {view.image_name}: PSNR={p:.4f}, SSIM={s:.4f}")

	import numpy as np
	print("--- Summary ---")
	print(f"Mean PSNR: {float(np.mean(psnrs)):.4f}")
	print(f"Mean SSIM: {float(np.mean(ssims)):.4f}")


if __name__ == '__main__':
	main()

