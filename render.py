#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import torchvision.io as io # temporary
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "new_repo_renders_SH3_with_correct_load_ply_{}".format(iteration), "renders") # temporary
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if (idx > 4):
            return # temporary
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            print("Warning: Cropping renderings and gts to right half of image due to train/test exposure handling.")
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        loaded_rendering = torchvision.io.read_image(os.path.join(render_path, '{0:05d}'.format(idx) + ".png")).float() / 255.0 # temporary
        # check difference, temporary
        loaded_rendering = loaded_rendering.to(rendering.device)
        diff = torch.abs(loaded_rendering - rendering).mean()
        if diff > 0.01:
            print("Warning: Saved rendering differs from original rendering by {}".format(diff.item()))
            print(rendering[:, 0:5, 0:5])
            print(loaded_rendering[:, 0:5, 0:5])
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # print SH coefficients of the 5th gaussian, temporary
        sh_coeffs_of_5th_gaussian = gaussians.sh_storage.sh_coeffs_flat[gaussians.sh_storage.gauss_offsets[4]:gaussians.sh_storage.gauss_offsets[4]+(gaussians.sh_storage.sh_degrees[4] + 1)**2]
        print("SH coeffs of 5th gaussian before modification:", sh_coeffs_of_5th_gaussian)
        print("Shape of SH coeffs flat:", sh_coeffs_of_5th_gaussian.shape)

        # temporary: Set all green and blue SH coeffs to zero
        print(gaussians.sh_storage.sh_coeffs_flat.shape)
        #gaussians.sh_storage.sh_coeffs_flat[:, 1:3] = 0.0
        


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)