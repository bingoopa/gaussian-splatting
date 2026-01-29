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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import LPIPS
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func # Newly added from original repo
#from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import numpy as np

# Newly added from original repo 
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Fused SSIM available")
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Gaussian Adam available")
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
# Für die Farbkodierung der Farbgradienten
# später SH-update einbauen und matplotlib nicht mehr importieren, weil nur für Visualisierung
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

    #BENNET: Für log dateei, auskommentieren, wenn du nicht brauchst
class Tee(object):
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_gui = False, sh_percentage=[0, 0], color_grad_stats=False, need_color_grads=False, visualize_degrees = False, visualize_gradients = False, visualize_gradients_iters = [0] ,adaptive_sh = False, psnr_ssim_iterations=[], lpips_iterations=[]):
    print(f"positions: init={opt.position_lr_init} final={opt.position_lr_final} delay_mult={opt.position_lr_delay_mult} max_steps={opt.position_lr_max_steps}")
    print(f"feature={opt.feature_lr} opacity={opt.opacity_lr} scaling={opt.scaling_lr} rotation={opt.rotation_lr}")
    print(f"densification: interval={opt.densification_interval} from={opt.densify_from_iter} until={opt.densify_until_iter} grad_threshold={opt.densify_grad_threshold}")
    #print("psnr_ssim_iterations={}".format(psnr_ssim_iterations))

    # Newly added from original repo
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # New: initialize sh_degrees randomly if needed
    if need_color_grads or color_grad_stats:
        gaussians.set_random_sh_degrees()

    # Temporary: print initializing infos to compare with original repo
    print("\n[ INFO ] Starting training with the following configuration:")
    print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"SH Degree: {gaussians.active_sh_degree}")
    print(f"Using optimizer: {opt.optimizer_type}")
    print(f"Using train_test_exposure: {dataset.train_test_exp}")
    print(f"SH storage size: {gaussians.sh_storage.num_gaussians}")
    print(f"SH storage size: {gaussians.sh_storage.num_gauss}")
    print(f"SH storage size (total coefficients): {gaussians.sh_storage.sh_coeffs_flat.shape[0], gaussians.sh_storage.sh_coeffs_flat.shape[1]}")
    print(f"average SH coeffs per gaussian: {gaussians.sh_storage.sh_coeffs_flat.shape[0] / gaussians.sh_storage.num_gaussians}")

    # New: introduce early stopping
    # ---- Early stopping state (define once, e.g. before training loop) ----
    global best_test_psnr, best_iter, patience, min_delta, early_stop
    best_test_psnr = -1e9
    best_iter = -1
    patience = 5000          # how many iterations to wait
    min_delta = 0.01 
    early_stop = False


    # Compute color gradients if needed (explicitly requested or for adaptive SH)
    need_color_grads = need_color_grads or adaptive_sh

    '''schedule 1:
    first_phase_start = 2000 # standard: 5000
    second_phase_start = 6000 # standard: 10000
    third_phase_start = 12000 # standard: 20000
    third_phase_end = opt.iterations - 1 # standard: opt.iterations - 1
    average_gradients_over = 100 
    first_phase_ratio = 0.01 # standard: 0.005
    second_phase_ratio = 0.015 # standard: 0.01
    third_phase_ratio = 0.005 # standard: 0.002
    first_phase_frequency = 250 # standard: 500
    second_phase_frequency = 250 # standard: 500
    third_phase_frequency = 1500 # standard: 1000
    '''
    '''
    schedule 2:
    first_phase_start = 2000 # standard: 5000
    second_phase_start = 6000 # standard: 10000
    third_phase_start = 12000 # standard: 20000
    third_phase_end = opt.iterations - 1 # standard: opt.iterations - 1
    average_gradients_over = 100 
    first_phase_ratio = 0.01 # standard: 0.005
    second_phase_ratio = 0.015 # standard: 0.01
    third_phase_ratio = 0.005 # standard: 0.002
    first_phase_frequency = 250 # standard: 500
    second_phase_frequency = 250 # standard: 500
    third_phase_frequency = 1500 # standard: 1000
    '''
    # schedule 4:
    global schedule_name 
    # schedule 5:
    schedule_name = "schedule2.3"
    first_phase_start = 1000
    second_phase_start = 6000 
    third_phase_start = 12000
    third_phase_end = 25000

    average_gradients_over = 100 

    first_phase_ratio = 0.02 
    second_phase_ratio = 0.015 
    third_phase_ratio = 0.005 

    first_phase_frequency = 750 
    second_phase_frequency = 750 
    third_phase_frequency = 1500 

    first_phase_max_degree = 1
    second_phase_max_degree = 3
    third_phase_max_degree = 3


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Newly added from original repo
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    if dataset.train_test_exp:
        print("Use training exposure is true. Consider checking it in comparison with original repo.")

    #viewpoint_stack = None 
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            print("Total number of gaussians: ", gaussians._xyz.shape[0])

        # Scannet: for _ in range(opt.optimizer_step_interval):

        # Old version : pick random camera
        # Pick a random Camera
        #if not viewpoint_stack:
        #    viewpoint_stack = scene.getTrainCameras().copy()
        #viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Newly added from original repo: shuffle every epoch
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]



        # Apply alpha mask from original repo
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # Newly added from original repo:
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) # Newly added from original repo
        # loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # old version
        #loss /= opt.optimizer_step_interval     # Gradient accumulation

        # Depth regularization newly added from original repo
        Ll1depth_pure = 0.0
        #if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        #    invDepth = render_pkg["depth"]
        #    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        #    depth_mask = viewpoint_cam.depth_mask.cuda()

        #    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        #    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
        #    loss += Ll1depth
        #    Ll1depth = Ll1depth.item()
        #else:
        Ll1depth = 0

        loss.backward()

        # Here we print the gradients of the first 5 gaussians for debugging, temporary
        if iteration == 0 or iteration == -1:
            continue
            print("Positions of first 5 gaussians at iteration {}:".format(iteration))
            print(gaussians.get_xyz[:5])
            print("Positional Gradients of first 5 gaussians at iteration {}:".format(iteration))
            print(gaussians.get_xyz.grad[:5])
            print("SH coefficient Gradients of first 5 gaussians at iteration {}:".format(iteration))
            print(gaussians.sh_storage.sh_coeffs_flat.grad[:5], "with shape", gaussians.sh_storage.sh_coeffs_flat.grad[:5].shape)
        if iteration < -1:
            continue


        iter_end.record()

        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # newly added from original repo
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"}) # old version
                # Newly added from original repo:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), psnr_ssim_iterations, lpips_iterations)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if early_stop:
                print(f"\n[EARLY STOP] Stopping training at iteration {iteration} due to no PSNR improvement.")
                #print(f"[PSNR] iter={iteration}, test={psnr_test:.3f}, best={best_test_psnr:.3f}")

                break

            # New: compute color gradient stats/ adapt SH degrees
            """
            if color_grad_stats or need_color_grads:
                color_grad_interval = 1000 # only works properly if color_grad_interval > 50, 
                # We average of the last 50 iters before densification -> we want to avoid effects of densification on the stats
                if iteration % color_grad_interval >= color_grad_interval - 50 and iteration >= opt.densify_from_iter:
                    gaussians.cumulate_color_gradients()
                if color_grad_stats and (iteration % color_grad_interval == 0):
                    # Compute color gradient stats, in the current configuration at iterations 500, 5500, 10500, ...
                    gaussians.getColorGradStats(iteration)
                if adaptive_sh and (iteration % color_grad_interval == 0) and iteration >= 5000 and iteration <= opt.iterations - 10000:
                    # Update SH degrees based on accumulated color gradients
                    gaussians.increase_sh_degree_based_on_color_grads()
                    gaussians.get_sh_degree_distribution()
            """

            # New: Aggregate color gradients at any iteration (below they are reset after every average_gradients_over iters)
            if need_color_grads: #and iteration >= first_phase_start - average_gradients_over:
                gaussians.cumulate_color_gradients()

            
            # New scheduling of SH degree increase based on color gradients
            if iteration >= first_phase_start and iteration < second_phase_start:
                if (iteration - first_phase_start) % first_phase_frequency == 0:
                    gaussians.increase_sh_degree_based_on_color_grads(ratio=first_phase_ratio, maximum_degree=first_phase_max_degree, iteration = iteration, cool_down_iter=50)
                    gaussians.get_sh_degree_distribution(iteration=iteration, schedule_name=schedule_name)
            elif iteration >= second_phase_start and iteration < third_phase_start:
                if (iteration - second_phase_start) % second_phase_frequency == 0:
                    gaussians.increase_sh_degree_based_on_color_grads(ratio=second_phase_ratio, maximum_degree=second_phase_max_degree, iteration = iteration, cool_down_iter=50)
                    gaussians.get_sh_degree_distribution(iteration=iteration, schedule_name=schedule_name)
            elif iteration >= third_phase_start and iteration <= third_phase_end:
                if (iteration - third_phase_start) % third_phase_frequency == 0:
                    gaussians.increase_sh_degree_based_on_color_grads(ratio=third_phase_ratio, maximum_degree=third_phase_max_degree, iteration = iteration, cool_down_iter=50)
                    gaussians.get_sh_degree_distribution(iteration=iteration, schedule_name=schedule_name)
            

            # ------------------------------
            # Save color-gradient visualization at specific iterations
            # ------------------------------
            if args.visualize_gradients and iteration in args.visualize_gradients_iters:
                outdir = os.path.join(dataset.model_path, "color_gradients_train", f"iter_{iteration}")
                save_color_gradient_visualization(scene, render, (pipe, background), outdir, iteration)
          
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Debugging densification: temporary
            if iteration == 1:
                gaussians.tmp_radii = radii
                gaussians.densify_and_clone(gaussians.xyz_gradient_accum / gaussians.denom, opt.densify_grad_threshold, scene.cameras_extent, debug=True)
                print("Position of old second gaussian: ", gaussians.get_xyz[1:2])
                print("Position of new cloned gaussian: ", gaussians.get_xyz[-1:])
                print("SH Coeffs of old second gaussian: ", gaussians.sh_storage.sh_coeffs_flat[1:2])
                print("SH Coeffs of new cloned gaussian: ", gaussians.sh_storage.sh_coeffs_flat[-1:])
            
            # New: reset color_gradient accumulator
            #if (color_grad_stats or need_color_grads) and iteration % color_grad_interval == opt.densify_from_iter:
                #gaussians.color_gradients_postfix()
            
            # New: reset color gradient accumulators after average_gradients_over iterations
            if need_color_grads and iteration % average_gradients_over == 0:
                gaussians.color_gradients_postfix()

            # Optimizer step # old version
            #if iteration < opt.iterations:
            #    gaussians.optimizer.step()
            #    gaussians.optimizer.zero_grad(set_to_none = True)

            # Newly added from original repo:
            # Optimizer step
            if iteration < opt.iterations:
                #gaussians.exposure_optimizer.step()
                #gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                #if use_sparse_adam:
                #    visible = radii > 0
                #    gaussians.optimizer.step(visible, radii.shape[0])
                #    gaussians.optimizer.zero_grad(set_to_none = True)
                #else:
                # Use custom optimizer step with proper SH coefficient learning rate scaling
                # This integrates the different LRs (DC vs rest) into Adam's bias correction
                gaussians.optimizer_step_with_scaled_sh_lr()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # New: Increase SH degree based on percentage schedule
            if sh_percentage[0] > 0 and sh_percentage[1] is not None:
                if iteration % sh_percentage[1] == 0:
                    gaussians.randomly_increase_sh_degrees_by_one(sh_percentage[0]/100.0)
                    gaussians.get_sh_degree_distribution()

    # New: save color gradient stats to CSV
    if color_grad_stats:
        print("Gleich werdn die color gradient stats gespeichert")
        gaussians.saveColorGradStatsToCSV(os.path.join(dataset.model_path, "color_gradient_stats.csv"))
    eval_and_save(dataset.model_path, scene, render, (pipe, background))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # ------------------------------
    # BENNET: Logger aktivieren
    # ------------------------------
    log_path = os.path.join(args.model_path, "log.txt")
    log_file = open(log_path, "a")
    sys.stdout = Tee(log_file, sys.__stdout__)
    sys.stderr = Tee(log_file, sys.__stderr__)
    print("Logging to:", log_path)
    #Ende neuer Code
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, psnr_ssim_iterations, lpips_iterations):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    compute_psnr_ssim = (iteration in psnr_ssim_iterations)
    compute_lpips = (iteration in lpips_iterations)

    # Report test and samples of training set
    if compute_lpips or compute_psnr_ssim: #(iteration + 1) % testing_iterations[0] == 0:
        #print("Evaluating lpips or psnr/ssim at iteration {}".format(iteration))
        if compute_lpips:
            lpips = LPIPS(net_type='vgg').to("cuda")
            torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            # {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
            {'name': 'train', 'cameras' : scene.getTrainCameras()[::10]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        image = image.unsqueeze(0)
                        gt_image = gt_image.unsqueeze(0)
                        viz_image = torch.nn.functional.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), viz_image, global_step=iteration)
                        # if iteration == testing_iterations[0]:
                        #     tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    if compute_psnr_ssim:
                        psnr_val = psnr(image, gt_image).mean()
                        ssim_val = ssim(image, gt_image)
                    if compute_lpips:
                        lpips_val = lpips(image, gt_image)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    # psnr_test += psnr(image, gt_image).mean().double()
                    if compute_psnr_ssim:
                        psnr_test += psnr_val.item()
                        ssim_test += ssim_val.item()
                    if compute_lpips:
                        lpips_test += lpips_val.item()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                global schedule_name
                with open(f"test_metrics_{schedule_name}.csv", "a") as f:
                    f.write(f"{iteration},{psnr_test},{ssim_test},{lpips_test}\n")


                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

                if config['name'] == 'test' and compute_psnr_ssim:
                    print("[EARLY STOP] Checking early stopping criteria at iter {}".format(iteration))
                    global best_test_psnr, best_iter, patience, min_delta, early_stop
                    if psnr_test > best_test_psnr + min_delta:
                        best_test_psnr = psnr_test
                        best_iter = iteration
                    else:
                        if iteration - best_iter > patience:
                            print(
                                f"[EARLY STOP] No PSNR improvement > {min_delta} "
                                f"for {patience} iterations. "
                                f"Best PSNR {best_test_psnr:.3f} at iter {best_iter}."
                            )
                            early_stop = True
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        

@torch.no_grad()
def eval_and_save(model_path, scene: Scene, renderFunc, renderArgs):
    cameras = scene.getTestCameras()
    assert len(cameras) > 0, "No test cameras found"
    lpips = LPIPS(net_type='vgg').to("cuda")

    all_psnr = []
    all_ssim = []
    all_lpips = []
    render_out_dir = os.path.join(model_path, "eval")
    print("Output folder: {}".format(render_out_dir))
    combine_out_dir = os.path.join(model_path, "combine")
    os.makedirs(render_out_dir, exist_ok=True)
    os.makedirs(combine_out_dir, exist_ok=True)
    """
    if args.visualize_gradients:  # BENNET: Ordner für color gradients visualisierung 
        cgrad_out_dir = os.path.join(model_path, "color_gradients") 
        os.makedirs(cgrad_out_dir, exist_ok=True)
        ply_path = os.path.join(cgrad_out_dir, "color_gradients.ply")
    """
    if args.visualize_degrees: # BENNET: Ordner für sh-degrees visualisierung
        shdeg_out_dir = os.path.join(model_path, "sh_degrees")
        os.makedirs(shdeg_out_dir, exist_ok=True) 

    for idx, viewpoint in enumerate(cameras):        
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0) 
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        save_path = os.path.join(render_out_dir, viewpoint.image_name + ".png")
        image_np = (image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
        gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
        Image.fromarray(image_np).save(save_path)
        save_path = os.path.join(combine_out_dir, viewpoint.image_name + ".png")
        Image.fromarray(np.concatenate((image_np, gt_image_np), axis=1)).save(save_path)

        image = image.unsqueeze(0)
        gt_image = gt_image.unsqueeze(0)
        psnr_val = psnr(image, gt_image)
        ssim_val = ssim(image, gt_image)
        lpips_val = lpips(image, gt_image)

        all_psnr.append(psnr_val.item())
        all_ssim.append(ssim_val.item())
        all_lpips.append(lpips_val.item())

    if args.visualize_degrees: 
        degree_colors = torch.tensor(scene.gaussians.get_sh_degree_colors(), device="cuda", dtype=torch.float32)
        for idx, viewpoint in enumerate(cameras):  # BENNET: Sh-Degrees rendern
            image_shdeg = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=degree_colors)["render"], 0.0, 1.0)
            image_shdeg_np = (image_shdeg * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
            save_path = os.path.join(shdeg_out_dir, viewpoint.image_name + ".png")
            Image.fromarray(np.concatenate((image_shdeg_np, gt_image_np), axis=1)).save(save_path)

    """ super unnötig, da gradients am ende 0 
    if args.visualize_gradients:
        color_np = get_colors_for_color_grad_vis(scene.gaussians)
        color_tensor = torch.tensor(color_np, device="cuda", dtype=torch.float32) / 255.0
        for idx, viewpoint in enumerate(cameras):  # BENNET: Color gradients rendern
            image_color = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=color_tensor)["render"], 0.0, 1.0)
            image_color_np = (image_color * 255.0).permute(1,2,0).cpu().byte().numpy()
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
            save_path = os.path.join(cgrad_out_dir, f"{viewpoint.image_name}.png")
            Image.fromarray(np.concatenate((image_color_np, gt_image_np), axis=1)).save(save_path)
    """

    print("Evaluation results:")
    print("PSNR: {}".format(np.mean(all_psnr)))
    print("SSIM: {}".format(np.mean(all_ssim)))
    print("LPIPS: {}".format(np.mean(all_lpips)))

def get_colors_for_color_grad_vis(gaussians):
    P = gaussians.get_xyz.shape[0]
    # Falls noch keine Gradienten gesammelt wurden
    if gaussians.color_denom == 0:
        return np.zeros((P, 3), dtype=np.uint8)
    # Sicherstellen, dass die Arrays die richtige Größe haben
    dc = gaussians.accum_color_grads_dc
    rest = gaussians.accum_color_grads_rest
    # Falls die Gradienten noch von einer älteren Iteration stammen
    if dc.shape[0] != P or rest.shape[0] != P:
        # Padding mit Nullen auf aktuelle Länge
        new_dc = torch.zeros((P,), device="cuda")
        new_rest = torch.zeros((P,), device="cuda")
        L = min(P, dc.shape[0])
        new_dc[:L] = dc[:L]
        new_rest[:L] = rest[:L]
        dc = new_dc
        rest = new_rest
    # Normierung
    grad_mag = dc + rest
    grad_mag = grad_mag / max(gaussians.color_denom, 1)
    grad_mag = grad_mag.cpu().numpy()
    # Normalisieren auf [0,1]
    grad_mag = np.maximum(grad_mag, 1e-12)     # kleine Werte schützen
    grad_mag_log = np.log(grad_mag)            # Logarithmische Streckung
    vmin = np.percentile(grad_mag_log, 1)      # robuste Min/Max
    vmax = np.percentile(grad_mag_log, 99)
    grad_mag_norm = np.clip((grad_mag_log - vmin) / (vmax - vmin), 0, 1)  # Normalisieren auf [0,1]
    # Farbmap
    colors = (cm.viridis(grad_mag_norm)[:, :3] * 255).astype(np.uint8)
    return colors

def save_color_gradient_visualization(scene, render, renderArgs, outdir, iteration): #Color gradients: Farben setzen, Szene rendern, speichern
    gaussians = scene.gaussians
    colors = torch.tensor(get_colors_for_color_grad_vis(gaussians), device="cuda", dtype=torch.float32) / 255.0
    os.makedirs(outdir, exist_ok=True)
    cameras = scene.getTestCameras()
    for viewpoint in cameras:
        image = torch.clamp(render(viewpoint, gaussians, *renderArgs, override_color=colors)["render"], 0.0, 1.0)
        img_np = (image * 255).permute(1, 2, 0).cpu().byte().numpy()
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
        save_path = os.path.join(outdir, f"{viewpoint.image_name}.png")
        Image.fromarray(np.concatenate((img_np, gt_image_np), axis=1)).save(save_path)
    print(f"[ColorGradients] Saved visualization at iter {iteration} → {outdir}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # New argument for SH percentage increase
    parser.add_argument("--sh_percentage", nargs="+", type=int, default=[0, 0]) # first: percentage, second: interval -> increases sh-degree‚ every interval iterations
    parser.add_argument("--color_grad_stats", type = bool, default=False, help="Whether to collect color gradient statistics during training")
    parser.add_argument("--need_color_grads", type = bool, default=False, help="Whether to track color gradients during training")
    # New argument um SH degrees zu visualisueren
    parser.add_argument("--visualize_degrees", action="store_true", help="Visualize Gaussians by SH-degree at the end of training")
    parser.add_argument("--visualize_gradients", action="store_true", help="Visualize Gaussians by SH-degree at the end of training")
    parser.add_argument("--visualize_gradients_iters",nargs="+",type=int,default=[],help="Iterations during training where color-gradient visualization should be saved.")
    # New argument um adaptive sh-degrees zu aktivieren basierend auf color gradients
    parser.add_argument("--adaptive_sh", action="store_true", default=False, help="Use adaptive SH degrees based on color gradients")
    # New testing iterations for PSNR/SSIM:
    parser.add_argument("--psnr_ssim_iterations", nargs="+", type=int, default=[], help="Iterations during training where PSNR/SSIM should be computed.")
    # New testing iterations for LPIPS:
    parser.add_argument("--lpips_iterations", nargs="+", type=int, default=[], help="Iterations during training where LPIPS should be computed.")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.use_gui:
        print(f"Starting GUI server on {args.ip}:{args.port}")
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_gui, args.sh_percentage, args.color_grad_stats, args.need_color_grads, args.visualize_degrees, args.visualize_gradients, args.visualize_gradients_iters, args.adaptive_sh, args.psnr_ssim_iterations, args.lpips_iterations)

    # All done
    print("\nTraining complete.")
