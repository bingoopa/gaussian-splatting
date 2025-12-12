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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# Neue SH Speicherung
from sh_storage import SHStorage

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # New
        self.accum_color_grads_dc = torch.empty(0)
        self.accum_color_grads_rest = torch.empty(0)
        self.color_denom = 0
        
        """ try:
            import pandas as pd
            print("Pandas successfully imported for getColorGradStats.")
            #pandas_installed = True
            self.df = pd.DataFrame(columns=['iteration', 'grads_dc', 'grads_rest', 'grads_ratio', 'sh_degrees'])
        except ImportError:
            print("Pandas not installed, getColorGradStats will not work.")
            #pandas_installed = False TODO: remove, if color_grads are no more necessary
            self.df = None """
        

        # New
        self.sh_degrees = torch.empty(0, dtype=torch.int64, device="cuda")


        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # New
            self.accum_color_grads_dc,
            self.accum_color_grads_rest,
            self.color_denom,
            self.df,
            self.sh_degrees
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale, 
        self.accum_color_grads_dc,
        self.accum_color_grads_rest,
        self.color_denom,
        self.df,
        self.sh_degrees) = model_args # New: sh_degrees and color grad stats
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # New
    @property
    def get_sh_degrees(self):
        return self.sh_degrees
    
    @property
    def get_accumulated_color_grads_dc(self):
        return self.accum_color_grads_dc
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # New
        self.sh_degrees = torch.zeros((fused_point_cloud.shape[0],), dtype = torch.int64, device="cuda") # Initial SH degree 0, überprüfen warum long
        #BENNET: torch.int32 zu torch.int64 geändert, hast du auch schon betrachtet, oder? 

        # Neue SH Speicherung
        self.sh_storage = SHStorage(
            num_gaussians=fused_point_cloud.shape[0]
        )
        self.sh_storage.initialize_sh_from_color(fused_color)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # New
        #sh_deg = self.sh_degrees.detach().cpu().numpy().astype(np.int32) if hasattr(self, 'sh_degrees') else np.full((xyz.shape[0],), self.max_sh_degree, dtype=np.int32)
        #BENNET: Hier änderungen vorgenommen (Copilot: Convert to i4 for PLY, keep internal dtype int64)
        sh_deg = (self.sh_degrees.detach().cpu().numpy().astype(np.int32)
                  if hasattr(self, 'sh_degrees') and self.sh_degrees.numel() == xyz.shape[0]
                  else np.full((xyz.shape[0],), self.max_sh_degree, dtype=np.int32))
        sh_deg = sh_deg.reshape(-1, 1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # New
        dtype_full.append(('sh_degrees', 'i4'))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, sh_deg), axis=1) # New: sh_deg
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # New
        try:
            sh_deg = np.asarray(plydata.elements[0]["sh_degrees"]).astype(np.int64)
            self.sh_degrees = torch.tensor(sh_deg, dtype=torch.int64, device="cuda")
            # set the active global degree to the maximum of the existing sh-degrees (keeps compatibility, but not necessary)
            self.active_sh_degree = int(self.sh_degrees.max())
        except Exception:
            # no per-vertex sh_degree stored: enable all coefficients
            P = xyz.shape[0]
            self.sh_degrees = torch.full((P,), self.max_sh_degree, dtype=torch.int64, device="cuda")
            self.active_sh_degree = self.max_sh_degree

        #self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # New
        self.sh_degrees = self.sh_degrees[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sh_degrees=None): #BENNET: New argument: new_sh_degrees=None
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # New: set sh-degrees for new points (set 0 if not provided)
        #if new_sh_degrees is None:
        #    new_sh_degrees = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device="cuda")
        #self.sh_degrees = torch.cat((self.sh_degrees, new_sh_degrees), dim=0)
        #BENNET: geändert, um dtype und device zu gewährleisten
        if new_sh_degrees is None:
            new_sh_degrees = torch.zeros((new_xyz.shape[0],), dtype=self.sh_degrees.dtype, device=self.sh_degrees.device)
        else:
            # ensure same dtype and device
            new_sh_degrees = new_sh_degrees.to(dtype=self.sh_degrees.dtype, device=self.sh_degrees.device)
        self.sh_degrees = torch.cat((self.sh_degrees, new_sh_degrees), dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2): #TODO: hier sollen sh-degrees mitkopiert werden beim splitting
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # New
        new_sh_degrees = self.sh_degrees[selected_pts_mask].repeat(N)

        # Neue SH Speicherung
        self.sh_storage.duplicate_sh_of_gaussians(selected_pts_mask)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_sh_degrees) # New: new_sh_degrees

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent): #TODO: hier sollen sh-degrees mitkopiert werden beim klonen
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # New
        new_sh_degrees = self.sh_degrees[selected_pts_mask]

        # Neue SH Speicherung
        self.sh_storage.duplicate_sh_of_gaussians(selected_pts_mask)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sh_degrees) # New: new_sh_degrees

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # New
    def cumulate_color_gradients(self): 
        if(self.color_denom == 0):
            print("Initializing color gradient accumulation tensors...")
            self.accum_color_grads_dc = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            self.accum_color_grads_rest = torch.zeros((self.get_xyz.shape[0],), device = "cuda")
        max_num_coeffs = (self.max_sh_degree + 1) **2 - 1
        num_coeffs = ((self.sh_degrees +1)**2 - 1).view(-1, 1)
        idxs = torch.arange(0, max_num_coeffs, device="cuda").view(1, -1)
        mask = (idxs < num_coeffs).float().view(-1, max_num_coeffs)
        #assert self._features_rest.shape == mask.shape, f"Wrong shape of mask: feature_rest gradient shape: {self._features_rest.shape}, mask shape: {mask.shape}"
    #print("shape of self._features_rest.grad:", self._features_rest.grad.shape)
    #print("shape of self._features_dc.grad:", self._features_dc.grad.shape)
    #print("shape of accum_color_grads_dc:", self.accum_color_grads_dc.shape)
    #print("shape of accum_color_grads_rest:", self.accum_color_grads_rest.shape)
    #print("shape of mask:", mask.shape)
        self.accum_color_grads_dc += torch.norm(self._features_dc.grad, dim = -1).squeeze(-1)
        self.accum_color_grads_rest += torch.norm(self._features_rest.grad * mask.unsqueeze(-1), dim = (1, 2))
        #print("shape of self.accum_color_grads_rest after update:", self.accum_color_grads_rest.shape)
        self.color_denom += 1 
        
    # New
    def color_gradients_postfix(self):
        self.accum_color_grads_dc = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.accum_color_grads_rest = torch.zeros((self.get_xyz.shape[0], 1), device = "cuda")
        self.color_denom = 0

    # New methods for SH degree management
    def get_max_sh_degree_in_model(self):
        return int(self.sh_degrees.max())
    
    def set_sh_degrees_by_indices(self, indices, degree):
        assert degree <= self.max_sh_degree and degree >=0, f"Degree {degree} is out of bounds [0,{self.max_sh_degree}]"
        assert indices.max() < self.sh_degrees.shape[0], f"Index {indices.max()} is out of bounds [0,{self.sh_degrees.shape[0]-1}]"
        self.sh_degrees[indices] = degree

    def set_random_sh_degrees(self):
        n_points = self.sh_degrees.shape[0]
        random_degrees = torch.randint(0, self.max_sh_degree + 1, (n_points,), device="cuda", dtype=torch.int64)
        self.sh_degrees = random_degrees

    def get_sh_degree_distribution(self):
        unique, counts = torch.unique(self.sh_degrees, return_counts=True)
        for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            print(f"SH degree {u}: {c} Gaussians")

    def get_sh_degrees_by_indices(self, indices):
        return self.sh_degrees[indices]
    
    def randomly_increase_sh_degrees_by_one(self, fraction):
        n_points = self.sh_degrees.shape[0]
        n_increase = int(n_points * fraction)
        all_indices = torch.arange(n_points, device="cuda")
        selected_indices = all_indices[torch.randperm(n_points)[:n_increase]]
        valid = selected_indices[self.sh_degrees[selected_indices] < self.max_sh_degree]
        self.sh_degrees[valid] += 1

        updated_percentage = valid.numel() / n_increase * 100.0
        print(f"Randomly increased SH degree for {valid.numel()} Gaussians ({updated_percentage:.2f}%), given was {fraction*100:.2f}%")

    
    # New
    def getColorGradStats(self, iteration):

        # Ensure that cumulate_color_gradients is the average over 50 iterations
        assert self.color_denom == 50, f"color_denom is {self.color_denom}, expected 50"

        # Compute ratios and average gradients and store in dataframe
        ratios = (self.accum_color_grads_rest / (self.accum_color_grads_dc + 1e-15)).detach().cpu().numpy()
        color_grads_dc = (self.accum_color_grads_dc / (self.color_denom + 1e-15)).detach().cpu().numpy()
        color_grads_rest = (self.accum_color_grads_rest / (self.color_denom + 1e-15)).detach().cpu().numpy()
        sh_degrees_array = self.sh_degrees.detach().cpu().numpy()

        if self.df is not None:
            P = len(color_grads_dc)
            assert len(color_grads_rest) == P and len(ratios) == P, "Inconsistent lengths of gradient arrays"
            
            new_df = pd.DataFrame({
                'iteration': np.full(P, iteration, dtype=int),
                'grads_dc': color_grads_dc,
                'grads_rest': color_grads_rest,
                'grads_ratio': ratios,
                'sh_degrees': sh_degrees_array
            })

            # an bestehenden df anhängen
            self.df = pd.concat([self.df, new_df], ignore_index=True)
    
    # New
    def saveColorGradStatsToCSV(self, path):
        if self.df is not None:
            mkdir_p(os.path.dirname(path))
            self.df.to_csv(path, index=False)
            print(f"Saved color gradient statistics to {path}")
        else:
            print("Pandas not installed, cannot save color gradient statistics.")

    def prepare_color_grads(self):
        ratios = (self.accum_color_grads_rest / (self.accum_color_grads_dc + 1e-15))
        color_grads_dc = (self.accum_color_grads_dc / (self.color_denom + 1e-15))
        color_grads_rest = (self.accum_color_grads_rest / (self.color_denom + 1e-15))
        return color_grads_dc, color_grads_rest, ratios

    def get_colorized_copy(self, get_color_fun):
        clone = GaussianModel(self.max_sh_degree)
        with torch.no_grad():
            # --- Basisdaten kopieren ---
            clone._xyz = self._xyz.clone()
            clone._scaling = self._scaling.clone()
            clone._rotation = self._rotation.clone()
            clone._opacity = self._opacity.clone()
            clone.sh_degrees = self.sh_degrees.clone()
            # --- Farben als RGB holen ---
            rgb_np = get_color_fun(self)     # (P,3) oder (P,4)
            rgb_np = rgb_np[:, :3]
            rgb = torch.tensor(rgb_np, device="cuda", dtype=torch.float32) / 255.0
            P = self.get_xyz.shape[0]
            # 1) DC-Features korrekt anlegen → (P, 1, 3)
            clone._features_dc = torch.zeros((P, 1, 3), device="cuda")
            clone._features_dc[:, 0, :] = rgb     # DC ist RGB
            # 2) SH-Rest Features: (P, num_rest_coeffs, 3)
            max_coeffs = (self.max_sh_degree + 1) ** 2 - 1  # z. B. 15
            clone._features_rest = torch.zeros((P, max_coeffs, 3), device="cuda")
            # Wichtig: NICHT transponieren!
            # Der Renderer erwartet: (P, COEFF, CHANNEL)
        return clone

    def color_gradients_postfix(self):
        self.accum_color_grads_dc = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        self.accum_color_grads_rest = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        self.color_denom = 0
      
    # Bennet New: visualize sh degrees    
    def visualize_sh_degrees(self):
        degree_colors = {
            0: torch.tensor([0.5, 0.5, 0.5], device="cuda"),  # grau
            1: torch.tensor([0.0, 1.0, 0.0], device="cuda"),  # grün
            2: torch.tensor([0.0, 0.0, 1.0], device="cuda"),  # blau
            3: torch.tensor([1.0, 0.0, 0.0], device="cuda"),  # rot      
            }
        max_defined = max(degree_colors.keys())
        sh_deg = self.sh_degrees
        with torch.no_grad():
            for d in torch.unique(sh_deg):
                d_int = int(d.item())
                color = degree_colors[d_int] if d_int in degree_colors else degree_colors[d_int % (max_defined + 1)]
                mask = (sh_deg == d)
                self._features_dc.data[mask, 0, :] = color
                if self._features_rest is not None and self._features_rest.numel() > 0:
                    self._features_rest.data[mask, :, :] = 0.0
    
    def increase_sh_degree_based_on_color_grads(self, ratio=0.05):
        quantile_value = torch.quantile(self.accum_color_grads_dc, 1-ratio)
        to_increase = (self.accum_color_grads_dc > quantile_value).squeeze()
        valid = to_increase & (self.sh_degrees < self.max_sh_degree)
        self.sh_degrees[valid] += 1
        updated_percentage = valid.sum().item() / to_increase.sum().item() * 100.0 if to_increase.sum().item() > 0 else 0.0
        print(f"Increased SH degree for {valid.sum().item()} Gaussians ({updated_percentage:.2f}%), given ratio threshold was {ratio}")

    