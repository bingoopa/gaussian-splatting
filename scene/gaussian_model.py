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
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
try:
    import pandas as pd
except ImportError:
    pd = None
# Neue SH Speicherung
from .sh_storage_new import SHStorage

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
        print("sh_degree in GaussianModel:", sh_degree)
        self._xyz = torch.empty(0)
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
        # last_promotion_iter: per-gaussian iteration index of last SH-degree promotion
        # -1 means never promoted
        self.last_promotion_iter = torch.empty(0, dtype=torch.int64)
        # New
        

        if pd is not None:
            self.df = pd.DataFrame(columns=['iteration', 'grads_dc', 'grads_rest', 'grads_ratio', 'sh_degrees'])
        else:
            self.df = None

        # SH storage
        self.sh_storage = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sh_degrees = torch.empty(0, dtype=torch.int64, device=device)

        # per-gaussian seen counts for color grad normalization
        #self.color_seen_counts = torch.empty(0, dtype=torch.float32, device=device)


        self.setup_functions()

    def capture(self):
        sh_state = self.sh_storage.serialize() if self.sh_storage is not None else None
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
            "accum_color_grads_dc": self.accum_color_grads_dc,
            "accum_color_grads_rest": self.accum_color_grads_rest,
            "color_denom": self.color_denom,
            "df": self.df,
            "sh_degrees": self.sh_degrees,
            "last_promotion_iter": self.last_promotion_iter,
            "sh_state": sh_state,
        }
    
    def restore(self, model_args, training_args):
        if isinstance(model_args, dict):
            state = model_args
        else:
            state = self._convert_legacy_state(model_args)

        self.active_sh_degree = state["active_sh_degree"]
        self._xyz = state["xyz"]
        self._scaling = state["scaling"]
        self._rotation = state["rotation"]
        self._opacity = state["opacity"]
        self.max_radii2D = state["max_radii2D"]
        self.spatial_lr_scale = state["spatial_lr_scale"]
        self.accum_color_grads_dc = state["accum_color_grads_dc"]
        self.accum_color_grads_rest = state["accum_color_grads_rest"]
        self.color_denom = state["color_denom"]
        self.df = state.get("df", None)
        sh_state = state.get("sh_state", None)
        if sh_state is not None:
            self.sh_storage = SHStorage.from_serialized(sh_state, device=self._xyz.device)
        else:
            dense = state["dense_sh"]
            sh_degrees = state["sh_degrees"]
            self.sh_storage = self._build_storage_from_dense(dense, sh_degrees)
        self._sync_sh_degrees_from_storage()

        # restore last_promotion_iter if present, otherwise initialize to -1 per gaussian
        device = self._xyz.device if hasattr(self, "_xyz") and isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0 else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        last_iter = state.get("last_promotion_iter", None)
        if last_iter is not None:
            # ensure tensor on correct device and dtype
            self.last_promotion_iter = last_iter.to(device=device, dtype=torch.int64)
        else:
            # init -1 matching number of gaussians
            n = int(self.sh_degrees.shape[0]) if self.sh_degrees is not None and self.sh_degrees.numel() > 0 else 0
            if n > 0:
                self.last_promotion_iter = torch.full((n,), -1, device=device, dtype=torch.int64)
            else:
                self.last_promotion_iter = torch.empty((0,), dtype=torch.int64, device=device)

        self.training_setup(training_args)
        self.xyz_gradient_accum = state["xyz_gradient_accum"]
        self.denom = state["denom"]
        opt_state = state.get("optimizer_state", None)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

    def _convert_legacy_state(self, legacy_tuple):
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            spatial_lr_scale,
            accum_color_grads_dc,
            accum_color_grads_rest,
            color_denom,
            df,
            sh_degrees,
        ) = legacy_tuple
        dense = torch.cat(
            (
                features_dc.detach().transpose(1, 2),
                features_rest.detach().transpose(1, 2),
            ),
            dim=2,
        )
        return {
            "active_sh_degree": active_sh_degree,
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "max_radii2D": max_radii2D,
            "xyz_gradient_accum": xyz_gradient_accum,
            "denom": denom,
            "optimizer_state": opt_dict,
            "spatial_lr_scale": spatial_lr_scale,
            "accum_color_grads_dc": accum_color_grads_dc,
            "accum_color_grads_rest": accum_color_grads_rest,
            "color_denom": color_denom,
            "df": df,
            "sh_degrees": sh_degrees,
            "dense_sh": dense,
            "sh_state": None,
        }

    def _build_storage_from_dense(self, dense_sh, sh_degrees):
        # dense_sh is expected to be [N, 3, K] (N gaussians, 3 color channels, K coeffs)
        # but older/alternate layouts might be [N, K, 3]. Detect and handle both.
        if dense_sh.dim() != 3:
            raise RuntimeError(f"dense_sh must be a 3D tensor, got shape {tuple(dense_sh.shape)}")

        # If layout is [N, K, 3], permute to [N, 3, K]
        if dense_sh.shape[1] != 3 and dense_sh.shape[2] == 3:
            dense_sh = dense_sh.permute(0, 2, 1).contiguous()

        device = dense_sh.device
        num_gauss = dense_sh.shape[0]
        storage = SHStorage(
            num_gaussians=num_gauss,
            init_deg=0,
            max_degree=self.max_sh_degree,
            device=device,
        )
        degrees_i32 = sh_degrees.to(device=device, dtype=torch.int32)
        counts = (degrees_i32 + 1) ** 2
        offsets = torch.cumsum(counts, dim=0) - counts
        total = int(counts.sum().item())
        flat = torch.zeros((total, 3), device=device, dtype=dense_sh.dtype)
        cursor = 0
        for i in range(num_gauss):
            count = int(counts[i].item())
            if count > 0:
                flat[cursor : cursor + count] = dense_sh[i, :, :count].transpose(0, 1)
            cursor += count
        storage.sh_coeffs_flat = nn.Parameter(flat)
        storage.gauss_offsets = offsets.to(torch.int32)
        storage.num_coeffs_per_gauss = counts
        storage.sh_degrees = degrees_i32
        storage.num_gauss = num_gauss
        storage.num_gaussians = num_gauss
        return storage

    def _sync_sh_degrees_from_storage(self):
        if self.sh_storage is None:
            device = self._xyz.device if self._xyz.numel() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sh_degrees = torch.empty(0, dtype=torch.int64, device=device)
            return
        device = self.sh_storage.sh_coeffs_flat.device
        self.sh_degrees = self.sh_storage.sh_degrees.to(device=device, dtype=torch.int64)

    def _validate_gaussian_consistency(self, context: str = ""):
        """
        Validates that all gaussian counts are consistent across different storage locations.
        Checks:
        - Number of xyz coordinates matches sh_degrees
        - Number of sh_degrees matches sh_storage.num_gaussians
        - Total sh_coeffs_flat size matches expected size based on sh_degrees
        - opacity, scaling, rotation counts match xyz
        """
        xyz_count = self._xyz.shape[0] if self._xyz.numel() > 0 else 0
        opacity_count = self._opacity.shape[0] if self._opacity.numel() > 0 else 0
        scaling_count = self._scaling.shape[0] if self._scaling.numel() > 0 else 0
        rotation_count = self._rotation.shape[0] if self._rotation.numel() > 0 else 0
        sh_degrees_count = self.sh_degrees.shape[0] if self.sh_degrees.numel() > 0 else 0
        
        errors = []
        
        # Check that all gaussian-level parameters have matching counts
        if xyz_count != opacity_count:
            errors.append(f"xyz({xyz_count}) != opacity({opacity_count})")
        if xyz_count != scaling_count:
            errors.append(f"xyz({xyz_count}) != scaling({scaling_count})")
        if xyz_count != rotation_count:
            errors.append(f"xyz({xyz_count}) != rotation({rotation_count})")
        if xyz_count != sh_degrees_count:
            errors.append(f"xyz({xyz_count}) != sh_degrees({sh_degrees_count})")
        
        if self.sh_storage is not None:
            storage_gauss_count = self.sh_storage.num_gaussians
            if xyz_count != storage_gauss_count:
                errors.append(f"xyz({xyz_count}) != sh_storage.num_gaussians({storage_gauss_count})")
            
            # Check sh_coeffs_flat size
            expected_total_coeffs = int(((self.sh_storage.sh_degrees + 1) ** 2).sum().item())
            actual_total_coeffs = self.sh_storage.sh_coeffs_flat.shape[0]
            if expected_total_coeffs != actual_total_coeffs:
                errors.append(f"expected_coeffs({expected_total_coeffs}) != actual_coeffs({actual_total_coeffs})")
            
            # Check gauss_offsets consistency
            offsets = self.sh_storage.gauss_offsets
            counts = self.sh_storage.num_coeffs_per_gauss
            if offsets.shape[0] != sh_degrees_count:
                errors.append(f"gauss_offsets({offsets.shape[0]}) != sh_degrees({sh_degrees_count})")
            if counts.shape[0] != sh_degrees_count:
                errors.append(f"num_coeffs_per_gauss({counts.shape[0]}) != sh_degrees({sh_degrees_count})")
        
        if errors:
            error_msg = f"Gaussian consistency check FAILED {context}:\n  " + "\n  ".join(errors)
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        else:
            print(f"✓ Gaussian consistency check OK {context}: {xyz_count} gaussians")

    def _reinit_sh_optimizer_state(self):
        """
        Reinitialize the SH coefficient optimizer state after degree increase.
        
        Called when SH storage is reorganized (e.g., degree increase via _repack_all).
        Since SHStorage._repack_all() reorganizes coefficients internally,
        we cannot reliably map old optimizer moments to new positions.
        Instead, we safely reinitialize with fresh zero moments, preserving the step count.
        """
        if self.optimizer is None:
            return
        
        for group in self.optimizer.param_groups:
            if group["name"] != "sh_coeffs":
                continue
            
            old_param = group["params"][0]
            new_param = self.sh_storage.sh_coeffs_flat
            
            # Ensure new parameter requires gradients
            new_param.requires_grad_(True)
            
            # Update parameter reference in optimizer group
            group["params"][0] = new_param
            
            # Get old optimizer state to preserve the step count
            old_state = self.optimizer.state.get(old_param, None)
            
            # Remove old optimizer state
            if old_param in self.optimizer.state:
                del self.optimizer.state[old_param]
            
            # Initialize fresh optimizer state for new parameter
            # Preserve step count from old state, reset moments
            new_state = {
                "exp_avg": torch.zeros_like(new_param),
                "exp_avg_sq": torch.zeros_like(new_param),
            }
            
            # Preserve step count if it existed
            # Note: PyTorch Adam stores step as a singleton tensor (0-dimensional)
            if old_state is not None and "step" in old_state:
                old_step = old_state["step"]
                # Ensure step is a tensor on the correct device
                if isinstance(old_step, torch.Tensor):
                    new_state["step"] = old_step.to(device=new_param.device)
                else:
                    # If it's an int or float, convert to a singleton tensor
                    new_state["step"] = torch.tensor(old_step, device=new_param.device, dtype=torch.long)
            else:
                # Initialize step to 0 if not present (as a singleton tensor)
                new_state["step"] = torch.tensor(0, device=new_param.device, dtype=torch.long)
            
            self.optimizer.state[new_param] = new_state
            break

    def _extend_sh_optimizer_state(self, clone_indices):
        """
        Extends the SH optimizer state when cloning/splitting gaussians.
        For each cloned gaussian, replicates the moments (exp_avg, exp_avg_sq) 
        of the corresponding old gaussian for the newly added coefficients.
        
        clone_indices: tensor of shape [num_new_gaussians] containing the indices
                       of the old gaussians that were cloned/split
        """
        if self.optimizer is None:
            return
        
        for group in self.optimizer.param_groups:
            if group["name"] != "sh_coeffs":
                continue
            
            param = group["params"][0]
            old_state = self.optimizer.state.get(param, None)
            if old_state is None:
                return
            
            # Get the offset mapping for each newly added gaussian
            # We need to know which coefficients in sh_coeffs_flat correspond to each gaussian
            old_gauss_offsets = self.sh_storage.gauss_offsets[:-clone_indices.shape[0]]  # offsets before extension
            old_gauss_counts = self.sh_storage.num_coeffs_per_gauss[:-clone_indices.shape[0]]
            new_gauss_offsets = self.sh_storage.gauss_offsets[-clone_indices.shape[0]:]  # offsets of newly added
            new_gauss_counts = self.sh_storage.num_coeffs_per_gauss[-clone_indices.shape[0]:]
            
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in old_state:
                    continue
                old_moments = old_state[key]
                
                # For each new gaussian, replicate the moments of its source gaussian
                for new_gauss_idx, source_gauss_idx in enumerate(clone_indices):
                    source_idx = int(source_gauss_idx.item())
                    source_offset = int(old_gauss_offsets[source_idx].item())
                    source_count = int(old_gauss_counts[source_idx].item())
                    
                    new_offset = int(new_gauss_offsets[new_gauss_idx].item())
                    new_count = int(new_gauss_counts[new_gauss_idx].item())
                    
                    # Copy moments from source coefficients to new coefficients
                    # Handle case where source and new have different numbers of coefficients
                    copy_count = min(source_count, new_count)
                    if copy_count > 0:
                        old_moments[new_offset:new_offset + copy_count] = old_moments[source_offset:source_offset + copy_count]
                    
                    # Zero out any additional coefficients if new has more than source
                    if new_count > source_count:
                        old_moments[new_offset + source_count:new_offset + new_count] = 0.0

    def _extend_sh_optimizer_state_for_densification(self, clone_indices):
        """
        Helper method to extend SH optimizer state after densification (clone/split).
        This method extracts the source gaussian indices from clone_indices and replicates
        their optimizer moments to the newly created gaussians.
        
        Args:
            clone_indices: Tensor of indices of source gaussians that were cloned/split.
                          The storage already contains the new cloned gaussians appended at the end.
        """
        if self.optimizer is None or clone_indices.numel() == 0:
            return
        
        for group in self.optimizer.param_groups:
            if group["name"] != "sh_coeffs":
                continue
            
            param = group["params"][0]
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is None:
                # If no state exists, initialize with zeros
                stored_state = {
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                }
                self.optimizer.state[param] = stored_state
                return
            
            # Get total number of new gaussians added (stored in sh_storage)
            total_old_gauss_before = self.sh_storage.num_gaussians - clone_indices.shape[0]
            
            # For each source gaussian in clone_indices, replicate its moments to the corresponding new gaussian
            for new_gauss_relative_idx, source_gauss_idx in enumerate(clone_indices):
                source_idx = int(source_gauss_idx.item())
                
                # Get old and new offsets and counts
                source_offset = int(self.sh_storage.gauss_offsets[source_idx].item())
                source_count = int(self.sh_storage.num_coeffs_per_gauss[source_idx].item())
                
                new_gauss_idx = total_old_gauss_before + new_gauss_relative_idx
                new_offset = int(self.sh_storage.gauss_offsets[new_gauss_idx].item())
                new_count = int(self.sh_storage.num_coeffs_per_gauss[new_gauss_idx].item())
                
                # Replicate moments for exp_avg and exp_avg_sq
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in stored_state:
                        moments = stored_state[key]
                        # Copy existing moments from source
                        copy_count = min(source_count, new_count)
                        if copy_count > 0 and source_offset + copy_count <= moments.shape[0] and new_offset + copy_count <= moments.shape[0]:
                            moments[new_offset:new_offset + copy_count] = moments[source_offset:source_offset + copy_count].clone()
                        
                        # Zero out extra coefficients if new has more than source
                        if new_count > source_count:
                            moments[new_offset + source_count:new_offset + new_count] = 0.0

    def _apply_new_sh_degrees(self, new_degrees):
        if self.sh_storage is None:
            return
        new_degrees = new_degrees.to(device=self.sh_storage.sh_coeffs_flat.device, dtype=torch.int32)
        self.sh_storage._repack_all(new_degrees)
        self._sync_sh_degrees_from_storage()
        # Degree-Increase NICHT mit cat_tensors_to_optimizer, da es eine Neuorganisation ist
        # (nicht nur Erweiterung wie bei Densification).
        # Die alte Methode cat_tensors_to_optimizer prüft: add_rows < 0 → Error
        # Aber bei Degree-Increase können sich die Anzahl der Reihen verringern!
        # Beispiel: 5 Gaussians mit Degree 3 → [1,1,1,1,1] coeffs → 5 rows
        #          dann Degree-Increase auf [1,1,3,3,3] → [1,1,16,16,16] coeffs → 50 rows
        # Die alte Moments-Struktur kann NICHT einfach mit torch.cat kombiniert werden,
        # da jeder Koeffizient eine neue Position hat.
        # Lösung: Sicher die Step-Count des Optimizers, reinitialize alle Moments.
        self._reinit_sh_optimizer_state()

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
        if not hasattr(self, "sh_storage"):
            return None
        dense = self.sh_storage.build_dense_sh(self.max_sh_degree)  # [N, 3, K]
        return dense.permute(0, 2, 1).contiguous()
    
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

        # Neue SH Speicherung
        self.sh_storage = SHStorage(
            num_gaussians=fused_point_cloud.shape[0],
            device=fused_point_cloud.device,
            max_degree=self.max_sh_degree,
            init_deg=0
        )
        self.sh_storage.initialize_sh_from_color(fused_color)
        self._sync_sh_degrees_from_storage()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # initialize last_promotion_iter to -1 (never promoted)
        n = self.get_xyz.shape[0]
        device = fused_point_cloud.device
        self.last_promotion_iter = torch.full((n,), -1, device=device, dtype=torch.int64)

    def training_setup(self, training_args):
        if self.sh_storage is None:
            raise RuntimeError("SHStorage must be initialized before training_setup")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.sh_storage.sh_coeffs_flat], 'lr': training_args.feature_lr, "name": "sh_coeffs"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        # Store feature_lr for gradient scaling (features_rest get 1/20 of DC learning rate)
        self.feature_lr = training_args.feature_lr
        self.feature_lr_rest_scale = 1.0 / 20.0

        # --- Ensure optimizer state is initialized for all parameter groups (prevents KeyError: 'step') ---
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
        #self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Consistency check after training setup
        print("[training_setup] Checking initial consistency...")
        #self._validate_gaussian_consistency("[after training_setup]")

    def scale_sh_gradients_before_optimizer_step(self):
        """
        Scale the gradients of features_rest (all SH coefficients except DC) by 1/20.
        This is equivalent to having two separate learning rates:
        - features_dc: lr = feature_lr
        - features_rest: lr = feature_lr / 20.0
        
        Must be called right before optimizer.step() to apply the different learning rates.
        """
        if self.sh_storage is None or self.sh_storage.sh_coeffs_flat.grad is None:
            return
        
        grad = self.sh_storage.sh_coeffs_flat.grad
        if grad is None:
            return
        
        device = grad.device
        counts = self.sh_storage.num_coeffs_per_gauss.to(device=device, dtype=torch.long)
        total = int(counts.sum().item())
        if total == 0:
            return
        
        # Create mask: DC coefficients (first coeff per gaussian) get scale 1.0, rest get scale 1/20
        gauss_ids = torch.repeat_interleave(torch.arange(counts.shape[0], device=device, dtype=torch.long), counts)
        local_idx = torch.arange(total, device=device, dtype=torch.long) - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
        
        dc_mask = local_idx == 0
        rest_mask = ~dc_mask
        
        # Scale rest gradients by 1/20
        grad[rest_mask] *= self.feature_lr_rest_scale

    def optimizer_step_with_scaled_sh_lr(self):
        """
        Perform optimizer step with scaled learning rates for SH coefficients.
        - DC coefficients (first coeff per gaussian): use full feature_lr
        - Rest coefficients: use feature_lr / 20
        
        This properly integrates the different LRs into Adam's bias correction.
        Handles both regular parameters and SH coefficients in sh_storage.
        """
        if self.optimizer is None:
            return
        
        # First, handle custom step for SH coefficients with different DC/rest learning rates
        for group in self.optimizer.param_groups:
            if group["name"] != "sh_coeffs":
                continue
            
            # Found the SH coeffs group - do custom step for it
            param = group["params"][0]
            if param.grad is None:
                continue
            
            state = self.optimizer.state.get(param, None)
            if state is None:
                continue
            
            # Get step count
            step = state.get("step", torch.tensor(0, device=param.device, dtype=torch.long))
            if isinstance(step, torch.Tensor):
                step_int = int(step.item())
            else:
                step_int = int(step)
            step_int += 1
            
            # Create mask for DC vs rest coefficients
            device = param.device
            counts = self.sh_storage.num_coeffs_per_gauss.to(device=device, dtype=torch.long)
            total = int(counts.sum().item())
            
            if total == 0:
                continue
            
            gauss_ids = torch.repeat_interleave(
                torch.arange(counts.shape[0], device=device, dtype=torch.long), counts
            )
            local_idx = torch.arange(total, device=device, dtype=torch.long) - torch.repeat_interleave(
                torch.cumsum(counts, dim=0) - counts, counts
            )
            
            dc_mask = local_idx == 0
            rest_mask = ~dc_mask
            
            # Get moments and gradient
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            grad = param.grad
            
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            
            # Bias correction factors
            bias_correction1 = 1 - beta1 ** step_int
            bias_correction2 = 1 - beta2 ** step_int
            
            # Update moments (same for both DC and rest)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Compute updates with different learning rates
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size_dc = group["lr"] / bias_correction1
            step_size_rest = (group["lr"] / 20.0) / bias_correction1
            
            # Apply update
            param.data[dc_mask].add_(exp_avg[dc_mask] / denom[dc_mask], alpha=-step_size_dc)
            param.data[rest_mask].add_(exp_avg[rest_mask] / denom[rest_mask], alpha=-step_size_rest)
            
            # Update step counter
            state["step"] = torch.tensor(step_int, device=device, dtype=torch.long)
            break
        
        # Now do a standard optimizer step for all other parameters
        # We need to temporarily exclude the sh_coeffs parameter from the step
        sh_param_to_exclude = None
        sh_group_idx = None
        for idx, group in enumerate(self.optimizer.param_groups):
            if group["name"] == "sh_coeffs":
                sh_param_to_exclude = group["params"][0]
                sh_group_idx = idx
                break
        
        # Temporarily remove sh_coeffs from optimizer groups
        saved_group = None
        if sh_group_idx is not None:
            saved_group = self.optimizer.param_groups.pop(sh_group_idx)
        
        try:
            # Do optimizer step for remaining parameters
            self.optimizer.step()
        finally:
            # Restore sh_coeffs group
            if saved_group is not None:
                self.optimizer.param_groups.insert(sh_group_idx, saved_group)
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        max_coeffs = (self.max_sh_degree + 1) ** 2
        for i in range(3):
            l.append(f'f_dc_{i}')
        for i in range(max(0, 3 * (max_coeffs - 1))):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    
    def save_ply(self, path, roundtrip_check: bool = False):
            # per vertex version
            # Save scene to a PLY using compact SH storage.
        

            # The PLY will contain two elements:
            # - 'vertex' with per-vertex attributes and integer metadata fields:
            # sh_degrees (i4), sh_offset (i4), sh_num_coeffs (i4)
            # - 'sh_coeffs' with all SH rows packed as (r,g,b) floats
            
            mkdir_p(os.path.dirname(path))

            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)

            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

            storage = self.sh_storage
            if storage is None:
                sh_coeffs_np = np.zeros((0, 3), dtype=np.float32)
                gauss_offsets_np = np.zeros((xyz.shape[0],), dtype=np.int32)
                num_coeffs_np = np.zeros((xyz.shape[0],), dtype=np.int32)
                sh_degrees_np = np.full((xyz.shape[0],), int(self.max_sh_degree), dtype=np.int32)
            else:
                sh_coeffs_np = storage.sh_coeffs_flat.detach().cpu().numpy().astype(np.float32)
                gauss_offsets_np = storage.gauss_offsets.detach().cpu().numpy().astype(np.int32)
                num_coeffs_np = storage.num_coeffs_per_gauss.detach().cpu().numpy().astype(np.int32)
                if hasattr(self, 'sh_degrees') and self.sh_degrees.numel() == xyz.shape[0]:
                    sh_degrees_np = self.sh_degrees.detach().cpu().numpy().astype(np.int32)
                else:
                    sh_degrees_np = np.full((xyz.shape[0],), int(self.max_sh_degree), dtype=np.int32)

            # Build vertex dtype and populate rows
            vertex_dtype = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('opacity', 'f4')
            ]
            for i in range(self._scaling.shape[1]):
                vertex_dtype.append((f'scale_{i}', 'f4'))
            for i in range(self._rotation.shape[1]):
                vertex_dtype.append((f'rot_{i}', 'f4'))
            vertex_dtype.extend([('sh_degrees', 'i4'), ('sh_offset', 'i4'), ('sh_num_coeffs', 'i4')])

            vertices = np.empty(xyz.shape[0], dtype=vertex_dtype)
            for i in range(xyz.shape[0]):
                row = []
                row.extend([float(v) for v in xyz[i]])
                row.extend([float(v) for v in normals[i]])
                row.append(float(opacities[i]))
                row.extend([float(v) for v in scale[i]])
                row.extend([float(v) for v in rotation[i]])
                row.append(int(sh_degrees_np[i]))
                row.append(int(gauss_offsets_np[i]))
                row.append(int(num_coeffs_np[i]))
                vertices[i] = tuple(row)

            # SH coeffs element
            sh_dtype = [('r', 'f4'), ('g', 'f4'), ('b', 'f4')]
            sh_elements = np.empty(sh_coeffs_np.shape[0], dtype=sh_dtype)
            if sh_coeffs_np.shape[0] > 0:
                sh_elements['r'] = sh_coeffs_np[:, 0]
                sh_elements['g'] = sh_coeffs_np[:, 1]
                sh_elements['b'] = sh_coeffs_np[:, 2]

            el_vert = PlyElement.describe(vertices, 'vertex')
            el_sh = PlyElement.describe(sh_elements, 'sh_coeffs')
            PlyData([el_vert, el_sh]).write(path)

            if roundtrip_check:
                try:
                    gm2 = GaussianModel(self.max_sh_degree)
                    gm2.load_ply(path)
                    orig_state = self.sh_storage.serialize() if self.sh_storage is not None else None
                    new_state = gm2.sh_storage.serialize() if gm2.sh_storage is not None else None
                    if (orig_state is None) != (new_state is None):
                        raise RuntimeError("PLY roundtrip failed: missing sh_storage after load")
                    if orig_state is not None:
                        orig_counts = orig_state["num_coeffs_per_gauss"].cpu().numpy()
                        new_counts = new_state["num_coeffs_per_gauss"].cpu().numpy()
                        if orig_counts.shape != new_counts.shape or not np.array_equal(orig_counts, new_counts):
                            raise RuntimeError("PLY roundtrip failed: coeff counts differ")
                        orig_degs = orig_state["sh_degrees"].cpu().numpy()
                        new_degs = new_state["sh_degrees"].cpu().numpy()
                        if orig_degs.shape != new_degs.shape or not np.array_equal(orig_degs, new_degs):
                            raise RuntimeError("PLY roundtrip failed: sh_degrees differ")
                        orig_coeffs = orig_state["sh_coeffs_flat"].cpu().numpy()
                        new_coeffs = new_state["sh_coeffs_flat"].cpu().numpy()
                        if orig_coeffs.shape != new_coeffs.shape or not np.allclose(orig_coeffs, new_coeffs, atol=1e-6):
                            raise RuntimeError("PLY roundtrip failed: sh_coeffs_flat differ (values or shape)")
                    print(f"PLY roundtrip check OK: {path}")
                except Exception as e:
                    print(f"PLY roundtrip check failed: {e}")
                    raise
                    
    

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    
    def load_ply(self, path):
        # per vertex version
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # Compact format: SH rows are stored separately in 'sh_coeffs' element
        # and per-vertex metadata ('sh_offset', 'sh_num_coeffs', optional 'sh_degrees')
        # We don't expect legacy f_dc_/f_rest_ fields in the compact format.

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

        # Choose runtime device (avoid hardcoded 'cuda')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        # Expect either the compact SH storage format (written by save_ply)
        # or the legacy per-vertex dense SH fields (f_dc_* and f_rest_*).
        vertex_props = [p.name for p in plydata.elements[0].properties]
        if ('sh_offset' in vertex_props and 'sh_num_coeffs' in vertex_props):
            # Read per-vertex metadata
            sh_offsets_np = np.asarray(plydata.elements[0]['sh_offset']).astype(np.int64)
            num_coeffs_np = np.asarray(plydata.elements[0]['sh_num_coeffs']).astype(np.int64)
            if 'sh_degrees' in vertex_props:
                sh_degrees_np = np.asarray(plydata.elements[0]['sh_degrees']).astype(np.int64)
            else:
                sh_degrees_np = np.array([int(math.isqrt(int(n)) - 1) for n in num_coeffs_np], dtype=np.int64)

            # Read packed SH coeffs element
            try:
                sh_el = next(e for e in plydata.elements if e.name == 'sh_coeffs')
            except StopIteration:
                raise RuntimeError("PLY missing 'sh_coeffs' element required for compact format.")

            sh_r = np.asarray(sh_el['r'])
            sh_g = np.asarray(sh_el['g'])
            sh_b = np.asarray(sh_el['b'])
            sh_coeffs_np = np.stack((sh_r, sh_g, sh_b), axis=1).astype(np.float32)

            # Build serialized state and load via SHStorage.from_serialized onto chosen device
            sh_state = {
                'sh_coeffs_flat': torch.tensor(sh_coeffs_np, dtype=torch.float32, device=device),
                'gauss_offsets': torch.tensor(sh_offsets_np, dtype=torch.int32, device=device),
                'num_coeffs_per_gauss': torch.tensor(num_coeffs_np, dtype=torch.int32, device=device),
                'sh_degrees': torch.tensor(sh_degrees_np, dtype=torch.int32, device=device),
                'max_degree': int(int(sh_degrees_np.max())) if sh_degrees_np.size > 0 else 0,
            }
            self.sh_storage = SHStorage.from_serialized(sh_state, device=device)
            self._sync_sh_degrees_from_storage()
            self.active_sh_degree = int(self.sh_degrees.max().item()) if self.sh_degrees.numel() > 0 else 0
        else:
            # Try legacy per-vertex dense SH attributes: f_dc_0..2 and f_rest_* (flattened r,g,b for coeffs 1..K-1)
            f_dc_names = [n for n in vertex_props if n.startswith('f_dc_')]
            f_rest_names = [n for n in vertex_props if n.startswith('f_rest_')]
            if len(f_dc_names) == 3 and len(f_rest_names) >= 0:
                # sort names by index
                f_dc_names = sorted(f_dc_names, key=lambda x: int(x.split('_')[-1]))
                f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
                # read arrays
                dc_r = np.asarray(plydata.elements[0][f_dc_names[0]])
                dc_g = np.asarray(plydata.elements[0][f_dc_names[1]])
                dc_b = np.asarray(plydata.elements[0][f_dc_names[2]])
                N = dc_r.shape[0]
                rest_count = len(f_rest_names)
                if rest_count % 3 != 0:
                    raise RuntimeError(f"Unexpected number of f_rest_ fields: {rest_count}, not divisible by 3")
                K = 1 + (rest_count // 3)
                # allocate dense SH [N, 3, K]
                dense = np.zeros((N, 3, K), dtype=np.float32)
                dense[:, 0, 0] = dc_r
                dense[:, 1, 0] = dc_g
                dense[:, 2, 0] = dc_b
                # fill remaining coeffs
                if rest_count > 0:
                    # Original PLY export flattens channel-major: [r_k0, r_k1, ..., g_k0, g_k1, ..., b_k0, b_k1, ...]
                    # Read the flat rest values and reshape to (N, 3, K-1) so that axis 1=channel, axis 2=coeff
                    rest_vals = np.vstack([np.asarray(plydata.elements[0][name]) for name in f_rest_names]).T  # shape (N, rest_count)
                    try:
                        rest_reshaped = rest_vals.reshape((N, 3, K - 1))
                    except Exception:
                        # Fallback to previous interleaved interpretation if reshape fails
                        rest_reshaped = np.zeros((N, 3, K - 1), dtype=np.float32)
                        for kk in range(1, K):
                            base = (kk - 1) * 3
                            rest_reshaped[:, 0, kk - 1] = rest_vals[:, base + 0]
                            rest_reshaped[:, 1, kk - 1] = rest_vals[:, base + 1]
                            rest_reshaped[:, 2, kk - 1] = rest_vals[:, base + 2]

                    dense[:, :, 1:] = rest_reshaped

                # build storage from dense representation
                dense_t = torch.tensor(dense, dtype=torch.float32, device=device)
                sh_degrees_np = np.full((N,), int(math.isqrt(K) - 1), dtype=np.int64)
                self.sh_storage = self._build_storage_from_dense(dense_t, torch.tensor(sh_degrees_np, dtype=torch.int32, device=device))
                self._sync_sh_degrees_from_storage()
                self.active_sh_degree = int(self.sh_degrees.max().item()) if self.sh_degrees.numel() > 0 else 0
            else:
                raise RuntimeError("PLY does not contain compact SH metadata ('sh_offset'/'sh_num_coeffs') nor legacy f_dc_/f_rest_ fields.")


    """
    def save_ply(self, path, roundtrip_check: bool = False):
        #Save compact PLY: vertex attributes in 'vertex' element and
        #all SH coefficients as a separate 'sh_coeff' element, plus per-gaussian
        #metadata in 'gaussian_meta'. This avoids per-vertex variable fields.
        
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Packed SH coefficients and per-gaussian metadata
        sh_coeffs = self.sh_storage.sh_coeffs_flat.detach().cpu().numpy().astype(np.float32)
        offsets = self.sh_storage.gauss_offsets.detach().cpu().numpy().astype(np.int32)
        counts = self.sh_storage.num_coeffs_per_gauss.detach().cpu().numpy().astype(np.int32)
        degrees = self.sh_storage.sh_degrees.detach().cpu().numpy().astype(np.int32)

        # Vertex dtype (no per-vertex SH fields)
        dtype_vert = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('opacity', 'f4')
        ]
        for i in range(scale.shape[1]):
            dtype_vert.append((f'scale_{i}', 'f4'))
        for i in range(rotation.shape[1]):
            dtype_vert.append((f'rot_{i}', 'f4'))

        attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        vertex_el = np.empty(xyz.shape[0], dtype=dtype_vert)
        vertex_el[:] = list(map(tuple, attributes))
        vertex = PlyElement.describe(vertex_el, 'vertex')

        # SH coefficients element: each row is one coeff [r,g,b]
        sh_dtype = [('r', 'f4'), ('g', 'f4'), ('b', 'f4')]
        sh_el = np.empty(sh_coeffs.shape[0], dtype=sh_dtype)
        sh_el['r'] = sh_coeffs[:, 0]
        sh_el['g'] = sh_coeffs[:, 1]
        sh_el['b'] = sh_coeffs[:, 2]
        sh_element = PlyElement.describe(sh_el, 'sh_coeff')

        # Meta element: per-gaussian offset, count, degree
        meta_dtype = [('offset', 'i4'), ('count', 'i4'), ('degree', 'i4')]
        meta_el = np.empty(offsets.shape[0], dtype=meta_dtype)
        meta_el['offset'] = offsets
        meta_el['count'] = counts
        meta_el['degree'] = degrees
        meta_element = PlyElement.describe(meta_el, 'gaussian_meta')

        PlyData([vertex, sh_element, meta_element]).write(path)

        if roundtrip_check:
            try:
                gm2 = GaussianModel(self.max_sh_degree)
                gm2.load_ply(path)
                orig_state = self.sh_storage.serialize() if self.sh_storage is not None else None
                new_state = gm2.sh_storage.serialize() if gm2.sh_storage is not None else None
                if (orig_state is None) != (new_state is None):
                    raise RuntimeError("PLY roundtrip failed: missing sh_storage after load")
                if orig_state is not None:
                    orig_counts = orig_state["num_coeffs_per_gauss"].cpu().numpy()
                    new_counts = new_state["num_coeffs_per_gauss"].cpu().numpy()
                    if orig_counts.shape != new_counts.shape or not np.array_equal(orig_counts, new_counts):
                        raise RuntimeError("PLY roundtrip failed: coeff counts differ")
                    orig_degs = orig_state["sh_degrees"].cpu().numpy()
                    new_degs = new_state["sh_degrees"].cpu().numpy()
                    if orig_degs.shape != new_degs.shape or not np.array_equal(orig_degs, new_degs):
                        raise RuntimeError("PLY roundtrip failed: sh_degrees differ")
                    orig_coeffs = orig_state["sh_coeffs_flat"].cpu().numpy()
                    new_coeffs = new_state["sh_coeffs_flat"].cpu().numpy()
                    if orig_coeffs.shape != new_coeffs.shape or not np.allclose(orig_coeffs, new_coeffs, atol=1e-6):
                        raise RuntimeError("PLY roundtrip failed: sh_coeffs_flat differ (values or shape)")
                print(f"PLY roundtrip check OK: {path}")
            except Exception as e:
                print(f"PLY roundtrip check failed: {e}")
                raise
    
    def load_ply(self, path):
        #Load compact PLY written by `save_compact_ply` and reconstruct
        #`sh_storage` and vertex attributes.
        
        ply = PlyData.read(path)

        v = ply['vertex'].data
        xyz = np.vstack((v['x'], v['y'], v['z'])).T

        # normals are not stored (zeros)
        # read opacity if present
        if 'opacity' in v.dtype.names:
            opacities = np.asarray(v['opacity'])[..., np.newaxis]
        else:
            opacities = np.ones((xyz.shape[0], 1), dtype=np.float32)

        # read scales and rotations if present
        scale_names = [n for n in v.dtype.names if n.startswith('scale_')]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, name in enumerate(scale_names):
            scales[:, idx] = np.asarray(v[name])

        rot_names = [n for n in v.dtype.names if n.startswith('rot_')]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, name in enumerate(rot_names):
            rots[:, idx] = np.asarray(v[name])

        # read sh_coeff element
        sh_el = ply['sh_coeff'].data
        sh_coeffs = np.vstack((sh_el['r'], sh_el['g'], sh_el['b'])).T.astype(np.float32)

        # read meta
        meta = ply['gaussian_meta'].data
        offsets = np.asarray(meta['offset']).astype(np.int32)
        counts = np.asarray(meta['count']).astype(np.int32)
        degrees = np.asarray(meta['degree']).astype(np.int32)

        # assign tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        # rebuild storage
        storage = SHStorage(num_gaussians=offsets.shape[0], init_deg=0, max_degree=self.max_sh_degree, device=device)
        storage.sh_coeffs_flat = nn.Parameter(torch.tensor(sh_coeffs, dtype=torch.float32, device=device))
        storage.gauss_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
        storage.num_coeffs_per_gauss = torch.tensor(counts, dtype=torch.int32, device=device)
        storage.sh_degrees = torch.tensor(degrees, dtype=torch.int32, device=device)
        storage.num_gauss = offsets.shape[0]
        storage.num_gaussians = offsets.shape[0]
        self.sh_storage = storage
        self._sync_sh_degrees_from_storage()
        self.active_sh_degree = int(self.sh_degrees.max().item()) if self.sh_degrees.numel() > 0 else 0
    """
        
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

    def _prune_optimizer(self, mask, extra_masks=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            param_mask = mask
            if extra_masks and name in extra_masks:
                param_mask = extra_masks[name]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][param_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][param_mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][param_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][param_mask].requires_grad_(True))
                optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        Prunes gaussians marked as False in mask.
        Verifies consistency before and after pruning.
        """
        # Consistency check before prune
        #print(f"[prune_points] Before prune: checking consistency...")
        #self._validate_gaussian_consistency("[before prune_points]")
        
        valid_points_mask = ~mask
        num_to_prune = mask.sum().item()
        num_remaining = valid_points_mask.sum().item()
        #print(f"[prune_points] Pruning {num_to_prune} gaussians, keeping {num_remaining}")
        
        counts = self.sh_storage.num_coeffs_per_gauss.to(device=valid_points_mask.device)
        if counts.shape[0] != valid_points_mask.shape[0]:
            raise RuntimeError(
                f"Mismatch between coeff counts ({counts.shape[0]}) and gaussian mask ({valid_points_mask.shape[0]})"
            )
        coeff_mask = torch.repeat_interleave(valid_points_mask.to(dtype=torch.long), counts).bool()
        optimizable_tensors = self._prune_optimizer(valid_points_mask, {"sh_coeffs": coeff_mask})

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # New
        self.sh_storage.prune_gaussians(valid_points_mask, optimizable_tensors.get("sh_coeffs"))
        self._sync_sh_degrees_from_storage()
        # prune last_promotion_iter flags to keep in sync with gaussians
        if hasattr(self, "last_promotion_iter") and self.last_promotion_iter is not None and self.last_promotion_iter.numel() != 0:
            try:
                self.last_promotion_iter = self.last_promotion_iter[valid_points_mask]
            except Exception:
                # fallback: reset to -1 if sizes mismatch
                n = int(self.sh_degrees.shape[0]) if self.sh_degrees is not None and self.sh_degrees.numel() > 0 else 0
                device = self.sh_storage.sh_coeffs_flat.device if self.sh_storage is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.last_promotion_iter = torch.full((n,), -1, device=device, dtype=torch.int64)
        
        # Consistency check after prune
        #print(f"[prune_points] After prune: checking consistency...")
        #self._validate_gaussian_consistency("[after prune_points]")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            name = group["name"]
            if name not in tensors_dict:
                continue
            extension_tensor = tensors_dict[name]
            if extension_tensor is None:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if name == "sh_coeffs":
                old_param = group["params"][0]
                new_param = self.sh_storage.sh_coeffs_flat
                add_rows = new_param.shape[0] - old_param.shape[0]
                if add_rows < 0:
                    raise RuntimeError("Packed SH parameter shrank unexpectedly during densification")
                if stored_state is None:
                    stored_state = {
                        "exp_avg": torch.zeros_like(new_param),
                        "exp_avg_sq": torch.zeros_like(new_param),
                    }
                else:
                    zeros_shape = (add_rows,) + stored_state["exp_avg"].shape[1:]
                    device = stored_state["exp_avg"].device
                    dtype = stored_state["exp_avg"].dtype
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros(zeros_shape, device=device, dtype=dtype)),
                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros(zeros_shape, device=device, dtype=dtype)),
                        dim=0)
                if old_param in self.optimizer.state:
                    del self.optimizer.state[old_param]
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state
                optimizable_tensors[name] = new_param
                continue
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_sh_coeffs=None, new_last_promotion_iter=None):
        d = {
            "xyz": new_xyz,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if new_sh_coeffs is not None:
            d["sh_coeffs"] = new_sh_coeffs

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "sh_coeffs" in optimizable_tensors:
            self.sh_storage.sh_coeffs_flat = optimizable_tensors["sh_coeffs"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._sync_sh_degrees_from_storage()
        # append last_promotion_iter entries for newly created gaussians (if provided), otherwise append -1
        try:
            n_existing = int(self.last_promotion_iter.shape[0]) if hasattr(self, "last_promotion_iter") and self.last_promotion_iter is not None and self.last_promotion_iter.numel() > 0 else 0
        except Exception:
            n_existing = 0
        n_new = int(self.get_xyz.shape[0]) - n_existing
        device = self.sh_storage.sh_coeffs_flat.device if self.sh_storage is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if new_last_promotion_iter is not None:
            # ensure correct device and dtype and length
            new_last_promotion_iter = new_last_promotion_iter.to(device=device, dtype=torch.int64)
            if new_last_promotion_iter.numel() != n_new:
                # if clone ids repeated, allow broadcasting or trim/pad
                if new_last_promotion_iter.numel() < n_new:
                    pad = torch.full((n_new - new_last_promotion_iter.numel(),), -1, device=device, dtype=torch.int64)
                    new_last_promotion_iter = torch.cat([new_last_promotion_iter, pad], dim=0)
                else:
                    new_last_promotion_iter = new_last_promotion_iter[:n_new]
            if n_existing > 0:
                self.last_promotion_iter = torch.cat([self.last_promotion_iter.to(device=device), new_last_promotion_iter], dim=0)
            else:
                self.last_promotion_iter = new_last_promotion_iter
        else:
            # extend with -1
            pad = torch.full((n_new,), -1, device=device, dtype=torch.int64)
            if n_existing > 0:
                self.last_promotion_iter = torch.cat([self.last_promotion_iter.to(device=device), pad], dim=0)
            else:
                self.last_promotion_iter = pad


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, debug=False):
        """
        Splits gaussians with high gradients and large size.
        Updates the SH optimizer state such that newly cloned gaussians keep their moments.
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if debug:
            # Choose second gaussian for debugging, temporary
            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[1] = True

        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(-1)
        if selected_indices.numel() == 0:
            return

        print(f"[densify_and_split] Splitting {selected_indices.shape[0]} gaussians (x{N})")
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # Neue SH Speicherung - record the index of old gaussians before duplication
        clone_ids = selected_indices.repeat_interleave(N)
        num_old_gaussians = self._xyz.shape[0]
        new_sh_coeffs = self.sh_storage.duplicate_sh_of_gaussians(clone_ids)

        # duplicate last_promotion_iter flags for the new gaussians
        if hasattr(self, "last_promotion_iter") and self.last_promotion_iter is not None and self.last_promotion_iter.numel() > 0:
            new_last = self.last_promotion_iter[clone_ids]
        else:
            new_last = None

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_sh_coeffs=new_sh_coeffs, new_last_promotion_iter=new_last)

        # Extended SH optimizer state: replicate moments for newly cloned gaussians
        self._extend_sh_optimizer_state_for_densification(clone_ids)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
        # Consistency check after split
        #self._validate_gaussian_consistency("[after densify_and_split]")

    def densify_and_clone(self, grads, grad_threshold, scene_extent, debug=False):
        """
        Clones gaussians with high gradients and small size.
        Updates the SH optimizer state such that newly cloned gaussians keep their moments.
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if debug:
            # Choose second gaussian for debugging, temporary
            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[1] = True
        
        num_clones = selected_pts_mask.sum().item()
        print(f"[densify_and_clone] Cloning {num_clones} gaussians")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # Neue SH Speicherung - record the index of old gaussians before duplication
        new_sh_coeffs = self.sh_storage.duplicate_sh_of_gaussians(selected_pts_mask)

        # duplicate last_promotion_iter for cloned gaussians
        if hasattr(self, "last_promotion_iter") and self.last_promotion_iter is not None and self.last_promotion_iter.numel() > 0:
            # selected_pts_mask can be boolean mask or indices
            if selected_pts_mask.dtype == torch.bool:
                clone_ids = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(-1)
            else:
                clone_ids = selected_pts_mask
            new_last = self.last_promotion_iter[clone_ids]
        else:
            new_last = None
            clone_ids = None

        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_sh_coeffs=new_sh_coeffs, new_last_promotion_iter=new_last)

        # Extended SH optimizer state: replicate moments for newly cloned gaussians
        if clone_ids is not None:
            self._extend_sh_optimizer_state_for_densification(clone_ids)
        
        # Consistency check after clone
        #self._validate_gaussian_consistency("[after densify_and_clone]")

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
        
        num_to_prune = prune_mask.sum().item()
        print(f"[densify_and_prune] Pruning {num_to_prune} gaussians")
        
        self.prune_points(prune_mask)
        
        # Consistency check after prune
        #self._validate_gaussian_consistency("[after densify_and_prune]")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    # New: Old cumulate_color_gradients which doesn't average over calls
    """
    def cumulate_color_gradients(self): 
        if(self.color_denom == 0):
            #print("Initializing color gradient accumulation tensors...")
            self.accum_color_grads_dc = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            self.accum_color_grads_rest = torch.zeros((self.get_xyz.shape[0],), device = "cuda")
        if self.sh_storage.sh_coeffs_flat.grad is None:
            return

        grad = self.sh_storage.sh_coeffs_flat.grad
        device = grad.device
        grad_norm = torch.norm(grad, dim=1)
        counts = self.sh_storage.num_coeffs_per_gauss.to(device=device, dtype=torch.long)
        total = int(counts.sum().item())
        if total == 0:
            return
        gauss_ids = torch.repeat_interleave(torch.arange(counts.shape[0], device=device, dtype=torch.long), counts)
        local_idx = torch.arange(total, device=device, dtype=torch.long) - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)

        dc_mask = local_idx == 0
        rest_mask = ~dc_mask

        dc_accum = torch.zeros(counts.shape[0], device=device)
        rest_accum = torch.zeros(counts.shape[0], device=device)
        dc_accum.scatter_add_(0, gauss_ids[dc_mask], grad_norm[dc_mask])
        rest_accum.scatter_add_(0, gauss_ids[rest_mask], grad_norm[rest_mask])

        self.accum_color_grads_dc += dc_accum
        self.accum_color_grads_rest += rest_accum
        self.color_denom += 1 
    """

    # New cumulate_color_gradients which averages over calls and tracks seen counts
    def cumulate_color_gradients(self): 
        if self.sh_storage is None:
            return

        # ensure accum buffers exist and have correct length N
        counts = self.sh_storage.num_coeffs_per_gauss.to(device=self.sh_storage.sh_coeffs_flat.device, dtype=torch.long)
        N = counts.shape[0]
        device = self.sh_storage.sh_coeffs_flat.device

        # If any of the accumulators do not match N, (re)initialize them to zeros of length N
        if self.accum_color_grads_dc.numel() != N:
            self.accum_color_grads_dc = torch.zeros((N,), device=device)
        if self.accum_color_grads_rest.numel() != N:
            self.accum_color_grads_rest = torch.zeros((N,), device=device)
        #if self.color_seen_counts.numel() != N:
            #self.color_seen_counts = torch.zeros((N,), device=device, dtype=torch.float32)
        # ensure color_denom is defined
        if not isinstance(self.color_denom, (int, float)):
            self.color_denom = 0

        if self.sh_storage.sh_coeffs_flat.grad is None:
            return

        grad = self.sh_storage.sh_coeffs_flat.grad
        grad_norm = torch.norm(grad, dim=1)
        total = int(counts.sum().item())
        if total == 0:
            return

        gauss_ids = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.long), counts)
        local_idx = torch.arange(total, device=device, dtype=torch.long) - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)

        dc_mask = local_idx == 0
        rest_mask = ~dc_mask

        dc_accum = torch.zeros(N, device=device)
        rest_accum = torch.zeros(N, device=device)
        if dc_mask.any():
            dc_accum.scatter_add_(0, gauss_ids[dc_mask], grad_norm[dc_mask])
        if rest_mask.any():
            rest_accum.scatter_add_(0, gauss_ids[rest_mask], grad_norm[rest_mask])

        # increment per-gaussian accumulators
        self.accum_color_grads_dc += dc_accum
        self.accum_color_grads_rest += rest_accum

        '''
        # update per-gaussian seen counts: mark a gaussian as seen if any of its coeffs had a non-zero grad
        seen_mask = grad_norm > 1e-12
        if seen_mask.any():
            seen_inc = torch.zeros(N, device=device, dtype=self.color_seen_counts.dtype)
            # ones for the seen coeffs
            one_vals = torch.ones((seen_mask.sum().item(),), device=device, dtype=self.color_seen_counts.dtype)
            seen_inc.scatter_add_(0, gauss_ids[seen_mask], one_vals)
            # clamp to at most 1 per gaussian per call
            seen_inc = (seen_inc > 0).to(self.color_seen_counts.dtype)
            self.color_seen_counts += seen_inc
        '''
        # keep a global counter for diagnostics
        self.color_denom += 1


    # New methods for SH degree management
    def get_max_sh_degree_in_model(self):
        return int(self.sh_degrees.max())
    
    def set_sh_degrees_by_indices(self, indices, degree):
        if indices.numel() == 0:
            return
        assert degree <= self.max_sh_degree and degree >=0, f"Degree {degree} is out of bounds [0,{self.max_sh_degree}]"
        assert indices.max() < self.sh_degrees.shape[0], f"Index {indices.max()} is out of bounds [0,{self.sh_degrees.shape[0]-1}]"
        new_degrees = self.sh_degrees.clone()
        new_degrees[indices.long()] = degree
        self._apply_new_sh_degrees(new_degrees)

    def set_random_sh_degrees(self):
        n_points = self.sh_degrees.shape[0]
        random_degrees = torch.randint(0, self.max_sh_degree + 1, (n_points,), device=self.sh_degrees.device, dtype=torch.int64)
        self._apply_new_sh_degrees(random_degrees)

    def get_sh_degree_distribution(self, schedule_name="", iteration=0):
        total_points = self.sh_degrees.shape[0]
        print(f"Total Gaussians: {total_points}")
        unique, counts = torch.unique(self.sh_degrees, return_counts=True)
        for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            print(f"SH degree {u}: {c/total_points*100:.2f}% Gaussians")
        # Save to CSV in new line:
        with open(f"sh_degree_distribution_{schedule_name}.csv", "a") as f:
            for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                f.write(f"{iteration},{u},{c/total_points*100:.2f},{c}\n")

    def get_sh_degrees_by_indices(self, indices):
        return self.sh_degrees[indices]
    
    def randomly_increase_sh_degrees_by_one(self, fraction):
        n_points = self.sh_degrees.shape[0]
        n_increase = int(n_points * fraction)
        if n_increase == 0:
            print("Random increase skipped because no Gaussians were selected.")
            return
        all_indices = torch.arange(n_points, device=self.sh_degrees.device)
        selected_indices = all_indices[torch.randperm(n_points)[:n_increase]]
        valid = selected_indices[self.sh_degrees[selected_indices] < self.max_sh_degree]
        if valid.numel() == 0:
            print("Random increase requested but no Gaussians can be upgraded.")
            return
        new_degrees = self.sh_degrees.clone()
        new_degrees[valid] += 1
        self._apply_new_sh_degrees(new_degrees)

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

    def color_gradients_postfix(self):
        self.accum_color_grads_dc = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        self.accum_color_grads_rest = torch.zeros((self.get_xyz.shape[0],), device="cuda")
        #self.color_seen_counts = torch.zeros((self.get_xyz.shape[0],), device="cuda", dtype=torch.float32)
        self.color_denom = 0
      
    def get_sh_degree_colors(self):
        degree_colors = {
            0: torch.tensor([0.5, 0.5, 0.5], device="cuda"),
            1: torch.tensor([0.0, 1.0, 0.0], device="cuda"),
            2: torch.tensor([0.0, 0.0, 1.0], device="cuda"),
            3: torch.tensor([1.0, 0.0, 0.0], device="cuda"),
        }
        max_defined = max(degree_colors.keys())
        sh_deg = self.sh_degrees
        colors = torch.zeros((sh_deg.shape[0], 3), device="cuda")
        with torch.no_grad():
            for d in torch.unique(sh_deg):
                d_int = int(d.item())
                color = degree_colors[d_int] if d_int in degree_colors else degree_colors[d_int % (max_defined + 1)]
                mask = (sh_deg == d)
                colors[mask] = color
        return colors.cpu().numpy()
    
    # Old method
    
    def increase_sh_degree_based_on_color_grads(self, iteration: int, cool_down_iter: int, ratio: float = 0.05, maximum_degree: int = 3, only_for_degree = None):
        """
        Increase SH degree for the top `ratio` fraction of gaussians by accumulated color gradients.

        Only gaussians satisfying (iteration - last_promotion_iter) > cool_down_iter are eligible. Gaussians with
        last_promotion_iter == -1 (never promoted) are immediately eligible.
        """
        valid_degree = (self.sh_degrees < maximum_degree)
        if only_for_degree is not None:
            valid_degree = valid_degree & (self.sh_degrees == only_for_degree)

        # determine eligibility based on cool-down
        if hasattr(self, "last_promotion_iter") and self.last_promotion_iter is not None and self.last_promotion_iter.numel() == self.sh_degrees.numel():
            # last_promotion_iter == -1 means never promoted -> eligible
            eligible = (self.last_promotion_iter < 0) | ((iteration - self.last_promotion_iter) > cool_down_iter)
        else:
            eligible = torch.ones_like(self.sh_degrees, dtype=torch.bool)

        candidate_mask = valid_degree & eligible
        if candidate_mask.sum().item() == 0:
            print("No Gaussians qualified for SH degree increase (degree/cooldown filter).")
            return

        try:
            quantile_value = torch.quantile(self.accum_color_grads_dc[candidate_mask], 1.0 - ratio)
        except Exception:
            print("Unable to compute quantile for SH degree increase.")
            return

        # select by accumulated dc grads above quantile, respecting degree & cooldown
        valid = (self.accum_color_grads_dc > quantile_value).squeeze() & valid_degree & eligible

        if valid.sum().item() == 0:
            print("No Gaussians qualified for SH degree increase after thresholding.")
            return

        new_degrees = self.sh_degrees.clone()
        new_degrees[valid] += 1
        self._apply_new_sh_degrees(new_degrees)

        # update last_promotion_iter for those promoted
        try:
            if not hasattr(self, "last_promotion_iter") or self.last_promotion_iter is None or self.last_promotion_iter.numel() != self.sh_degrees.numel():
                device = self.sh_storage.sh_coeffs_flat.device if self.sh_storage is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.last_promotion_iter = torch.full((self.sh_degrees.shape[0],), -1, device=device, dtype=torch.int64)
            self.last_promotion_iter[valid] = int(iteration)
        except Exception:
            # best-effort: ignore update on error
            pass

        updated_percentage = valid.sum().item() / max(1, self.sh_degrees.shape[0]) * 100.0
        print(f"Increased SH degree for {valid.sum().item()} Gaussians ({updated_percentage:.2f}%) based on color gradients.")
    
    '''
    # New method that uses averaged gradients and ratio of rest vs dc and seen counts
    def increase_sh_degree_based_on_color_grads(self, ratio=0.05, maximum_degree=3):
        # use normalized averages (per-gaussian) to avoid bias from visibility frequency
        if self.color_seen_counts.numel() == 0 or self.color_seen_counts.sum() == 0:
            print("No color gradient statistics collected yet.")
            return

        eps = 1e-6
        dc_avg = self.accum_color_grads_dc / (self.color_seen_counts + eps)
        rest_avg = self.accum_color_grads_rest / (self.color_seen_counts + eps)

        # score: how large rest is relative to dc (higher => worth increasing degree)
        score = dc_avg

        # pick top fraction by score
        quantile_value = torch.quantile(score, 1.0 - ratio)
        to_increase = score > quantile_value
        valid = to_increase & (self.sh_degrees < maximum_degree)
        print(valid.sum().item(), "Gaussians will have their SH degree increased.")
        if valid.sum().item() == 0:
            print("No Gaussians qualified for SH degree increase.")
            return

        new_degrees = self.sh_degrees.clone()
        new_degrees[valid] += 1
        self._apply_new_sh_degrees(new_degrees)

        updated_percentage = valid.sum().item() / max(1, to_increase.sum().item()) * 100.0
        print(f"Increased SH degree for {valid.sum().item()} Gaussians ({updated_percentage:.2f}%), given ratio threshold was {ratio}")
        # print variance of seen counts for diagnostics
        seen_var = torch.var(self.color_seen_counts.float()).item()
        print(f"Variance of color gradient seen counts: {seen_var:.6f}")
        seen_avg = torch.mean(self.color_seen_counts.float()).item()
        print(f"Average of color gradient seen counts: {seen_avg:.6f}")
    '''
    
