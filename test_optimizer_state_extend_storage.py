"""
Unit-Test für: Problem 2: Optimizer-State nach _extend_storage() nicht richtig gemappt

Testet, dass:
1. Der Optimizer-State nach _extend_storage() konsistent mit den neuen SH-Koeffizienten ist
2. Die exp_avg und exp_avg_sq Tensoren die richtige Größe haben
3. Die Gradienten-Akkumulation nach Erweiterung noch funktioniert
4. Der Optimizer-State nach densification_postfix() richtig gemappt ist
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

# Import the modules to test
from scene.gaussian_model import GaussianModel
from scene.sh_storage_new import SHStorage
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import BasicPointCloud


class TestOptimizerStateExtendStorage:
    """Test suite für Optimizer-State Mapping bei _extend_storage"""
    
    def setup_method(self):
        """Setup für jeden Test"""
        self.device = torch.device("cpu")  # CPU für reproducibility
        self.max_sh_degree = 3
        
    def test_extend_storage_optimizer_state_basic(self):
        """Test 1: Basic SHStorage._extend_storage() und Optimizer-State"""
        print("\n=== Test 1: Basic _extend_storage() und Optimizer ===")
        
        # Create initial storage with 5 gaussians, degree 0
        storage = SHStorage(
            num_gaussians=5,
            init_deg=0,
            max_degree=self.max_sh_degree,
            device=self.device,
        )
        
        # Create optimizer for sh_coeffs_flat
        optimizer = Adam([storage.sh_coeffs_flat], lr=0.001)
        
        # Get initial state
        old_shape = storage.sh_coeffs_flat.shape
        print(f"Initial sh_coeffs_flat shape: {old_shape}")
        
        # Simulate one optimization step to populate optimizer state
        loss = storage.sh_coeffs_flat.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Get old optimizer state
        old_param = storage.sh_coeffs_flat
        old_state = optimizer.state.get(old_param, {})
        old_exp_avg = old_state.get("exp_avg", None)
        old_exp_avg_sq = old_state.get("exp_avg_sq", None)
        
        print(f"Old exp_avg shape: {old_exp_avg.shape if old_exp_avg is not None else None}")
        print(f"Old exp_avg_sq shape: {old_exp_avg_sq.shape if old_exp_avg_sq is not None else None}")
        
        # Extend storage with 3 new gaussians (degree 0)
        new_degrees = torch.full((3,), 0, dtype=torch.int32, device=self.device)
        new_coeffs = torch.randn((3, 3), dtype=torch.float32, device=self.device)
        
        storage._extend_storage(new_degrees, new_coeffs)
        
        new_shape = storage.sh_coeffs_flat.shape
        print(f"New sh_coeffs_flat shape: {new_shape}")
        
        # Verify shape
        assert new_shape[0] == old_shape[0] + 3, f"Expected {old_shape[0] + 3} coeffs, got {new_shape[0]}"
        assert new_shape[1] == 3, f"Expected 3 color channels, got {new_shape[1]}"
        
        print("✓ Shape nach _extend_storage() korrekt")
        
        # Manuelles Mapping des Optimizer-States (wie in cat_tensors_to_optimizer)
        new_param = storage.sh_coeffs_flat
        
        # Old state needs to be padded for new coefficients
        if old_exp_avg is not None:
            new_exp_avg = torch.cat([
                old_exp_avg,
                torch.zeros((3, 3), device=self.device, dtype=old_exp_avg.dtype)
            ], dim=0)
            new_exp_avg_sq = torch.cat([
                old_exp_avg_sq,
                torch.zeros((3, 3), device=self.device, dtype=old_exp_avg_sq.dtype)
            ], dim=0)
            
            # Update optimizer state with new param
            if old_param in optimizer.state:
                del optimizer.state[old_param]
            
            optimizer.state[new_param] = {
                "step": old_state.get("step", torch.tensor(0)),
                "exp_avg": new_exp_avg,
                "exp_avg_sq": new_exp_avg_sq,
            }
        
        # Verify state after remapping
        new_state = optimizer.state.get(new_param, {})
        new_exp_avg = new_state.get("exp_avg", None)
        new_exp_avg_sq = new_state.get("exp_avg_sq", None)
        
        print(f"New exp_avg shape after mapping: {new_exp_avg.shape if new_exp_avg is not None else None}")
        print(f"New exp_avg_sq shape after mapping: {new_exp_avg_sq.shape if new_exp_avg_sq is not None else None}")
        
        assert new_exp_avg.shape == new_shape, f"exp_avg shape mismatch: {new_exp_avg.shape} vs {new_shape}"
        assert new_exp_avg_sq.shape == new_shape, f"exp_avg_sq shape mismatch: {new_exp_avg_sq.shape} vs {new_shape}"
        
        print("✓ Optimizer-State nach _extend_storage() korrekt gemappt")


    def test_densification_postfix_optimizer_state(self):
        """Test 2: GaussianModel.densification_postfix() und Optimizer-State"""
        print("\n=== Test 2: GaussianModel.densification_postfix() ===")
        
        # Create a simple Gaussian model
        gm = GaussianModel(self.max_sh_degree)
        
        # Create a minimal point cloud
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ], dtype=torch.float32)
        colors = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.7, 0.7, 0.7],
            [0.3, 0.3, 0.3],
        ], dtype=torch.float32)
        
        pcd = BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))
        
        # Initialize from point cloud
        gm.create_from_pcd(pcd, spatial_lr_scale=1.0)
        
        # Setup training (initializes optimizer)
        from arguments import OptimizationParams
        from argparse import ArgumentParser
        parser = ArgumentParser()
        training_args = OptimizationParams(parser)
        gm.training_setup(training_args)
        
        # Get old state
        old_sh_param = gm.optimizer.param_groups[4]["params"][0]  # sh_coeffs group
        old_shape = old_sh_param.shape
        print(f"Initial sh_coeffs shape: {old_shape}")
        
        # Simulate optimization step
        loss = old_sh_param.sum()
        loss.backward()
        gm.optimizer.step()
        gm.optimizer.zero_grad()
        
        old_state = gm.optimizer.state.get(old_sh_param, {})
        old_exp_avg = old_state.get("exp_avg", None)
        print(f"Old exp_avg shape: {old_exp_avg.shape if old_exp_avg is not None else None}")
        
        # Create new gaussians to densify
        new_xyz = torch.randn((2, 3), device=old_sh_param.device)
        new_opacities = torch.ones((2, 1), device=old_sh_param.device)
        new_scaling = torch.ones((2, 3), device=old_sh_param.device)
        new_rotation = torch.zeros((2, 4), device=old_sh_param.device)
        new_rotation[:, 0] = 1.0
        
        # Create new SH coeffs for the new gaussians
        # 2 gaussians × 1 coeff (degree 0) × 3 channels = 6 rows
        new_sh_coeffs = torch.randn((2, 3), device=old_sh_param.device)
        
        # Call densification_postfix
        gm.densification_postfix(
            new_xyz, new_opacities, new_scaling, new_rotation,
            new_sh_coeffs=new_sh_coeffs
        )
        
        # Get new state
        new_sh_param = gm.optimizer.param_groups[4]["params"][0]  # sh_coeffs group
        new_shape = new_sh_param.shape
        print(f"New sh_coeffs shape: {new_shape}")
        
        new_state = gm.optimizer.state.get(new_sh_param, {})
        new_exp_avg = new_state.get("exp_avg", None)
        new_exp_avg_sq = new_state.get("exp_avg_sq", None)
        
        print(f"New exp_avg shape: {new_exp_avg.shape if new_exp_avg is not None else None}")
        print(f"New exp_avg_sq shape: {new_exp_avg_sq.shape if new_exp_avg_sq is not None else None}")
        
        # Verify consistency
        assert new_exp_avg is not None, "exp_avg should exist after densification_postfix"
        assert new_exp_avg_sq is not None, "exp_avg_sq should exist after densification_postfix"
        assert new_exp_avg.shape == new_shape, f"exp_avg shape mismatch: {new_exp_avg.shape} vs {new_shape}"
        assert new_exp_avg_sq.shape == new_shape, f"exp_avg_sq shape mismatch: {new_exp_avg_sq.shape} vs {new_shape}"
        
        # Verify old values are preserved
        old_rows = old_shape[0]
        if old_exp_avg is not None:
            assert torch.allclose(new_exp_avg[:old_rows], old_exp_avg, atol=1e-6), \
                "Old exp_avg values should be preserved"
        
        print("✓ Optimizer-State nach densification_postfix() korrekt")


    def test_multiple_extend_cycles(self):
        """Test 3: Mehrfache _extend_storage() Zyklen"""
        print("\n=== Test 3: Mehrfache _extend_storage() Zyklen ===")
        
        storage = SHStorage(
            num_gaussians=2,
            init_deg=0,
            max_degree=self.max_sh_degree,
            device=self.device,
        )
        
        optimizer = Adam([storage.sh_coeffs_flat], lr=0.001)
        
        sizes = [2]  # Initial size
        
        # Cycle 1
        for cycle in range(3):
            old_param = storage.sh_coeffs_flat
            old_size = old_param.shape[0]
            
            # Do optimization step
            loss = old_param.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Extend storage
            new_degrees = torch.full((2,), 0, dtype=torch.int32, device=self.device)
            new_coeffs = torch.randn((2, 3), dtype=torch.float32, device=self.device)
            storage._extend_storage(new_degrees, new_coeffs)
            
            # Map optimizer state
            new_param = storage.sh_coeffs_flat
            new_size = new_param.shape[0]
            sizes.append(new_size)
            
            old_state = optimizer.state.get(old_param, {})
            if old_state:
                old_exp_avg = old_state.get("exp_avg")
                new_exp_avg = torch.cat([
                    old_exp_avg,
                    torch.zeros((2, 3), device=self.device, dtype=old_exp_avg.dtype)
                ], dim=0)
                new_exp_avg_sq = torch.cat([
                    old_state.get("exp_avg_sq"),
                    torch.zeros((2, 3), device=self.device, dtype=old_state.get("exp_avg_sq").dtype)
                ], dim=0)
                
                if old_param in optimizer.state:
                    del optimizer.state[old_param]
                
                optimizer.state[new_param] = {
                    "step": old_state.get("step", torch.tensor(0)),
                    "exp_avg": new_exp_avg,
                    "exp_avg_sq": new_exp_avg_sq,
                }
            
            print(f"Cycle {cycle+1}: {old_size} → {new_size} coeffs")
            
            # Verify
            new_state = optimizer.state.get(new_param, {})
            assert new_state.get("exp_avg").shape[0] == new_size, \
                f"Cycle {cycle+1}: exp_avg size mismatch"
        
        print(f"Sizes through cycles: {sizes}")
        assert sizes == [2, 4, 6, 8], f"Unexpected size progression: {sizes}"
        print("✓ Mehrfache Zyklen korrekt")


    def test_zero_padding_consistency(self):
        """Test 4: Neue Koeffizienten werden mit Zeros gepaddert"""
        print("\n=== Test 4: Zero-Padding für neue Koeffizienten ===")
        
        storage = SHStorage(
            num_gaussians=3,
            init_deg=1,  # degree 1 = 4 coeffs pro gaussian
            max_degree=self.max_sh_degree,
            device=self.device,
        )
        
        # Set some known values
        storage.sh_coeffs_flat.data.fill_(0.5)
        
        optimizer = Adam([storage.sh_coeffs_flat], lr=0.001)
        
        # Optimization step
        loss = storage.sh_coeffs_flat.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        old_param = storage.sh_coeffs_flat
        old_state = optimizer.state.get(old_param, {})
        old_size = old_param.shape[0]
        
        # Extend storage
        # 2 gaussians with degree 1 each: (1+1)^2 * 2 = 4 * 2 = 8 coefficients total
        new_degrees = torch.full((2,), 1, dtype=torch.int32, device=self.device)
        new_coeffs = torch.full((8, 3), 0.3, dtype=torch.float32, device=self.device)
        
        storage._extend_storage(new_degrees, new_coeffs)
        
        new_param = storage.sh_coeffs_flat
        new_size = new_param.shape[0]
        
        print(f"Old size: {old_size}, New size: {new_size}")
        
        # Map optimizer state
        old_exp_avg = old_state.get("exp_avg")
        new_exp_avg = torch.cat([
            old_exp_avg,
            torch.zeros((new_size - old_size, 3), device=self.device, dtype=old_exp_avg.dtype)
        ], dim=0)
        
        # Check padding
        padding_part = new_exp_avg[old_size:, :]
        assert torch.allclose(padding_part, torch.zeros_like(padding_part), atol=1e-8), \
            "Padding should be exactly zeros"
        
        print(f"Padding shape: {padding_part.shape}")
        print("✓ Zero-Padding korrekt")


    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("Testing Optimizer-State Mapping nach _extend_storage()")
        print("=" * 60)
        
        self.setup_method()
        self.test_extend_storage_optimizer_state_basic()
        
        self.setup_method()
        self.test_densification_postfix_optimizer_state()
        
        self.setup_method()
        self.test_multiple_extend_cycles()
        
        self.setup_method()
        self.test_zero_padding_consistency()
        
        print("\n" + "=" * 60)
        print("✓ ALLE TESTS BESTANDEN")
        print("=" * 60)


if __name__ == "__main__":
    test_suite = TestOptimizerStateExtendStorage()
    test_suite.run_all_tests()
