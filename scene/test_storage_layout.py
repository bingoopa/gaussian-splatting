# Test, whether the SHs are correctly stored and loaded in the GaussianModel's storage layout.
import torch
from scene import Scene
import os
from gaussian_renderer import GaussianModel
from arguments import ModelParams   

def test_storage_layout():
    #Set up small gaussian model with 5 gaussians
    sh_degree = 3
    num_gaussians = 5
    sh_degrees = [1, 2, 3, 0, 1]
    gaussians = GaussianModel(sh_degree)
    gaussians.sh_storage = torch.zeros((num_gaussians, (sh_degree + 1) ** 2, 3), dtype=torch.float32, device="cuda")