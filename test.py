from scene import GaussianModel, Scene
from gaussian_renderer import render as grender
import os
import shutil
import argparse
import torch
from scene.sh_storage_new import SHStorage



def save_converted_ply(input_ply: str, model_path: str, iteration: int):
	print(iteration)
	gauss = GaussianModel(3)
	print(f"Loading PLY {input_ply} into GaussianModel(max_sh_degree=3)")
	gauss.load_ply(input_ply)
	print(f"Saving converted PLY to new repo format")
	out_dir = os.path.join(model_path, "point_cloud_new_repo", f"iteration_{iteration}")
	os.makedirs(out_dir, exist_ok=True)
	out_ply = os.path.join(out_dir, "point_cloud_new_repo.ply")
	gauss.save_ply(out_ply)
	print("Saved converted PLY to:")
	print(out_ply)


def main():
	old_ply = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4/point_cloud/iteration_30000/point_cloud.ply"
	new_model_path = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4"
	iteration = 30000
	print("Converting legacy PLY to new repo format...")
	save_converted_ply(old_ply, new_model_path, iteration)

def main2testlayoutofloadedSH():
	model_path = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4/point_cloud/iteration_30000/point_cloud.ply"
	iteration = 30000
	gm = GaussianModel(sh_degree=3)  # or appropriate max degree
	print(f"Loading PLY {model_path} into GaussianModel(max_sh_degree=3)")
	gm.load_ply(model_path)
	print("Loaded GaussianModel:")
	storage = gm.sh_storage
	# print coeffs for gaussian index 4
	offset = int(storage.gauss_offsets[4].item())
	count = int(storage.num_coeffs_per_gauss[4].item())
	coeffs = storage.sh_coeffs_flat[offset:offset+count]
	print(coeffs)

def test_duplicate_sh_of_gaussians():
	
	device = torch.device("cuda")
	storage = SHStorage(num_gaussians=10, init_deg=1, device=device)

	# Initialisiere mit einfachen Werten zum Testen
	storage.sh_coeffs_flat.data[:] = torch.arange(storage.sh_coeffs_flat.shape[0], device=device).float().unsqueeze(-1) / 100.0

	# Klone Gaussians 2, 5
	clone_ids = torch.tensor([2, 5], device=device, dtype=torch.long)
	old_coeffs = storage._gather_coeff_blocks(clone_ids)
	print(f"Old coeffs shape: {old_coeffs.shape}")
	print(f"Old coeffs:\n{old_coeffs}")

	storage.duplicate_sh_of_gaussians(clone_ids)

	# Erwartung: neue Gaussians 10, 11 sollten die gleichen Koeffizienten haben wie 2, 5
	new_coeffs = storage._gather_coeff_blocks(torch.tensor([10, 11], device=device, dtype=torch.long))
	print(f"New coeffs:\n{new_coeffs}")
	print(f"Are they equal? {torch.allclose(old_coeffs, new_coeffs, atol=1e-6)}")


def test_mapping():
	device = torch.device("cpu")
	storage = SHStorage(num_gaussians=5, init_deg=1, device=device)

	# Setze Koeffizienten auf einfache Werte
	for i in range(storage.sh_coeffs_flat.shape[0]):
		storage.sh_coeffs_flat.data[i] = torch.full((3,), i, dtype=storage.sh_coeffs_flat.dtype)

	old_coeffs = storage.sh_coeffs_flat.data.clone()

	# Erhöhe degree von Gaussian 0 und 2
	new_degrees = storage.sh_degrees.clone()
	new_degrees[0] = 2
	new_degrees[2] = 2

	mapping = storage._repack_all(new_degrees)

	# Überprüfe: alte Koeffizienten sollten an neue Stellen gemappt worden sein
	for i in range(old_coeffs.shape[0]):
		new_idx = mapping[i].item()
		if not torch.allclose(old_coeffs[i], storage.sh_coeffs_flat.data[new_idx], atol=1e-6):
			print(f"ERROR: coeff[{i}] not correctly remapped to new_idx[{new_idx}]")
			print(f"  old: {old_coeffs[i]}, new: {storage.sh_coeffs_flat.data[new_idx]}")
	
if __name__ == '__main__':
    test_mapping()