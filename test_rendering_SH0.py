# Compare renderings between original and new repo only using SH0
import os
import torch
import torchvision
import torchvision.io as io

#original_renders_dir = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4/test/old_repo_renders_30000/renders"  # TODO: set this path
#new_renders_dir = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4/test/new_repo_renders_30000/renders"  # TODO: set this path

def compare_renderings(original_dir, new_dir, tolerance=1e-3):
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')])
    new_files = sorted([f for f in os.listdir(new_dir) if f.endswith('.png')])

    #assert original_files == new_files, "Mismatch in rendering files between original and new repo."

    for filename in original_files:
        original_path = os.path.join(original_dir, filename)
        new_path = os.path.join(new_dir, filename)

        original_image = torch.tensor(torchvision.io.read_image(original_path), dtype=torch.float32) / 255.0
        new_image = torch.tensor(torchvision.io.read_image(new_path), dtype=torch.float32) / 255.0

        if not torch.allclose(original_image, new_image, atol=tolerance):
            diff = torch.abs(original_image - new_image).mean().item()
            print(f"Rendering mismatch in {filename}: mean difference {diff}")
        else:
            print(f"{filename} matches.")

if __name__ == "__main__":
    original_renders_dir = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4/test/old_repo_renders_SH3_30000/renders"  # TODO: set this path
    new_renders_dir = "/mnt/data/3dgs_out/garden__performance_of_SH3_original_repo_1_iter30000_images4/test/new_repo_renders_SH3_with_correct_load_ply_30000/renders"  # TODO: set this path
    compare_renderings(original_renders_dir, new_renders_dir)