import torch
from sh_storage import SHStorage

def test_adaptive_sh():
    N = 4
    storage = SHStorage(num_gaussians=N, init_deg=0, max_degree=3)

    # Init: alle haben 1 Koeff (deg=0)
    print("Base:", storage.gauss_offsets)
    print("Num:", storage.num_coeffs_per_gauss)
    print("Deg:", storage.sh_degrees)

    # Mach ein bisschen Random-Init
    with torch.no_grad():
        storage.sh_coeffs_flat.uniform_(-0.1, 0.1)

    dense = storage.build_dense_sh(max_degree=3)
    print("Dense shape:", dense.shape)  # [4, 3, 16]

    loss = dense.sum()
    loss.backward()
    print("Grad shape:", storage.sh_coeffs_flat.grad.shape)

    # Degree für Gaussian 1 und 3 erhöhen
    ids = torch.tensor([1, 2])
    storage.increase_degree(ids, step=1)

    print("After increase:")
    print("Base:", storage.gauss_offsets)
    print("Num:", storage.num_coeffs_per_gauss)
    print("Deg:", storage.sh_degrees)

if __name__ == "__main__":
    test_adaptive_sh()
