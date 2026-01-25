import torch
import torch.nn as nn

class SHStorage(nn.Module):

    """
        sh_coeffs_flat: (T, 3) array mit den SH Koeffizienten aller gaussians hintereinander
        sh_degrees: (P,) array mit dem SH degree aller gaussians
        gauss_offsets: (P,) array mit den offsets aller gaussians: gauss_offsets[i] = Index in sh_coeffs_flat,
          an denen die SH Koeffizienten von gaussian i anfangen
        num_coeffs_per_gauss:
    """

    def __init__(self, num_gaussians, init_deg=0,
                 max_degree=3, dtype = torch.float32, device=None): # CPU zum Testen ohne VM
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gauss = num_gaussians
        self.num_gaussians = num_gaussians
        self.max_degree = max_degree
        self.device = device
        num_coeff_per_gaussian = (init_deg + 1) ** 2
        # gauss_offsets dim(P,) gibt für jeden gaussian an, an welcher Stelle seine SH coefficients anfangen
        gauss_offsets = torch.arange(
            0, self.num_gauss, device=self.device, dtype=torch.int32
        ) * num_coeff_per_gaussian

        self.register_buffer(
            "gauss_offsets",
            gauss_offsets,
            persistent=True,
        )

        self.register_buffer(
            "sh_degrees",
            torch.full(
                (num_gaussians,),
                init_deg,
                dtype=torch.int32,
                device=device,
            ),
            persistent=True,
        )

        self.register_buffer(
            "num_coeffs_per_gauss",
            torch.full(
                (num_gaussians,),
                num_coeff_per_gaussian,
                dtype=torch.int32,
                device=device,
            ),
            persistent=True,
        )

        total_coeffs = int(num_gaussians * num_coeff_per_gaussian)

        # Trainierbare SH-Koeffs: [T, 3]
        self.sh_coeffs_flat = nn.Parameter(
            torch.zeros((total_coeffs, 3), device=device, dtype=dtype)
        )
        self.num_gaussians = self.num_gauss

    def _segment_indices(self, offsets: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Given per-gaussian offsets and counts, return the flat indices that cover
        all corresponding coefficient segments
        """
        if counts.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)

        counts_long = counts.to(dtype=torch.long, device=self.device)
        total = int(counts_long.sum().item())
        if total == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)

        prefix = torch.cumsum(counts_long, dim=0) - counts_long
        expanded_prefix = torch.repeat_interleave(prefix, counts_long)
        expanded_offsets = torch.repeat_interleave(offsets.to(dtype=torch.long, device=self.device), counts_long)
        entry_idx = torch.arange(total, device=self.device, dtype=torch.long)
        return expanded_offsets + (entry_idx - expanded_prefix)

    def _gather_coeff_blocks(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Returns a concatenated [sum(count_i), 3] tensor with all SH coefficients
        for the provided gaussian indices.
        """
        if ids.numel() == 0:
            return torch.zeros((0, 3), device=self.device, dtype=self.sh_coeffs_flat.dtype)

        counts = self.num_coeffs_per_gauss[ids]
        offsets = self.gauss_offsets[ids]
        idx = self._segment_indices(offsets, counts)
        if idx.numel() == 0:
            return torch.zeros((0, 3), device=self.device, dtype=self.sh_coeffs_flat.dtype)
        return self.sh_coeffs_flat[idx]

    
    
    def _repack_all(
        self,
        new_degrees: torch.Tensor,
    ):
        """
        Repacking aller Gaussians ohne for-Schleife.
        """
        
        new_counts = (new_degrees + 1) ** 2
        new_offsets = torch.cumsum(new_counts, dim=0) - new_counts
        total_new_coeffs = int(new_counts.sum().item())
        new_coeffs_flat = torch.zeros((total_new_coeffs, 3), device=self.device, dtype=self.sh_coeffs_flat.dtype)

        # old indices (positions in the old flat array) for all existing coeffs
        #idx = self._segment_indices(self.gauss_offsets, self.num_coeffs_per_gauss)
        idx = torch.arange(self.sh_coeffs_flat.shape[0], device=self.device)

        # target positions in the new flat array for each old block entry
        new_idx = self._segment_indices(new_offsets, self.num_coeffs_per_gauss)

        # place old coeffs into new flat buffer
        if idx.numel() > 0:
            new_coeffs_flat[new_idx] = self.sh_coeffs_flat[idx]

        # Build explicit mapping old_index -> new_index (length = total_old_coeffs)
        total_old_coeffs = int(self.sh_coeffs_flat.shape[0])
        if total_old_coeffs > 0:
            mapping = torch.empty((total_old_coeffs,), dtype=torch.long, device=self.device)
            # idx contains positions in old flat corresponding to each entry of new_idx
            # so mapping[old_pos] = new_pos
            mapping[idx] = new_idx
        else:
            mapping = torch.empty((0,), dtype=torch.long, device=self.device)

        # install new state
        self.sh_coeffs_flat = nn.Parameter(new_coeffs_flat)
        self.gauss_offsets = new_offsets.to(torch.int32)
        self.num_coeffs_per_gauss = new_counts.to(torch.int32)
        self.sh_degrees = new_degrees.to(torch.int32)
        self.num_gauss = int(new_degrees.shape[0])
        self.num_gaussians = int(new_degrees.shape[0])

        return mapping

    def _extend_storage(self, new_degrees: torch.Tensor, new_coeffs: torch.Tensor):
        """
        Fügt neue gaussians zum Storage hinzu
        
        Parameters:
            new_degrees: Tensor [M], die sh degrees der neuen gaussians
            new_coeffs:  Tensor [sum((deg+1)^2), 3] die SH-Koeffizienten aller neuen gaussians
                        
        """
        if new_degrees.numel() == 0:
            return

        device = self.device
        new_degrees = new_degrees.to(device=device, dtype=torch.int32)
        new_coeffs = new_coeffs.to(device)
        new_counts = (new_degrees + 1) ** 2
        total_new = int(new_counts.sum().item())

        if new_coeffs.shape[0] != total_new:
            raise ValueError("Mismatch between provided coefficients and degrees.")

        base_offset = self.sh_coeffs_flat.shape[0]
        appended_offsets = base_offset + torch.cumsum(new_counts, dim=0) - new_counts

        self.sh_coeffs_flat = nn.Parameter(torch.cat([self.sh_coeffs_flat, new_coeffs], dim=0))
        self.gauss_offsets = torch.cat([self.gauss_offsets, appended_offsets.to(torch.int32)], dim=0)
        self.num_coeffs_per_gauss = torch.cat([self.num_coeffs_per_gauss, new_counts.to(torch.int32)], dim=0)
        self.sh_degrees = torch.cat([self.sh_degrees, new_degrees], dim=0)
        self.num_gauss += new_degrees.shape[0]
        self.num_gaussians = self.num_gauss

    
    def duplicate_sh_of_gaussians(self, clone_ids: torch.Tensor):

        device = self.device
        if clone_ids.dtype == torch.bool:
            clone_ids = torch.nonzero(clone_ids, as_tuple=False).squeeze(-1)
        clone_ids = clone_ids.to(device).long()
        if clone_ids.numel() == 0:
            return None

        # Degrees der neuen Gaussians
        new_degrees = self.sh_degrees[clone_ids].clone()

        # Sammle alle Koeffizienten der zu klonenden/splittenden Gaussians
        new_coeffs = self._gather_coeff_blocks(clone_ids)

        # Storage erweitern
        self._extend_storage(new_degrees=new_degrees, new_coeffs=new_coeffs)
        return new_coeffs

    def prune_gaussians(self, keep_mask: torch.Tensor, new_param: torch.nn.Parameter = None):
        """
        Removes Gaussians that are marked False in keep_mask and compacts storage.
        """
        keep_mask = keep_mask.to(self.device)
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.bool()

        keep_ids = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        new_N = int(keep_ids.shape[0])
        if new_N == self.num_gauss:
            return

        new_counts = self.num_coeffs_per_gauss[keep_ids]
        new_offsets = torch.cumsum(new_counts, dim=0) - new_counts

        if new_param is not None:
            self.sh_coeffs_flat = new_param
        else:
            gathered = self._gather_coeff_blocks(keep_ids)
            self.sh_coeffs_flat = nn.Parameter(gathered)
        self.gauss_offsets = new_offsets.to(torch.int32)
        self.num_coeffs_per_gauss = new_counts
        self.sh_degrees = self.sh_degrees[keep_ids]
        self.num_gauss = new_N
        self.num_gaussians = new_N



    
    def set_base_color(self, colors):
        self.sh_coeffs_flat[self.gauss_offsets] = colors


    def increase_degree(
        self,
        gauss_ids: torch.Tensor,
        step: int = 1
    ):
        """
        Erhöht den SH-degree ausgewählter Gaussians.

        gauss_ids: Gaussian-Indizes.
        step:      wie viele L-Levels höher (typisch 1).
        hard_max_degree: optionaler Cap, damit man nicht über L_max hinausgeht.
        """
        if gauss_ids.numel() == 0:
            return None

        gauss_ids = gauss_ids.to(self.device).long()

        current_deg = self.sh_degrees.clone()
        target_deg = current_deg.clone()

        target_deg[gauss_ids] += step

        

        # auf hard_max clampen
        target_deg = torch.clamp(target_deg, max=int(self.max_degree))

        # nur wirklich erhöhte Gaussians betrachten
        if torch.equal(target_deg, current_deg):
            return None  # nichts zu tun

        return self._repack_all(target_deg)

    
    def initialize_sh_from_color(self, colors: torch.Tensor):
        """
        Setzt die SH-DC-Koeffizienten (Grad 0) für alle Gaussians.
        
        colors: Tensor [N, 3], die per-Gaussian RGB-Werte aus create_from_pcd.
        """
        device = self.device
        colors = colors.to(device)

        N = self.num_gaussians
        assert colors.shape[0] == N, "Color length must match number of Gaussians"

        offsets = self.gauss_offsets.long()
        self.sh_coeffs_flat.data[offsets] = colors



    def increase_sh_degree_without_repacking_everything(self, idx): #Doesn't work

        # K: number of gaussians to be increased
        # G: number of gaussians that need new space

        #gaussians that get an sh degree increase
        idx = idx[self.sh_degrees[idx] < self.max_degree] # dim K

        old_degrees = self.sh_degrees[idx]
        new_degrees = old_degrees + 1
        num_coeffs_old = (old_degrees + 1) ** 2
        num_coeffs_new = (new_degrees + 1) ** 2 # all of dim K

        #check which gaussians need space at the end of the flat coeff array. Achtung! Das macht so keinen Sinn mit dem idx + 1!!!
        needs_relayout = (idx == self.num_gauss - 1) or (self.gauss_offsets[idx] + num_coeffs_new > self.gauss_offsets[idx + 1]) # dim K

        #starting offsets of the old and new coefficients in the flat coeff array
        new_offsets_in_added_part = torch.cumsum(num_coeffs_new[needs_relayout], dim=0) - num_coeffs_new[0] # dim G
        new_coeffs_array = torch.zeros((num_coeffs_new[needs_relayout].sum().item(), 3), device=self.coeffs_flat.device, dtype=self.coeffs_flat.dtype)
        #copy old coefficients of the newly allocated gaussians
        new_coeffs_array[new_offsets_in_added_part:(new_offsets_in_added_part + num_coeffs_old[needs_relayout])] = self.coeffs_flat[self.gauss_offsets[idx[needs_relayout]]:(self.gauss_offsets[idx[needs_relayout]] + num_coeffs_old[needs_relayout])]
        #new offsets for reallocated gaussians (already for the larger merged array)
        self.gauss_offsets[idx[needs_relayout]] = self.total_coeffs + new_offsets_in_added_part
        #set newly added coefficients for not reallocated gaussians to zero
        self.coeffs_flat[(self.gauss_offsets[idx[~needs_relayout]] + num_coeffs_old[idx[~needs_relayout]]):(self.gauss_offsets[idx[~needs_relayout]] + num_coeffs_new[idx[~needs_relayout]])] = 0.0

        self.sh_degrees[idx] += 1


    def build_dense_sh(self, max_degree = None) -> torch.Tensor:
            """
            Baut einen dichten SH-Tensor [N, 3, (max_degree+1)^2] für den Rasterizer.

            max_degree:
                - Wenn None → self.max_sh_degree
                - Wichtig: max_degree >= max(self.sh_degree)
            """
            if max_degree is None:
                max_degree = int(self.max_degree)

            N = self.num_gauss
            max_coeffs = (max_degree + 1) ** 2

            device = self.device

            dense = torch.zeros(
                (N, 3, max_coeffs), device=device, dtype=self.sh_coeffs_flat.dtype
            )

            flat = self.sh_coeffs_flat  # [T, 3]
            base = self.gauss_offsets    # [N]
            count = self.num_coeffs_per_gauss   # [N]

            # einfache, korrekte Referenzimplementierung (Python-Schleife)
            # später kann man das vektorisieren
            for i in range(N):
                c = int(count[i].item())
                if c == 0:
                    continue
                start = int(base[i].item())
                coeffs_i = flat[start : start + c]  # [c, 3]
                # -> [3, c]
                dense[i, :, :c] = coeffs_i.transpose(0, 1)

            return dense

    def serialize(self):
        return {
            "sh_coeffs_flat": self.sh_coeffs_flat.detach().clone(),
            "gauss_offsets": self.gauss_offsets.detach().clone(),
            "num_coeffs_per_gauss": self.num_coeffs_per_gauss.detach().clone(),
            "sh_degrees": self.sh_degrees.detach().clone(),
            "max_degree": self.max_degree,
        }

    @classmethod
    def from_serialized(cls, data, device="cuda"):
        coeffs = data["sh_coeffs_flat"].to(device)
        offsets = data["gauss_offsets"].to(device).to(torch.int32)
        counts = data["num_coeffs_per_gauss"].to(device).to(torch.int32)
        degrees = data["sh_degrees"].to(device).to(torch.int32)
        num_gauss = int(offsets.shape[0])
        storage = cls(
            num_gaussians=num_gauss,
            init_deg=0,
            max_degree=int(data.get("max_degree", 3)),
            device=device,
        )
        storage.gauss_offsets = offsets
        storage.num_coeffs_per_gauss = counts
        storage.sh_degrees = degrees
        storage.sh_coeffs_flat = nn.Parameter(coeffs)
        storage.num_gauss = num_gauss
        storage.num_gaussians = num_gauss
        return storage
