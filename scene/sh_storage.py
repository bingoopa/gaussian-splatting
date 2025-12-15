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
                 max_degree=3, dtype = torch.float32, device="cpu"): # CPU zum Testen ohne VM
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gauss = num_gaussians
        self.max_degree = max_degree
        self.device = device
        num_coeff_per_gaussian = (init_deg + 1) ** 2
        # gauss_offsets dim(P,) gibt für jeden gaussian an, an welcher Stelle seine SH coefficients anfangen
        gauss_offsets = torch.arange(0, self.num_gauss, device = self.device) * num_coeff_per_gaussian

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
        

    def _repack_all(
        self,
        new_degrees: torch.Tensor,
    ):
        """
        Vollständiges Repacking aller Gaussians in einen neuen flachen Puffer.

        new_degrees: [N] int32, neue SH-Grade (inkl. nicht veränderter).
        (Einfach, aber O(T) – dafür extrem robust und gut nachvollziehbar.)
        """

        device = self.device
        N = self.num_gauss

        new_num_coeff = (new_degrees + 1) ** 2  # dim N

        # alter Zustand merken
        old_flat = self.sh_coeffs_flat.data.clone()
        old_offsets = self.gauss_offsets.clone()
        old_num_coeffs_per_gauss = self.num_coeffs_per_gauss.clone()

        # neuen Puffer anlegen
        total_new_coeffs = int(new_num_coeff.sum().item())
        new_flat = torch.zeros(
            (total_new_coeffs, 3),
            device=device,
            dtype=self.sh_coeffs_flat.dtype,
        )

        new_offsets = torch.empty_like(self.gauss_offsets)

        # Alle Gaussians einmal in den neuen Puffer kopieren
        cursor = 0
        for i in range(N):
            old_start = int(old_offsets[i].item())
            old_count = int(old_num_coeffs_per_gauss[i].item())
            new_count = int(new_num_coeff[i].item())

            new_offsets[i] = cursor

            if old_count > 0:
                new_flat[cursor : cursor + old_count] = old_flat[
                    old_start : old_start + old_count
                ]
            # zusätzliche Koeffs (falls degree erhöht) bleiben 0-init
            cursor += new_count

        # in Module-State übernehmen
        self.sh_coeffs_flat = nn.Parameter(new_flat)
        self.gauss_offsets = new_offsets
        self.num_coeffs_per_gauss = new_num_coeff
        self.sh_degrees = new_degrees

    def _extend_storage(self, new_degrees: torch.Tensor, new_coeffs: torch.Tensor):
        """
        Fügt neue Gaussians zum Storage hinzu.
        
        Parameters:
            new_degrees: Tensor [M], dtype int32 – die Degrees der neuen Gaussians
            new_coeffs:  Tensor [sum((deg+1)^2), 3] – die SH-Koeffizienten aller neuen Gaussians
                        konkatenierte Segmente in der Reihenfolge der neuen Gaussians.
        """
        device = self.device

        # Alte Werte
        old_N = self.num_gauss
        old_deg = self.sh_degrees.clone()
        old_flat = self.sh_coeffs_flat.data.clone()
        old_offsets = self.gauss_offsets.clone()
        old_num = self.num_coeffs_per_gauss.clone()

        # Neue Werte
        new_N = old_N + new_degrees.shape[0]
        all_degrees = torch.cat([old_deg, new_degrees.to(device)], dim=0)
        all_num = (all_degrees + 1)**2

        # Neuen flachen Speicher allozieren
        total_new_coeffs = int(all_num.sum().item())
        new_flat = torch.zeros(
            (total_new_coeffs, 3), device=device, dtype=self.sh_coeffs_flat.dtype
        )
        new_offsets = torch.empty(new_N, dtype=torch.int32, device=device)

        # --- Kopieren der alten Gaussians ---
        cursor = 0
        for i in range(old_N):
            count = int(old_num[i].item())
            start = int(old_offsets[i].item())
            new_offsets[i] = cursor

            if count > 0:
                new_flat[cursor : cursor + count] = old_flat[start : start + count]

            cursor += count

        # --- Kopieren der neuen Gaussians ---
        new_cursor = 0
        for j in range(new_degrees.shape[0]):
            deg_j = int(new_degrees[j].item())
            count_j = (deg_j + 1)**2

            new_offsets[old_N + j] = cursor
            new_flat[cursor : cursor + count_j] = new_coeffs[new_cursor : new_cursor + count_j]

            cursor += count_j
            new_cursor += count_j

        # Speicher aktualisieren
        self.sh_coeffs_flat = nn.Parameter(new_flat)
        self.gauss_offsets = new_offsets
        self.num_coeffs_per_gauss = all_num
        self.sh_degrees = all_degrees
        self.num_gauss = new_N

    
    def duplicate_sh_of_gaussians(self, clone_ids: torch.Tensor):

        device = self.device
        clone_ids = clone_ids.to(device).long()

        # Degrees der neuen Gaussians
        new_degrees = self.sh_degrees[clone_ids].clone()

        # Sammle alle Koeffizienten der zu klonenden/splittenden gaussians in einem flachen Block
        coeff_blocks = []
        for id in clone_ids:
            id = int(id.item())
            start = int(self.gauss_offsets[id].item())
            count = int(self.num_coeffs_per_gauss[id].item())
            coeff_blocks.append(self.sh_coeffs_flat[start : start + count])

        if len(coeff_blocks) > 0:
            new_coeffs = torch.cat(coeff_blocks, dim=0)
        else:
            new_coeffs = torch.zeros((0, 3), device=device, dtype=self.sh_coeffs_flat.dtype)

        # Storage erweitern
        self._extend_storage(new_degrees=new_degrees, new_coeffs=new_coeffs)



    
    def set_base_color(self, colors):
        self.coeffs_flat[self.gauss_offsets] = colors


    def increase_degree(
        self,
        gauss_ids: torch.Tensor,
        step: int = 1
    ):
        """
        Erhöht den SH-degree ausgewählter Gaussians.

        gauss_ids: 1D-Long/Int-Tensor mit Gaussian-Indizes.
        step:      wie viele L-Levels höher (typisch 1).
        hard_max_degree: optionaler Cap, damit man nicht über L_max hinausgeht.
        """
        if gauss_ids.numel() == 0:
            return

        gauss_ids = gauss_ids.to(self.device).long()

        current_deg = self.sh_degrees.clone()
        target_deg = current_deg.clone()

        target_deg[gauss_ids] += step

        

        # auf hard_max clampen
        target_deg = torch.clamp(target_deg, max=int(self.max_degree))

        # nur wirklich erhöhte Gaussians betrachten
        if torch.equal(target_deg, current_deg):
            return  # nichts zu tun

        self._repack_all(target_deg)

    
    def initialize_sh_from_color(self, colors: torch.Tensor):
        """
        Setzt die SH-DC-Koeffizienten (Grad 0) für alle Gaussians.
        
        colors: Tensor [N, 3], die per-Gaussian RGB-Werte aus create_from_pcd.
        """
        device = self.device
        colors = colors.to(device)

        N = self.num_gaussians
        assert colors.shape[0] == N, "Color length must match number of Gaussians"

        offsets = self.gauss_offsets  
        for i in range(N):
            idx = int(offsets[i].item())
            self.sh_coeffs_flat.data[idx] = colors[i]



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
                max_degree = int(self.max_sh_degree)

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
