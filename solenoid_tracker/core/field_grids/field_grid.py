import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
from ..dataclasses.dataclasses import FieldGridBase
from ..physics import *
# ============================================================
# БЛОК II — FNTFieldGrid
# ============================================================

class FNTFieldGrid(FieldGridBase):
    """
    Неравномерная (tanh-растянутая) сетка магнитного поля одного соленоида.

    Хранит ПОЛОВИННУЮ сетку по z: z ∈ [0, b] (только z ≥ 0 относительно
    центра соленоида). Продолжение на z < 0 выполняется в CUDA-ядре через
    симметрию/антисимметрию:
        Bz(r, -Δz) = +Bz(r, +Δz)   — чётная
        Br(r, -Δz) = -Br(r, +Δz)   — нечётная

    При двух и более одинаковых соленоидах используется ОДНА эта сетка.
    Суперпозиция поля выполняется в CUDA-ядре.

    Parameters
    ----------
    r_grid   : (Nr,)      float — физическая ось r [м], r ∈ [0, r_max]
    z_half   : (Nz_h,)    float — физическая ось z [м], z ∈ [0, b]
    Br_half  : (Nr, Nz_h) float — радиальная компонента [Тл]
    Bz_half  : (Nr, Nz_h) float — осевая компонента    [Тл]
    beta_r   : float            — параметр tanh-растяжки по r
    alpha_z  : float            — параметр tanh-растяжки по z
    """

    def __init__(self, r_grid, z_half, Br_half, Bz_half, beta_r, alpha_z):
        self._r_grid  = np.asarray(r_grid,  dtype=np.float32)
        self._z_half  = np.asarray(z_half,  dtype=np.float32)
        self._Br_half = np.asarray(Br_half, dtype=np.float32)
        self._Bz_half = np.asarray(Bz_half, dtype=np.float32)

        self._beta_r  = float(beta_r)
        self._alpha_z = float(alpha_z)

        Nr, Nz_h = self._Br_half.shape
        assert self._r_grid.shape[0] == Nr,   "r_grid не совпадает с Br_half по r"
        assert self._z_half.shape[0] == Nz_h, "z_half не совпадает с Br_half по z"

        self._Nr   = Nr
        self._Nz_h = Nz_h

        self._r_min_phys = float(self._r_grid[0])    # = 0
        self._r_max_phys = float(self._r_grid[-1])
        self._z_max_half = float(self._z_half[-1])   # = b

        print(f"FNTFieldGrid создана:")
        print(f"  Nr = {Nr},  Nz_half = {Nz_h}")
        print(f"  r  ∈ [{self._r_min_phys:.4f}, {self._r_max_phys:.4f}] м")
        print(f"  z  ∈ [0, {self._z_max_half:.4f}] м  (половинная)")
        print(f"  beta_r  = {self._beta_r},  alpha_z = {self._alpha_z}")
        print(f"  Bz_max  = {self._Bz_half.max()*1e3:.2f} мТл")
        print(f"  Br_max  = {self._Br_half.max()*1e3:.2f} мТл")

    # ── FieldGridBase interface ────────────────────────────────────────────

    @property
    def r_min_phys(self): return self._r_min_phys

    @property
    def r_max_phys(self): return self._r_max_phys

    @property
    def z_min_phys(self): return -self._z_max_half

    @property
    def z_max_phys(self): return  self._z_max_half

    @property
    def Nr(self): return self._Nr

    @property
    def Nz(self): return self._Nz_h

    @property
    def Br_host(self): return self._Br_half

    @property
    def Bz_host(self): return self._Bz_half

    # ── Специфичные свойства FNT ──────────────────────────────────────────

    @property
    def beta_r(self):          return self._beta_r

    @property
    def alpha_z(self):         return self._alpha_z

    @property
    def z_max_half_phys(self): return self._z_max_half

    # ── GPU upload ────────────────────────────────────────────────────────

    def upload_to_gpu(self):
        """Копирует сетку на GPU. Вызывать один раз перед запуском трекера."""
        self.Br_d = cuda.to_device(self._Br_half)
        self.Bz_d = cuda.to_device(self._Bz_half)
        print("FNTFieldGrid: данные скопированы на GPU.")

    # ── Проверка физических свойств поля ─────────────────────────────────

    def check_symmetry(self, rtol=1e-3):
        """
        Проверяет физические свойства половинной сетки без экстраполяции.

        Три независимых теста, каждый работает только в области z ≥ 0:

        1. Br(r=0, z) = 0  — ось симметрии: на оси радиальное поле равно нулю.

        2. Br(r, z=0) = 0  — граница симметрии: Br нечётна по z,
           значит при z=0 она обязана обращаться в ноль.

        3. ∇·B = 0  — закон Гаусса в цилиндрических координатах:
           (1/r)·∂(r·Br)/∂r + ∂Bz/∂z = 0
           Проверяется через конечные разности во внутренних точках.
        """
        ok = True

        # ── Тест 1: Br на оси (первая строка по r) ────────────────────────
        Br_axis     = self._Br_half[0, :]                        # (Nz_h,)
        err_axis    = np.max(np.abs(Br_axis)) / (np.max(np.abs(self._Br_half)) + 1e-30)
        status_axis = "✓" if err_axis < rtol else "✗  ОШИБКА"
        print(f"Тест 1  Br(r=0):   max|Br_axis| / max|Br| = {err_axis:.2e}  {status_axis}")
        if err_axis >= rtol:
            ok = False

        # ── Тест 2: Br при z=0 (первый столбец по z) ──────────────────────
        Br_z0     = self._Br_half[:, 0]                          # (Nr,)
        err_z0    = np.max(np.abs(Br_z0)) / (np.max(np.abs(self._Br_half)) + 1e-30)
        status_z0 = "✓" if err_z0 < rtol else "✗  ОШИБКА"
        print(f"Тест 2  Br(z=0):   max|Br_z0|  / max|Br| = {err_z0:.2e}  {status_z0}")
        if err_z0 >= rtol:
            ok = False

        # ── Тест 3: ∇·B = 0 через конечные разности ──────────────────────
        # Работаем на внутренних узлах (1 .. Nr-2) × (1 .. Nz_h-2),
        # чтобы избежать граничных эффектов.
        r = self._r_grid[1:-1, np.newaxis]           # (Nr-2, 1)
        dr = (self._r_grid[2:]  - self._r_grid[:-2]) / 2.0   # центральная разность
        dz = (self._z_half[2:]  - self._z_half[:-2]) / 2.0

        Br_inner = self._Br_half[1:-1, 1:-1]         # (Nr-2, Nz_h-2)
        dBrr_dr  = (self._Br_half[2:,  1:-1] * self._r_grid[2:,  np.newaxis]
                  - self._Br_half[:-2, 1:-1] * self._r_grid[:-2, np.newaxis]
                   ) / (2.0 * dr[:, np.newaxis] * r)

        dBz_dz   = (self._Bz_half[1:-1, 2:]
                  - self._Bz_half[1:-1, :-2]
                   ) / (2.0 * dz[np.newaxis, :])

        divB     = dBrr_dr + dBz_dz
        divB_rel = np.max(np.abs(divB)) / (np.max(np.abs(self._Bz_half)) + 1e-30)
        status_div = "✓" if divB_rel < rtol else "✗  ОШИБКА"
        print(f"Тест 3  ∇·B=0:     max|∇·B| / max|Bz|    = {divB_rel:.2e}  {status_div}")
        if divB_rel >= rtol:
            ok = False

        print("  Все тесты пройдены ✓" if ok else "  Есть нарушения — проверьте исходные данные.")

    # ── Визуализация ──────────────────────────────────────────────────────

    def plot(self, figsize=(13, 9)):
        """
        2×2 подграфика:
          Строка 1 — поля в физических координатах (r, z) [см]
          Строка 2 — те же поля в индексном пространстве (i, j)
                     наглядно показывает сгущение узлов tanh-сетки
        """
        from matplotlib.colors import SymLogNorm

        r_cm = self._r_grid * 1e2    # физические оси
        z_cm = self._z_half * 1e2

        ir = np.arange(self._Nr)      # индексные оси
        iz = np.arange(self._Nz_h)

        fields = [self._Bz_half * 1e3, self._Br_half * 1e3]
        titles_phys  = ["Bz [мТл]  —  физ. координаты",
                        "Br [мТл]  —  физ. координаты"]
        titles_index = ["Bz [мТл]  —  индексное пространство (i,j)",
                        "Br [мТл]  —  индексное пространство (i,j)"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for col, field in enumerate(fields):

            vmax = np.max(np.abs(field))
            norm = SymLogNorm(linthresh=0.01 * vmax + 1e-10,
                              vmin=-vmax, vmax=vmax)

            # ── Строка 0: физические координаты ──────────────────────────
            Z_phys, R_phys = np.meshgrid(z_cm, r_cm)
            cf0 = axes[0, col].pcolormesh(Z_phys, R_phys, field,
                                          cmap='RdBu_r', norm=norm,
                                          shading='auto')
            plt.colorbar(cf0, ax=axes[0, col])
            axes[0, col].set_xlabel('z [см]')
            axes[0, col].set_ylabel('r [см]')
            axes[0, col].set_title(titles_phys[col])

            # ── Строка 1: индексное пространство ─────────────────────────
            IZ, IR = np.meshgrid(iz, ir)
            cf1 = axes[1, col].pcolormesh(IZ, IR, field,
                                          cmap='RdBu_r', norm=norm,
                                          shading='auto')
            plt.colorbar(cf1, ax=axes[1, col])
            axes[1, col].set_xlabel('индекс j  (ось z)')
            axes[1, col].set_ylabel('индекс i  (ось r)')
            axes[1, col].set_title(titles_index[col])

        plt.suptitle("FNT-сетка: половинное поле (z ≥ 0 от центра соленоида)")
        plt.tight_layout()
        plt.show()