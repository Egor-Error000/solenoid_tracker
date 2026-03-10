import numpy as np
from numba import cuda
from ..dataclasses.dataclasses import DimensionlessScaling, BeamPhysicalParams, SolenoidParams
from ..field_grids.field_grid import FNTFieldGrid
from ..physics import *
# ============================================================
# БЛОК IV — Масштабирование, диагностика, адаптер сетки
# ============================================================


class ScalingTransform:
    """
    Масштабирование между физическими и безразмерными переменными.

    Масштабы выбираются из начальных параметров частицы:

        P0 = m·c·√(γ₀² - 1)      — начальный импульс [кг·м/с]
        T0 = L0 / c               — характерное время [с]
        B0 = P0 / (|q|·L0)       — характерное поле [Тл]

    Безразмерные переменные:
        x̃ = x / L0,   p̃ = p / P0,   t̃ = t / T0,   B̃ = B / B0

    В безразмерных переменных уравнения движения:
        γ     = √(1 + |p̃|²)
        ṽ     = p̃ / γ
        dp̃/dt̃ = ṽ × B̃
        dx̃/dt̃ = ṽ
    """

    def __init__(self, scaling: 'DimensionlessScaling', phys: 'BeamPhysicalParams'):
        self.L0 = scaling.length      # [м]
        self.T0 = scaling.time        # [с]
        self.P0 = scaling.momentum    # [кг·м/с]
        self.B0 = scaling.field       # [Тл]

        self.m = phys.mass            # [кг]
        self.q = phys.charge          # [Кл]
        self.c = phys.c               # [м/с]

        # Коэффициент для пересчёта безразмерного p̃ → физический γ:
        #   γ = √(1 + |p̃·P0/(mc)|²) = √(1 + |p̃|²·α²)
        #   α = P0 / (m·c) = √(γ₀² − 1)
        self._alpha2 = (self.P0 / (self.m * self.c)) ** 2

    # ── физические → безразмерные ─────────────────────────────────────────

    def to_dimless_x(self, x): return x / self.L0
    def to_dimless_p(self, p): return p / self.P0
    def to_dimless_t(self, t): return t / self.T0

    # ── безразмерные → физические ─────────────────────────────────────────

    def from_dimless_x(self, x): return x * self.L0
    def from_dimless_p(self, p): return p * self.P0
    def from_dimless_t(self, t): return t * self.T0

    # ── релятивистские вспомогательные ────────────────────────────────────

    def gamma_phys(self, p_dimless: np.ndarray) -> np.ndarray:
        """
        Физический γ-фактор из безразмерного импульса p̃.

            γ = √(1 + |p̃·P0/(mc)|²) = √(1 + |p̃|²·α²)

        При p̃=1 (начальный импульс) возвращает γ₀ — проверяемо.
        """
        return np.sqrt(1.0 + np.sum(p_dimless ** 2, axis=-1) * self._alpha2)

    def velocity(self, p_dimless: np.ndarray) -> np.ndarray:
        """ṽ = p̃ / γ̃  где γ̃ = √(1 + |p̃|²)  — безразмерная скорость."""
        gamma_tilde = np.sqrt(1.0 + np.sum(p_dimless ** 2, axis=-1))
        return p_dimless / gamma_tilde[..., None]

    def energy_to_momentum(self, Ek: float) -> float:
        """
        Кинетическая энергия → безразмерный импульс p̃ = p / P0.

            γ  = 1 + Eₖ / (mc²)
            p  = mc · √(γ² - 1)   [кг·м/с]
            p̃  = p / P0

        Parameters
        ----------
        Ek : float  кинетическая энергия [Дж]

        Returns
        -------
        p_dimless : float  (используется для инициализации p0)
        """
        gamma = 1.0 + Ek / (self.m * self.c ** 2)
        p_si  = self.m * self.c * np.sqrt(gamma ** 2 - 1.0)
        return p_si / self.P0

    # ── фабричный метод ───────────────────────────────────────────────────

    @classmethod
    def from_physical_problem(cls, phys: 'BeamPhysicalParams', L0: float):
        """
        Создать ScalingTransform из физических параметров частицы.

            dp/dt = q·v×B  →  dp̃/dt̃ = ṽ × B̃
            ⟹  B0 = P0 / (|q|·L0)

        Parameters
        ----------
        phys : BeamPhysicalParams
        L0   : float  характерная длина [м]
        """
        gamma0 = 1.0 + phys.kinetic_energy / (phys.mass * phys.c ** 2)
        P0     = phys.mass * phys.c * np.sqrt(gamma0 ** 2 - 1.0)
        T0     = L0 / phys.c
        B0     = P0 / (abs(phys.charge) * L0)

        scaling = DimensionlessScaling(
            length   = L0,
            time     = T0,
            momentum = P0,
            field    = B0,
        )
        obj = cls(scaling=scaling, phys=phys)

        print("ScalingTransform создан:")
        print(f"  γ₀   = {gamma0:.6f}")
        print(f"  L0   = {L0:.3e} м")
        print(f"  P0   = {P0:.3e} кг·м/с")
        print(f"  T0   = {T0:.3e} с  ({T0*1e9:.4f} нс)")
        print(f"  B0   = {B0:.3e} Тл")

        return obj


# ─────────────────────────────────────────────────────────────────────────────


class BeamDiagnostics:
    """
    Усреднённые характеристики пучка, записываемые на каждом шаге.

    record() принимает безразмерные x̃, p̃, B̃.
    Физический γ восстанавливается через scaling._alpha2.
    """

    def __init__(self):
        self.steps         = []
        self.mean_gamma    = []   # физический γ
        self.mean_p        = []   # [кг·м/с]
        self.mean_B        = []   # [Тл]
        self.mean_F        = []   # безразм.  |ṽ × B̃|
        self.mean_z        = []   # [м]
        self.rms_r         = []   # [м]
        self.frac_in_field = []

    def record(self, step: int, x: np.ndarray, p: np.ndarray,
               B_dimless: np.ndarray, scaling: 'ScalingTransform'):
        """
        Parameters
        ----------
        step      : int
        x         : (N, 3) float — безразмерные координаты x̃
        p         : (N, 3) float — безразмерные импульсы   p̃
        B_dimless : (N, 3) float — безразмерное поле       B̃
        scaling   : ScalingTransform
        """
        p2 = np.sum(p ** 2, axis=1)

        # ── Физический γ: sqrt(1 + |p̃|²·α²),  α² = (P0/mc)² ──────────────
        gamma = np.sqrt(1.0 + p2 * scaling._alpha2)

        # ── Безразмерная скорость ṽ = p̃ / γ̃,  γ̃ = sqrt(1 + |p̃|²) ─────────
        gamma_tilde = np.sqrt(1.0 + p2)
        v     = p / gamma_tilde[:, None]

        B_abs = np.linalg.norm(B_dimless, axis=1)
        F_abs = np.linalg.norm(np.cross(v, B_dimless), axis=1)

        x_phys = scaling.from_dimless_x(x)
        r = np.sqrt(x_phys[:, 0] ** 2 + x_phys[:, 1] ** 2)

        self.steps.append(step)
        self.mean_gamma.append(float(gamma.mean()))
        self.mean_p.append(float(np.sqrt(p2).mean() * scaling.P0))
        self.mean_B.append(float(B_abs.mean() * scaling.B0))
        self.mean_F.append(float(F_abs.mean()))
        self.mean_z.append(float(x_phys[:, 2].mean()))
        self.rms_r.append(float(np.sqrt(np.mean(r ** 2))))
        self.frac_in_field.append(float(np.mean(B_abs > 0.0)))


# ─────────────────────────────────────────────────────────────────────────────


class DimensionlessGridAdapter:
    """
    Переводит FNTFieldGrid (или SimpleFieldGrid) в безразмерные единицы
    и загружает данные на GPU.
    """

    def __init__(self, field_grid, scaling: ScalingTransform,
                 solenoids=None):
        L0 = scaling.L0
        B0 = scaling.B0

        self.is_fnt = isinstance(field_grid, FNTFieldGrid)
        self.Nr     = field_grid.Nr
        self.Nz     = field_grid.Nz

        Br_norm = (field_grid.Br_host / B0).astype(np.float32)
        Bz_norm = (field_grid.Bz_host / B0).astype(np.float32)
        self.Br_d = cuda.to_device(Br_norm)
        self.Bz_d = cuda.to_device(Bz_norm)

        if self.is_fnt:
            fg = field_grid

            self.r_min      = np.float32(fg.r_min_phys      / L0)
            self.r_max      = np.float32(fg.r_max_phys      / L0)
            self.z_max_half = np.float32(fg.z_max_half_phys / L0)
            self.beta_r     = np.float32(fg.beta_r)
            self.alpha_z    = np.float32(fg.alpha_z)

            if solenoids is None:
                raise ValueError("Для FNTFieldGrid необходимо передать solenoids=...")
            z_centers = np.array([sol.center[2] / L0 for sol in solenoids],
                                 dtype=np.float32)
            self.sol_centers_d = cuda.to_device(z_centers)
            self.n_solenoids   = len(solenoids)

            self.z_min = np.float32(-self.z_max_half)
            self.z_max = np.float32( self.z_max_half)

            print("DimensionlessGridAdapter (FNT):")
            print(f"  r̃ ∈ [{self.r_min:.3f}, {self.r_max:.3f}]")
            print(f"  z̃_half ∈ [0, {self.z_max_half:.3f}]")
            print(f"  B0 = {B0:.3e} Тл")
            print(f"  Соленоиды ẑ_c = {z_centers}")

        else:
            fg = field_grid

            self.r_min = np.float32(fg.r_min_phys / L0)
            self.r_max = np.float32(fg.r_max_phys / L0)
            self.z_min = np.float32(fg.z_min_phys / L0)
            self.z_max = np.float32(fg.z_max_phys / L0)

            self.beta_r        = None
            self.alpha_z       = None
            self.z_max_half    = None
            self.sol_centers_d = None
            self.n_solenoids   = 0

            print("DimensionlessGridAdapter (Simple):")
            print(f"  r̃ ∈ [{self.r_min:.3f}, {self.r_max:.3f}]")
            print(f"  z̃ ∈ [{self.z_min:.3f}, {self.z_max:.3f}]")