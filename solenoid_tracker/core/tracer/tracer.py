import time, numpy as np
from functools import wraps
from numba import cuda
from tqdm.notebook import tqdm
from ..kernels.kernels import beam_step_kernel_fnt
from ..dataclasses.dataclasses import BeamPhysicalParams, BeamNumericalParams
from ..scaling.scaling import ScalingTransform, BeamDiagnostics, DimensionlessGridAdapter
from ..physics import *
# ── Декоратор замера времени ──────────────────────────────────────────────────

def timed_run(func):
    """Замеряет wall-time и GPU-time выполнения метода run()."""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_wall = time.perf_counter()
        result     = func(*args, **kwargs)
        wall_time  = time.perf_counter() - start_wall
        print(f"\n── Время выполнения: {wall_time:.3f} с ──")
        return result
    return wrapper


# ── Трекировщик ───────────────────────────────────────────────────────────────

class BeamTracerCUDA:
    """
    Трекировщик пучка на GPU (метод Бориса, релятивистский).

    Принимает начальные условия в ФИЗИЧЕСКИХ единицах:
        x0_phys : (N, 3) [м]       координаты
        p0_phys : (N, 3) [кг·м/с]  импульсы

    Parameters
    ----------
    adapted_grid : DimensionlessGridAdapter
    phys         : BeamPhysicalParams
    num          : BeamNumericalParams
    scaling      : ScalingTransform
    diagnostics  : BeamDiagnostics | None
    """

    def __init__(self, adapted_grid, phys, num, scaling, diagnostics=None):
        self.grid        = adapted_grid
        self.phys        = phys
        self.num         = num
        self.scaling     = scaling
        self.diagnostics = diagnostics

    @timed_run
    def run(self, x0_phys: np.ndarray, p0_phys: np.ndarray,
            *, store_history: bool = False, stride: int = 1):
        sc      = self.scaling
        fg      = self.grid
        N       = x0_phys.shape[0]
        n_steps = self.num.n_steps

        if stride < 1:
            raise ValueError(f"stride должен быть ≥ 1, получено {stride}")

        # ── Безразмерный шаг — вычисляется из физических t_start, t_end ──────
        dt = (self.num.t_end - self.num.t_start) / (n_steps * sc.T0)

        # ── Обезразмеривание ─────────────────────────────────────────────────
        x = (x0_phys / sc.L0).astype(np.float32)
        p = (p0_phys / sc.P0).astype(np.float32)

        # ── GPU-буферы ───────────────────────────────────────────────────────
        x_d  = cuda.to_device(x)
        p_d  = cuda.to_device(p)
        Bx_d = cuda.device_array(N, dtype=np.float32)
        By_d = cuda.device_array(N, dtype=np.float32)
        Bz_d = cuda.device_array(N, dtype=np.float32)

        threads = self.num.threads_per_block
        blocks  = (N + threads - 1) // threads

        # ── Инициализация истории ─────────────────────────────────────────────
        if store_history:
            n_saved = n_steps // stride + 1
            x_hist  = np.zeros((n_saved, N, 3), dtype=np.float32)
            p_hist  = np.zeros((n_saved, N, 3), dtype=np.float32)
            t_hist  = np.zeros(n_saved,          dtype=np.float64)
            x_hist[0] = x
            p_hist[0] = p
            hist_idx  = 1
            print(f"История: {n_saved} кадров  "
                  f"(каждый {stride}-й шаг из {n_steps})  "
                  f"≈ {2 * n_saved * N * 3 * 4 / 1e6:.1f} МБ RAM")

        # ── Основной цикл ─────────────────────────────────────────────────────
        for step in tqdm(range(1, n_steps + 1),
                        desc="Трекировка", unit="шаг", leave=True):

            beam_step_kernel_fnt[blocks, threads](
                x_d, p_d,
                fg.Br_d, fg.Bz_d,
                fg.sol_centers_d,
                fg.r_min, fg.r_max, fg.z_max_half,
                fg.beta_r, fg.alpha_z,
                fg.Nr, fg.Nz,
                dt,                         # ← вычисленный выше
                Bx_d, By_d, Bz_d,
            )

            save_now = store_history and (step % stride == 0)
            need_x_p = save_now or (self.diagnostics is not None)

            if need_x_p:
                x_cpu = x_d.copy_to_host()
                p_cpu = p_d.copy_to_host()

            if save_now:
                x_hist[hist_idx] = x_cpu
                p_hist[hist_idx] = p_cpu
                t_hist[hist_idx] = self.num.t_start + step * dt * sc.T0
                hist_idx += 1

            if self.diagnostics is not None:
                B_cpu = np.empty((N, 3), dtype=np.float32)
                B_cpu[:, 0] = Bx_d.copy_to_host()
                B_cpu[:, 1] = By_d.copy_to_host()
                B_cpu[:, 2] = Bz_d.copy_to_host()
                self.diagnostics.record(step=step, x=x_cpu, p=p_cpu,
                                        B_dimless=B_cpu, scaling=sc)

        # ── Финал → физические единицы ───────────────────────────────────────
        x_f = x_d.copy_to_host() * sc.L0
        p_f = p_d.copy_to_host() * sc.P0

        if store_history:
            return x_hist * sc.L0, p_hist * sc.P0, t_hist
        return x_f, p_f