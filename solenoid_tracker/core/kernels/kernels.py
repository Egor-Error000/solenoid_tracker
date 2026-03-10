
import math
from numba import cuda
from ..physics import *
# ============================================================
# БЛОК V — CUDA device-функции и трекировщик
# ============================================================

from tqdm.notebook import tqdm


# ── Вспомогательные device-функции ───────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _phys_to_index(coord, c_min, c_max, beta, N):
    """Физическая координата → дробный индекс на tanh-сетке."""
    if coord <= c_min:
        return 0.0
    if coord >= c_max:
        return float(N - 1)
    s   = (coord - c_min) / (c_max - c_min)
    arg = s * math.tanh(beta)
    if arg >=  1.0: arg =  1.0 - 1e-6
    if arg <= -1.0: arg = -1.0 + 1e-6
    u = 0.5 * math.log((1.0 + arg) / (1.0 - arg)) / beta
    return u * float(N - 1)


@cuda.jit(device=True, inline=True)
def _catmull_rom_weights(t):
    """Веса кубического сплайна Catmull-Rom для дробного смещения t ∈ [0,1]."""
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5*t3 + 1.0*t2 - 0.5*t
    w1 =  1.5*t3 - 2.5*t2         + 1.0
    w2 = -1.5*t3 + 2.0*t2 + 0.5*t
    w3 =  0.5*t3 - 0.5*t2
    return w0, w1, w2, w3


@cuda.jit(device=True, inline=True)
def _bicubic_interp(field, fr, fz, Nr, Nz):
    """
    Бикубическая интерполяция Catmull-Rom на сетке field[Nr, Nz].
    Вблизи границ автоматически деградирует до билинейной.
    """
    ir = int(fr)
    iz = int(fz)

    # ── Граничная зона: билинейная интерполяция ───────────────────────────
    if ir < 1 or ir >= Nr - 2 or iz < 1 or iz >= Nz - 2:
        ir_c = min(max(ir, 0), Nr - 2)
        iz_c = min(max(iz, 0), Nz - 2)
        tr = min(max(fr - float(ir_c), 0.0), 1.0)
        tz = min(max(fz - float(iz_c), 0.0), 1.0)
        return (field[ir_c,     iz_c    ] * (1.0 - tr) * (1.0 - tz)
              + field[ir_c + 1, iz_c    ] *        tr  * (1.0 - tz)
              + field[ir_c,     iz_c + 1] * (1.0 - tr) *        tz
              + field[ir_c + 1, iz_c + 1] *        tr  *        tz)

    # ── Внутренняя область: Catmull-Rom ───────────────────────────────────
    tr = fr - float(ir)
    tz = fz - float(iz)
    wr0, wr1, wr2, wr3 = _catmull_rom_weights(tr)
    wz0, wz1, wz2, wz3 = _catmull_rom_weights(tz)

    result = 0.0
    for di, wr in enumerate((wr0, wr1, wr2, wr3)):
        ri  = ir - 1 + di
        row = (field[ri, iz - 1] * wz0
             + field[ri, iz    ] * wz1
             + field[ri, iz + 1] * wz2
             + field[ri, iz + 2] * wz3)
        result += wr * row
    return result


@cuda.jit(device=True, inline=True)
def _field_one_solenoid(Br_grid, Bz_grid,
                        r_t, dz,
                        r_min, r_max, z_max_half,
                        beta_r, alpha_z,
                        Nr, Nz_half):
    """
    Поле одного соленоида в точке (r_t, dz) в безразмерных единицах.
    Симметрия учитывается аналитически: Bz чётна, Br нечётна по dz.

    Returns
    -------
    Br_cyl, Bz_cyl : float32
    """
    # На оси: Br = 0 по определению
    if r_t <= 0.0:
        fz = _phys_to_index(math.fabs(dz), 0.0, z_max_half, alpha_z, Nz_half)
        return 0.0, _bicubic_interp(Bz_grid, 0.0, fz, Nr, Nz_half)

    # За пределами сетки: поле = 0
    if r_t >= r_max or math.fabs(dz) >= z_max_half:
        return 0.0, 0.0

    sign_Br = 1.0 if dz >= 0.0 else -1.0
    fr = _phys_to_index(r_t,           r_min, r_max,      beta_r,  Nr)
    fz = _phys_to_index(math.fabs(dz), 0.0,   z_max_half, alpha_z, Nz_half)

    return (_bicubic_interp(Br_grid, fr, fz, Nr, Nz_half) * sign_Br,
            _bicubic_interp(Bz_grid, fr, fz, Nr, Nz_half))


# ── CUDA-ядро: один шаг методом Бориса ───────────────────────────────────────
#
#  Алгоритм Бориса (релятивистский):
#      γ⁻  = √(1 + |p̃|²)
#      t   = B̃·(dt̃/2)/γ⁻       ← вектор поворота
#      p'  = p̃ + p̃×t             ← первичный поворот
#      s   = 2t/(1+|t|²)          ← нормированный вектор
#      p̃⁺  = p̃ + p'×s             ← финальный поворот  →  |p̃⁺|=|p̃| точно
#      x̃⁺  = x̃ + (p̃⁺/γ⁺)·dt̃     ← шаг координаты

@cuda.jit
def beam_step_kernel_fnt(
    x,            # (N, 3) float32  [in/out]  безразмерные координаты
    p,            # (N, 3) float32  [in/out]  безразмерные импульсы
    Br_grid,      # (Nr, Nz_half) float32
    Bz_grid,      # (Nr, Nz_half) float32
    sol_centers,  # (n_sols,) float32         безразм. центры соленоидов
    r_min, r_max, z_max_half,
    beta_r, alpha_z,
    Nr, Nz_half,
    dt,
    Bx_out, By_out, Bz_out,  # (N,) float32  [out]  поле для диагностики
):
    i = cuda.grid(1)
    if i >= x.shape[0]:
        return

    x_t = x[i, 0];  y_t = x[i, 1];  z_t = x[i, 2]
    r_t = math.sqrt(x_t*x_t + y_t*y_t)

    # ── Суперпозиция поля всех соленоидов ────────────────────────────────
    Br_cyl = 0.0;  Bz_cyl = 0.0
    for k in range(sol_centers.shape[0]):
        Br_k, Bz_k = _field_one_solenoid(
            Br_grid, Bz_grid,
            r_t, z_t - sol_centers[k],
            r_min, r_max, z_max_half,
            beta_r, alpha_z, Nr, Nz_half,
        )
        Br_cyl += Br_k
        Bz_cyl += Bz_k

    # ── Цилиндрические → декартовы ───────────────────────────────────────
    if r_t > 0.0:
        cos_phi = x_t / r_t;  sin_phi = y_t / r_t
    else:
        cos_phi = 1.0;        sin_phi = 0.0

    Bx = Br_cyl * cos_phi
    By = Br_cyl * sin_phi
    Bz = Bz_cyl

    Bx_out[i] = Bx;  By_out[i] = By;  Bz_out[i] = Bz

    # ── Метод Бориса ─────────────────────────────────────────────────────
    px = p[i, 0];  py = p[i, 1];  pz = p[i, 2]

    gamma_minus = math.sqrt(1.0 + px*px + py*py + pz*pz)
    half_dt     = 0.5 * dt
    tx = Bx * half_dt / gamma_minus
    ty = By * half_dt / gamma_minus
    tz = Bz * half_dt / gamma_minus

    ppx = px + (py*tz - pz*ty)
    ppy = py + (pz*tx - px*tz)
    ppz = pz + (px*ty - py*tx)

    coef = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz)
    sx = tx*coef;  sy = ty*coef;  sz = tz*coef

    px_new = px + (ppy*sz - ppz*sy)
    py_new = py + (ppz*sx - ppx*sz)
    pz_new = pz + (ppx*sy - ppy*sx)

    gamma_plus = math.sqrt(1.0 + px_new*px_new + py_new*py_new + pz_new*pz_new)

    x[i, 0] = x_t + (px_new / gamma_plus) * dt
    x[i, 1] = y_t + (py_new / gamma_plus) * dt
    x[i, 2] = z_t + (pz_new / gamma_plus) * dt

    p[i, 0] = px_new;  p[i, 1] = py_new;  p[i, 2] = pz_new


