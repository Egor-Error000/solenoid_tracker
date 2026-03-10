import numpy as np, matplotlib.pyplot as plt
from ..dataclasses.dataclasses import SolenoidParams
from ..physics import *
# ============================================================
# БЛОК III — Геометрия и визуализация
# ============================================================
#
# Содержит:
#   1. cartesian_to_cylindrical  — для диагностики геометрии пучка
#   2. plot_solenoid_system      — визуализация конфигурации системы
#
# ============================================================


def cartesian_to_cylindrical(points: np.ndarray):
    """
    Декартовы (x, y, z) → цилиндрические компоненты.

    Используется в BeamDiagnostics для вычисления r и phi частиц.
    Деление на ноль при r=0 исключено через маску.

    Parameters
    ----------
    points : (N, 3) [м]

    Returns
    -------
    r       : (N,)  радиальная координата √(x²+y²) [м]
    z       : (N,)  осевая координата [м]
    cos_phi : (N,)  cos φ = x / r  (0 при r=0)
    sin_phi : (N,)  sin φ = y / r  (0 при r=0)
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r     = np.sqrt(x * x + y * y)
    mask  = r > 0.0
    inv_r = np.where(mask, 1.0 / np.where(mask, r, 1.0), 0.0)

    return r, z, x * inv_r, y * inv_r


def plot_solenoid_system(solenoids, z_start=0.0, r_beam=1e-3):
    """
    Поперечное сечение системы соленоидов в плоскости (z, x).

    Parameters
    ----------
    solenoids : list[SolenoidParams]
    z_start   : float  — осевая позиция старта пучка [м]
    r_beam    : float  — радиус пучка для отображения [м]
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for k, sol in enumerate(solenoids):
        z_c = sol.center[2]
        z1  = z_c - sol.length / 2
        z2  = z_c + sol.length / 2

        # Обмотка
        ax.fill_betweenx(
            [-sol.R_outer, sol.R_outer], z1, z2,
            color='tab:blue', alpha=0.25,
            label=f'Соленоид {k+1}' if k == 0 else f'Соленоид {k+1}',
        )
        # Внутренний канал (вырез)
        ax.fill_betweenx(
            [-sol.R_inner, sol.R_inner], z1, z2,
            color='white',
        )
        # Осевая метка центра соленоида
        ax.plot([z_c, z_c], [-sol.R_outer, sol.R_outer],
                'k--', alpha=0.4, linewidth=0.8)
        ax.text(z_c, sol.R_outer * 1.05, f'S{k+1}',
                ha='center', va='bottom', fontsize=8, color='tab:blue')

    # Стартовое сечение пучка
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        z_start + np.zeros_like(theta),
        r_beam  * np.cos(theta),
        'r', linewidth=1.5, label=f'Старт пучка  r={r_beam*1e3:.1f} мм',
    )

    ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax.set_xlabel('z [м]')
    ax.set_ylabel('x [м]')
    ax.set_title('Система соленоидов — сечение xz')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()