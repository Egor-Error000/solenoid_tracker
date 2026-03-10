# ============================================================
# БЛОК I — Импорты и датаклассы
# ============================================================

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numba import cuda
from ..physics import *

# ── Датаклассы ────────────────────────────────────────────────

@dataclass
class SolenoidParams:
    """Геометрия и масштаб одного соленоида."""
    center:  np.ndarray   # (3,) [м]  центр (x, y, z)
    const:   float        # безразмерный коэффициент масштаба поля
    length:  float        # полная длина [м]
    R_inner: float        # внутренний радиус [м]
    R_outer: float        # внешний радиус [м]


@dataclass
class BeamPhysicalParams:
    """Физические параметры частиц пучка (все в СИ)."""
    mass:           float   # [кг]
    charge:         float   # [Кл]
    c:              float   # [м/с]
    kinetic_energy: float   # [Дж]  (не эВ!)


@dataclass
class BeamNumericalParams:
    """
    Параметры численного интегрирования.

    t_start, t_end задаются в физических единицах [с].
    Безразмерный шаг dt̃ вычисляется автоматически в BeamTracerCUDA:
        dt̃ = (t_end - t_start) / (n_steps · T0)

    Пример:
        BeamNumericalParams(t_start=0.0, t_end=15e-9,
                            n_steps=16_000, threads_per_block=128)
    """
    t_start:           float   # [с]  начало интегрирования (обычно 0)
    t_end:             float   # [с]  конец  интегрирования
    n_steps:           int     # число шагов
    threads_per_block: int     # потоков на блок CUDA


@dataclass
class DimensionlessScaling:
    """
    Масштабы для обезразмеривания.
    Заполняется фабричным методом ScalingTransform.from_physical_problem().
    """
    length:   float   # L0 [м]
    time:     float   # T0 [с]
    momentum: float   # P0 [кг·м/с]
    field:    float   # B0 [Тл]


from abc import ABC, abstractmethod
class FieldGridBase(ABC):
    """Базовый класс для любой сетки поля. Все значения в [м] и [Тл]."""

    @property
    @abstractmethod
    def r_min_phys(self) -> float: ...

    @property
    @abstractmethod
    def r_max_phys(self) -> float: ...

    @property
    @abstractmethod
    def z_min_phys(self) -> float: ...

    @property
    @abstractmethod
    def z_max_phys(self) -> float: ...

    @property
    @abstractmethod
    def Nr(self) -> int: ...

    @property
    @abstractmethod
    def Nz(self) -> int: ...

    @property
    @abstractmethod
    def Br_host(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def Bz_host(self) -> np.ndarray: ...