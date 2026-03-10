# solenoid_tracker/core/__init__.py

# Импортируем датаклассы
from ..dataclasses.dataclasses import (
    SolenoidParams,
    BeamPhysicalParams,
    BeamNumericalParams,
    DimensionlessScaling,
    FieldGridBase
)

# Импортируем сетки
from ..field_grids.field_grid import FNTFieldGrid

# Импортируем геометрию
from ..geometry.geometry import (
    cartesian_to_cylindrical,
    plot_solenoid_system
)

# Импортируем инструменты масштабирования и диагностики
from ..scaling.scaling import (
    ScalingTransform,
    BeamDiagnostics,
    DimensionlessGridAdapter
)

# Импортируем сам трекер
from ..tracer.tracer import BeamTracerCUDA, timed_run

# Теперь из ноутбука достаточно написать: from core import BeamTracerCUDA