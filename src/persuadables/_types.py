"""Array aliases used across the package (numpy arrays are the lingua franca, not DataFrames)."""

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
StrArray = npt.NDArray[np.str_]
AnyArray = npt.NDArray[Any]
