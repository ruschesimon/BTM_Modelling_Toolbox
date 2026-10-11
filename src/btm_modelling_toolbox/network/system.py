"""Time context shared by energy system components."""

from dataclasses import dataclass

import pandas as pd


@dataclass(kw_only=True)
class EnergySystem:
    """Provide the time index used to resolve component time series."""

    time_index: pd.RangeIndex
