# -------------------- Public Imports --------------------
from numbers import Real
from typing import TypeAlias
from unicodedata import name
import numpy as np
import pandas as pd



ScalarValue: TypeAlias = Real
NumericValue: TypeAlias = ScalarValue | pd.Series
NormalizedNumeric: TypeAlias = np.float64 | pd.Series


def normalize_numeric_input(
    value: NumericValue,
    *,
    name: str,
    allow_negative: bool = True,
    allow_nan: bool = False,
    allow_series: bool = True,
) -> NormalizedNumeric:
    """
    Normalisiert eine Zahl oder optional eine pandas Series.

    Skalare Zahlen werden als np.float64 zurückgegeben.
    pandas Series werden numerisch konvertiert und erhalten
    den dtype float64.

    Parameters
    ----------
    value:
        Zu normalisierender Wert.
    name:
        Name des Parameters für verständliche Fehlermeldungen.
    allow_negative:
        Gibt an, ob negative Werte erlaubt sind.
    allow_nan:
        Gibt an, ob fehlende beziehungsweise NaN-Werte erlaubt sind.
    allow_series:
        Gibt an, ob pandas Series als Eingabe erlaubt sind.
        Standardwert ist True.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a real number" + (" or pandas Series" if allow_series else "") + ", not bool.")

    if isinstance(value, pd.Series):
        if not allow_series:
            raise TypeError(f"{name} must be a real number; pandas Series are not allowed.")

        try:
            normalized = pd.to_numeric(
                value,
                errors="raise",
            ).astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only numeric values.") from exc

        if not allow_nan and normalized.isna().any():
            raise ValueError(f"{name} must not contain missing values.")

        if not allow_negative and (normalized < 0).any():
            raise ValueError(f"{name} must not contain negative values.")

        return normalized

    if isinstance(value, Real):
        normalized = np.float64(value)

        if not allow_nan and np.isnan(normalized):
            raise ValueError(f"{name} must not be NaN.")

        if not allow_negative and normalized < 0:
            raise ValueError(f"{name} must not be negative.")

        return normalized

    raise TypeError(f"{name} must be a " + ("a real number or pandas Series" if allow_series else "a real number") + ".")




def build_time_series(
    value: Real | pd.Series | None,
    name: str,
    time_index: pd.Index,
) -> pd.Series:
    """
    Hilfsfunktion zum Erstellen einer Zeitreihe für ein Attribut.

    Parameters
    ----------
    name:
        Name des Attributs für Fehlermeldungen.
    value:
        Wert des Attributs, entweder als Skalar (Real) oder als pandas Series.
    time_index:
        Index der Zeitpunkte, für die die Zeitreihe erstellt werden soll.
    """

    if value is None:
        raise RuntimeError(f"{name} is not defined for this object.")

    time_index = pd.Index(time_index)

    if time_index.has_duplicates:
        raise ValueError("time_index must not contain duplicate entries.")

    if isinstance(value, pd.Series):
        if value.index.has_duplicates:
            raise ValueError(f"{name} must not have duplicate index entries.")

        missing_timestamps = time_index.difference(value.index)
        if not missing_timestamps.empty:
            raise KeyError(f"{name} is missing timestamps: {missing_timestamps.tolist()}")

        try:
            time_series = pd.to_numeric(value.reindex(time_index), errors="raise").astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only numeric values.") from exc

        if time_series.isna().any():
            missing_values = time_series.index[time_series.isna()].tolist()
            raise ValueError(f"{name} contains missing values at: {missing_values}")

    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        time_series = pd.Series(
            data=np.float64(value),
            index=time_index,
            dtype=np.float64,
        )

    else:
        raise TypeError(f"{name} must be either a real number or a pandas Series.")

    time_series.name = name

    return time_series
