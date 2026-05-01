"""
Shared technical indicator utilities.
Used by risk_manager.py, base_bot.py, and other modules.
"""

import pandas as pd


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    Returns a pandas Series aligned with the input index.
    """
    try:
        up       = high.diff()
        down     = -low.diff()
        plus_dm  = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr      = tr.ewm(span=period).mean()
        plus_di  = 100 * plus_dm.ewm(span=period).mean() / (atr + 1e-9)
        minus_di = 100 * minus_dm.ewm(span=period).mean() / (atr + 1e-9)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        return dx.ewm(span=period).mean()
    except Exception:
        return pd.Series([25.0] * len(close), index=close.index)
