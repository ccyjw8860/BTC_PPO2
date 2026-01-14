"""
Data models for whale threshold calculator
"""

from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class ThresholdDocument:
    """Document structure for threshold storage in MongoDB"""
    _id: str
    datetime: pd.Timestamp
    threshold_99_5: float
    total_trades_window: int
    created_at: datetime


@dataclass
class WindowStats:
    """Statistics for the sliding window"""
    size: int
    total_trades: int
    min_amount: float
    max_amount: float


class ThresholdCalculationError(Exception):
    """Raised when threshold calculation fails"""
    pass
