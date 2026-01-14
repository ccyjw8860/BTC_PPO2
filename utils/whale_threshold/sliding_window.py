"""
Sliding Window Buffer for maintaining 24-hour (288 candle) window

Uses collections.deque for O(1) append/popleft operations
"""

from collections import deque
from typing import List, Dict
import numpy as np


class SlidingWindowBuffer:
    """
    Maintains a sliding window of 288 candles (24 hours at 5-minute intervals)

    Efficiently tracks trade amounts and total trade counts across the window
    using deque data structure for O(1) operations.
    """

    def __init__(self, window_size: int = 288):
        """
        Initialize sliding window buffer

        Args:
            window_size: Number of candles to maintain (default: 288 for 24 hours)
        """
        self.window_size = window_size
        # 각 캔들별 거래 금액 리스트를 저장
        self.amounts_by_candle = deque(maxlen=window_size)
        # 각 캔들별 전체 거래 수를 저장
        self.total_trades_by_candle = deque(maxlen=window_size)

    def add_candle_trades(self, amounts: List[float], total_trades: int) -> None:
        """
        Add trade data from a single candle to the window

        Args:
            amounts: List of trade amounts >= $2M for this candle
            total_trades: Total number of trades in this candle (including < $2M)
        """
        self.amounts_by_candle.append(amounts)
        self.total_trades_by_candle.append(total_trades)

    def get_amounts(self) -> np.ndarray:
        """
        Get flattened array of all trade amounts in the window

        Returns:
            numpy array of all amounts from all candles in the window
        """
        all_amounts = []
        for candle_amounts in self.amounts_by_candle:
            all_amounts.extend(candle_amounts)
        return np.array(all_amounts, dtype=np.float64)

    def get_total_trades_in_window(self) -> int:
        """
        Get total number of trades across all candles in the window

        Returns:
            Sum of total_trades from all candles
        """
        return sum(self.total_trades_by_candle)

    def is_full(self) -> bool:
        """
        Check if the window has reached its target size

        Returns:
            True if window contains window_size candles, False otherwise
        """
        return len(self.amounts_by_candle) >= self.window_size

    def get_window_stats(self) -> Dict:
        """
        Get statistics about the current window

        Returns:
            Dictionary containing size, total_trades, min_amount, max_amount, total_amounts
        """
        amounts = self.get_amounts()

        return {
            'size': len(self.amounts_by_candle),
            'total_trades': self.get_total_trades_in_window(),
            'min_amount': float(amounts.min()) if len(amounts) > 0 else 0.0,
            'max_amount': float(amounts.max()) if len(amounts) > 0 else 0.0,
            'total_amounts': len(amounts)
        }
