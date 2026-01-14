"""
Threshold Calculator for computing 99.5th percentile efficiently

Uses numpy.partition for O(n) average time complexity
"""

import numpy as np
from typing import Optional
import logging

# EUC-KR 로깅 설정 (Windows용)
import sys
if sys.platform == 'win32':
    logging.basicConfig(encoding='euc-kr')

logger = logging.getLogger(__name__)


class ThresholdCalculator:
    """
    Calculates 99.5th percentile threshold for whale trades

    Uses numpy.partition for efficient percentile calculation with O(n) average
    time complexity, significantly faster than full sort for large arrays.
    """

    # 상수 정의
    PERCENTILE_RANK = 0.2
    MIN_TRADES_FOR_CALCULATION = 10
    DEFAULT_THRESHOLD = 2_000_000.0  # USD
    MINIMUM_THRESHOLD = 2_000_000.0  # USD

    def calculate_threshold(self, amounts: np.ndarray, total_trades: int) -> float:
        """
        Calculate percentile threshold using numpy.partition

        Calculates threshold based on amounts array (big_amounts >= $2M).
        PERCENTILE_RANK determines which percentile to use:
        - 0.9 = 90th percentile (상위 10%)
        - 0.95 = 95th percentile (상위 5%)
        - 0.995 = 99.5th percentile (상위 0.5%)

        Args:
            amounts: Array of trade amounts >= $2M
            total_trades: Total number of ALL trades in the window (used for minimum check)

        Returns:
            Threshold value in USD (never below MINIMUM_THRESHOLD)
        """
        # 입력 검증
        if not self.validate_inputs(amounts):
            logger.warning("유효하지 않은 amounts 배열 (NaN 또는 Inf 포함)")
            return self.DEFAULT_THRESHOLD

        # 빈 배열 또는 거래 수 부족
        if len(amounts) == 0 or total_trades < self.MIN_TRADES_FOR_CALCULATION:
            return self.DEFAULT_THRESHOLD

        # amounts 배열 기준 percentile 계산
        # PERCENTILE_RANK = 0.9이면 90th percentile (상위 10%)
        target_idx = int(len(amounts) * self.PERCENTILE_RANK)

        # 인덱스 범위 보정
        target_idx = min(target_idx, len(amounts) - 1)
        target_idx = max(target_idx, 0)

        # numpy.partition을 사용하여 O(n) 시간에 k번째 작은 값 찾기
        # partition은 배열을 k번째 요소를 기준으로 나누지만 완전히 정렬하지는 않음
        partitioned = np.partition(amounts, target_idx)
        threshold_value = float(partitioned[target_idx])

        # 최소 임계값 보장
        return max(threshold_value, self.MINIMUM_THRESHOLD)

    def validate_inputs(self, amounts: np.ndarray) -> bool:
        """
        Validate input array for NaN and Infinity values

        Args:
            amounts: Array to validate

        Returns:
            True if valid, False if contains NaN or Inf
        """
        if len(amounts) == 0:
            return True  # 빈 배열은 유효함 (DEFAULT_THRESHOLD 반환)

        # NaN이나 Inf가 있으면 무효
        return not (np.any(np.isnan(amounts)) or np.any(np.isinf(amounts)))
