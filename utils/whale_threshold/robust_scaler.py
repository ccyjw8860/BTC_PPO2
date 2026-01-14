"""
Robust Scaling 유틸리티

중앙값/IQR 기반 rolling robust scaling 구현
Phase 2.1: Pandas reference implementation
Phase 2.3: Numba JIT 최적화 (우선 시도)
Phase 2.2: NumPy fallback (대비책)
"""

import pandas as pd
import numpy as np
from typing import Union
import logging

# Phase 2.3: Numba JIT 최적화 시도
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available, falling back to pandas implementation")

# EUC-KR 로깅 설정 (Windows용)
import sys
if sys.platform == 'win32':
    logging.basicConfig(encoding='euc-kr')

logger = logging.getLogger(__name__)


def _rolling_robust_scale_pandas(
    series: pd.Series, window: int, min_iqr: float
) -> pd.Series:
    """
    Reference implementation using pandas (기존 로직)

    시간적 인과관계 보장: shift(1) 적용
    - 시간 t의 스케일링은 t-1까지의 통계량만 사용

    Args:
        series: 입력 시계열
        window: 롤링 윈도우 크기 (default: 2016 candles ~1 week)
        min_iqr: 최소 IQR (0 방지, default: 0.001)

    Returns:
        Robust scaled 시계열 (첫 window+1개는 NaN)
    """
    # Rolling statistics 계산 후 shift(1) 적용
    # 현재 시점 t의 값이 통계량에 포함되지 않도록 함
    rolling_median = series.rolling(window).median().shift(1)
    rolling_q75 = series.rolling(window).quantile(0.75).shift(1)
    rolling_q25 = series.rolling(window).quantile(0.25).shift(1)

    # IQR 계산 및 최소값 적용
    iqr = rolling_q75 - rolling_q25
    iqr = iqr.clip(lower=min_iqr)  # IQR이 0에 가까우면 min_iqr로 대체

    # Robust scaling 적용
    scaled = (series - rolling_median) / iqr

    return scaled


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def _compute_rolling_quantile_numba(
        values: np.ndarray, window: int, q: float
    ) -> np.ndarray:
        """
        Numba JIT 컴파일된 rolling quantile

        nopython=True: Pure Python 코드 금지 (최대 성능)
        cache=True: 첫 컴파일 결과를 캐싱하여 재사용

        Args:
            values: 입력 배열
            window: 윈도우 크기
            q: Quantile (0.0 ~ 1.0)

        Returns:
            Rolling quantile 배열
        """
        n = len(values)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_data = values[i - window + 1 : i + 1]
            result[i] = np.quantile(window_data, q)

        return result

    def _rolling_robust_scale_numba(
        values: np.ndarray,
        window: int,
        min_iqr: float
    ) -> np.ndarray:
        """
        Numba 최적화 rolling robust scaling

        Phase 2.3: JIT 컴파일로 5-20배 속도 향상 목표

        Args:
            values: 입력 배열 (numpy array)
            window: 롤링 윈도우 크기
            min_iqr: 최소 IQR

        Returns:
            Robust scaled numpy array
        """
        # JIT 컴파일된 rolling quantile 계산
        medians = _compute_rolling_quantile_numba(values, window, 0.5)
        q25 = _compute_rolling_quantile_numba(values, window, 0.25)
        q75 = _compute_rolling_quantile_numba(values, window, 0.75)

        # shift(1) 적용 (vectorized)
        medians_shifted = np.roll(medians, 1)
        medians_shifted[0] = np.nan

        # IQR 계산 및 clipping
        iqr = np.clip(q75 - q25, min_iqr, None)
        iqr_shifted = np.roll(iqr, 1)
        iqr_shifted[0] = min_iqr

        # Robust scaling
        return (values - medians_shifted) / iqr_shifted


# def robust_scale(
#     values: Union[pd.Series, np.ndarray],
#     window: int,
#     min_iqr: float
# ) -> np.ndarray:
#     """
#     Robust scaling 진입점

#     Phase 2.1: pandas 구현 사용 (reference)
#     Phase 2.3: numba 구현 우선 사용 (5-20배 빠름)
#     Phase 2.2: numpy 구현 (fallback, 미구현)

#     Args:
#         values: 입력 데이터 (pd.Series 또는 np.ndarray)
#         window: 롤링 윈도우 크기
#         min_iqr: 최소 IQR

#     Returns:
#         Robust scaled numpy array
#     """
#     # Convert to numpy array
#     if isinstance(values, pd.Series):
#         values_array = values.values
#     else:
#         values_array = values

#     # Phase 2.3: Use Numba if available
#     if NUMBA_AVAILABLE:
#         return _rolling_robust_scale_numba(values_array, window, min_iqr)
#     else:
#         # Fallback to pandas
#         series = pd.Series(values_array)
#         return _rolling_robust_scale_pandas(series, window, min_iqr).values


def robust_scale(series: pd.Series, window: int, min_iqr: float) -> np.ndarray:
    rolling = series.rolling(window=window)
    median = rolling.median().shift(1)
    q1 = rolling.quantile(0.25).shift(1)
    q3 = rolling.quantile(0.75).shift(1)
    
    iqr = (q3 - q1).clip(lower=min_iqr) # 분모가 0이 되지 않도록 보정
    
    scaled = (series - median) / iqr
    
    # inf를 0으로 치환하고 남은 NaN은 앞의 값으로 채우거나 0으로 처리
    return scaled.replace([np.inf, -np.inf], np.nan).fillna(0).values