"""
Price Feature Calculator for OHLCV data

가격 특성 계산기 - OHLC, EMA, BB 특성을 robust scaling과 함께 계산

주요 기능:
- OHLC 로그 특성: 수익률, 봉 비율, 그림자 (4개)
- EMA 특성: 5/20/40/60/120 기간 기울기 (ATR-normalized) 및 거리 (10개)
- Bollinger Band 특성: 폭, 위치 (2개)
- Robust Scaling: 중앙값/IQR 기반 이상치 강건 스케일링 (16개 전체 적용)
- ATR Normalization: EMA slopes를 ATR로 나누어 scale-invariant하게 만듦

Phase 4: MA → EMA 전환, EMA40 윈도우 추가 (14 → 16 features)
"""

import pandas as pd
import numpy as np
import logging

# Phase 2.1: Shared robust scaling utility
from .robust_scaler import robust_scale

# EUC-KR 로깅 설정 (Windows용)
import sys
if sys.platform == 'win32':
    logging.basicConfig(encoding='euc-kr')

logger = logging.getLogger(__name__)


class PriceFeatureCalculator:
    """
    가격 특성 계산기 - OHLC, EMA, BB 특성을 robust scaling과 함께 계산

    Features:
    - OHLC 특성: 로그 수익률, 봉 비율 (4개)
    - EMA 특성: 5, 20, 40, 60, 120 기간 기울기 (ATR-normalized) 및 거리 (10개)
    - BB 특성: (20,2) 폭 및 위치 (2개)
    - Robust Scaling: 2016 캔들 윈도우, min_iqr로 0 방지
    - ATR Normalization: EMA slopes를 ATR로 나누어 scale-invariant

    Total: 16 features (Phase 4: EMA40 추가)
    """

    # 상수 정의
    EMA_WINDOWS = [5, 20, 40, 60, 120]  # EMA 윈도우 크기 (Phase 4: MA → EMA, 40 추가)
    BB_WINDOW = 20  # Bollinger Band 윈도우
    BB_STD_DEV = 2  # Bollinger Band 표준편차 배수
    ATR_WINDOW = 14  # ATR 윈도우 (표준값)

    def __init__(self, window_size: int = 2016, min_iqr: float = 0.001):
        """
        Initialize price feature calculator

        Args:
            window_size: Rolling window for robust scaling (~1 week = 2016 candles)
            min_iqr: Minimum IQR to prevent division by zero (default 0.001 = 1e-3)
        """
        self.window_size = window_size
        self.min_iqr = min_iqr

    def calculate_features(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        DataFrame에 모든 가격 특성 컬럼 추가

        Args:
            df: OHLCV 데이터가 포함된 DataFrame
                Required columns: open, high, low, close, volume
            inplace: True이면 df를 직접 수정, False이면 복사본 생성 (기본값: False)

        Returns:
            특성 컬럼이 추가된 DataFrame
        """
        logger.info(f"가격 특성 계산 시작: {len(df)}개 캔들")

        # 원본 DataFrame 복사 (inplace=False일 때만)
        if not inplace:
            df = df.copy()

        # OHLC 특성 계산
        df = self._calculate_ohlc_features(df)

        # EMA 특성 계산 (Phase 4: MA → EMA)
        df = self._calculate_ema_features(df)

        # BB 특성 계산
        df = self._calculate_bb_features(df)

        # Robust Scaling 적용
        # df = self._apply_robust_scaling(df)

        # 생성된 특성 개수 확인
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
        logger.info(f"가격 특성 계산 완료: {len(feature_cols)}개 특성 생성됨")

        return df

    def _calculate_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLC 로그 특성 계산

        Features:
        - feat_close_ret: Log return (로그 수익률)
        - feat_body: Log(close/open) - signed body (양수=양봉, 음수=음봉)
        - feat_upper: Log(high/close) - upper shadow (항상 >= 0)
        - feat_lower: Log(low/close) - lower shadow (항상 <= 0)

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with 4 new feature columns
        """
        # 로그 수익률 (close-to-close return)
        df['feat_close_ret'] = np.log(df['close'] / df['close'].shift(1))

        # 로그 봉 비율 (signed: 양봉이면 +, 음봉이면 -)
        df['feat_body'] = np.log(df['close'] / df['open'])

        # 로그 위쪽 그림자 (high에서 close까지 거리)
        df['feat_upper'] = np.log(df['high'] / df['close'])

        # 로그 아래쪽 그림자 (low에서 close까지 거리)
        df['feat_lower'] = np.log(df['low'] / df['close'])

        return df

    def _calculate_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EMA 기울기 및 거리 특성 계산 (Phase 4: MA → EMA 전환, ATR normalization 유지)

        For each EMA window (5, 20, 40, 60, 120):
        - feat_ema{w}_slope: EMA의 로그 기울기 / ATR (scale-invariant)
        - feat_dist_ema{w}: 가격과 EMA의 로그 거리 (bias)

        Args:
            df: DataFrame with close, high, low columns

        Returns:
            DataFrame with 10 new feature columns (5 windows × 2 metrics)
        """
        # ATR 계산 (slopes normalization용)
        atr = self._calculate_atr(df, window=self.ATR_WINDOW)

        for window in self.EMA_WINDOWS:
            # 지수 이동 평균 계산 (Phase 4: .rolling() → .ewm())
            ema = df['close'].ewm(span=window, adjust=False).mean()

            # 로그 기울기 (EMA의 변화율)
            raw_slope = np.log(ema / ema.shift(1))

            # ATR로 나누어 scale-invariant하게 만들기
            # min_iqr를 사용하여 0으로 나누기 방지
            safe_atr = atr.clip(lower=self.min_iqr)
            df[f'feat_ema{window}_slope'] = raw_slope / safe_atr

            # 로그 거리 (가격과 EMA의 bias)
            df[f'feat_dist_ema{window}'] = np.log(df['close'] / ema)

        return df

    def _calculate_bb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Band (20,2) 특성 계산

        Features:
        - feat_bb_width: Log(band_width / close) - 밴드 폭의 로그
        - feat_bb_pos: %B position - 밴드 내 위치 (0~1, 밴드 밖이면 범위 벗어남)

        Args:
            df: DataFrame with close column

        Returns:
            DataFrame with 2 new feature columns
        """
        # MA와 표준편차 계산
        ma20 = df['close'].rolling(self.BB_WINDOW).mean()
        std20 = df['close'].rolling(self.BB_WINDOW).std()

        # 상단/하단 밴드
        upper_band = ma20 + self.BB_STD_DEV * std20
        lower_band = ma20 - self.BB_STD_DEV * std20
        band_width = upper_band - lower_band

        # 로그 폭 (0으로 나누기 방지)
        df['feat_bb_width'] = np.log(band_width / df['close'] + self.min_iqr)

        # %B (band position) - 0으로 나누기 방지
        df['feat_bb_pos'] = (df['close'] - lower_band) / (band_width + self.min_iqr)

        return df

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Average True Range (ATR) 계산

        ATR은 가격 변동성의 척도로, True Range의 이동평균입니다.
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)

        Iteration 5에서 MA slopes를 정규화하는데 사용됩니다.

        Args:
            df: DataFrame with high, low, close columns
            window: ATR 계산 윈도우 (기본값 14 = 표준값)

        Returns:
            ATR 시리즈 (첫 window개는 NaN)
        """
        # True Range 3가지 성분 계산
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()

        # True Range = max of 3 components
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR = rolling mean of True Range
        atr = true_range.rolling(window=window, min_periods=1).mean()

        return atr

    def _apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 특성에 rolling robust scaling 적용

        Robust Scaling Formula: (x - rolling_median) / rolling_IQR

        Args:
            df: DataFrame with feat_* columns

        Returns:
            DataFrame with all feat_* columns scaled
        """
        feature_cols = [col for col in df.columns if col.startswith('feat_')]

        logger.info(f"Robust scaling 적용: {len(feature_cols)}개 특성, window={self.window_size}")

        for col in feature_cols:
            df[col] = self._rolling_robust_scale(df[col], self.window_size, self.min_iqr)

        return df

    def _rolling_robust_scale(
        self, series: pd.Series, window: int, min_iqr: float
    ) -> pd.Series:
        """
        Rolling robust scaling: (x - median) / IQR

        Phase 2.1: Delegate to shared utility (robust_scaler.py)

        ✅ Data leakage 방지: shift(1) 적용으로 시간적 인과관계 보장
        - 시간 t의 스케일링은 t-1까지의 통계량만 사용

        Args:
            series: 입력 시리즈
            window: 롤링 윈도우 크기
            min_iqr: 최소 IQR (0 방지)

        Returns:
            Scaled 시리즈 (첫 window개는 NaN)
        """
        # Delegate to shared utility
        result = robust_scale(series, window, min_iqr)
        return pd.Series(result, index=series.index)
