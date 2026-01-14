"""
Volume Feature Calculator for crypto trading data

볼륨 특성 계산기 - 거래량, OI, CVD 특성을 robust scaling과 함께 계산

주요 기능:
- 거래량 특성: Log1p 변환 + Robust Scaling (1개)
- OI 특성: Open Interest + Robust Scaling (1개)
- CVD 특성: Rolling Volume Delta + Robust Scaling (1개, Iteration 3에서 stationary로 변경)

Phase 3: 고래 비율 특성 제거됨 (whale_buy_ratio, whale_sell_ratio)
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


class VolumeFeatureCalculator:
    """
    볼륨 특성 계산기 - 거래량, OI, CVD 특성 계산

    Features (모두 Robust Scaling 적용):
    - feat_vol: Log1p(volume) + Robust Scaling (1개)
    - feat_oi: Open Interest + Robust Scaling (1개)
    - feat_cvd: Rolling Volume Delta + Robust Scaling (1개, stationary)

    Total: 3 features (모두 robust scaling 적용)

    Phase 3: whale ratio features 제거됨
    """

    def __init__(self, window_size: int = 2016, min_iqr: float = 0.001, cvd_window: int = 2016):
        """
        Initialize volume feature calculator

        Args:
            window_size: Rolling window for robust scaling (~1 week = 2016 candles)
            min_iqr: Minimum IQR to prevent division by zero (default 0.001 = 1e-3)
            cvd_window: Rolling window for CVD calculation (default: 2016 = 1 week)
        """
        self.window_size = window_size
        self.min_iqr = min_iqr
        self.cvd_window = cvd_window

    def calculate_features(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        DataFrame에 모든 볼륨 특성 컬럼 추가

        Args:
            df: 볼륨 데이터가 포함된 DataFrame
                Required columns: quote_volume, sum_open_interest_value, taker_buy_quote
            inplace: True이면 df를 직접 수정, False이면 복사본 생성 (기본값: False)

        Returns:
            특성 컬럼이 추가된 DataFrame (3개 특성: feat_vol, feat_oi, feat_cvd)
        """
        logger.info(f"볼륨 특성 계산 시작: {len(df)}개 캔들")

        # 원본 DataFrame 복사 (inplace=False일 때만)
        if not inplace:
            df = df.copy()

        # 볼륨 특성 계산
        df = self._calculate_volume_feature(df)

        # OI 특성 계산
        df = self._calculate_oi_feature(df)

        # CVD 특성 계산
        # df = self._calculate_cvd_feature(df)

        # Robust Scaling 적용 (3개 특성만)
        df = self._apply_robust_scaling(df)

        # 생성된 특성 개수 확인
        # volume_features = ['feat_vol', 'feat_oi', 'feat_cvd']
        volume_features = ['feat_vol', 'feat_oi']
        created_features = [f for f in volume_features if f in df.columns]
        logger.info(f"볼륨 특성 계산 완료: {len(created_features)}개 특성 생성됨")

        return df

    def _calculate_volume_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 로그 특성 계산

        Feature:
        - feat_vol: Log1p(quote_volume) - log(1 + x) 변환으로 0 처리

        Args:
            df: DataFrame with quote_volume column

        Returns:
            DataFrame with feat_vol column (before scaling)
        """
        # 1. 거래량: Log1p 변화율 (Log-Difference)로 수정
        # 전 캔들 대비 거래량 증가/감소에 집중
        log_vol = np.log1p(df['quote_volume'])
        df['feat_vol'] = log_vol - log_vol.shift(1)

        return df

    def _calculate_oi_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Open Interest 특성 계산

        Feature:
        - feat_oi: sum_open_interest_value (변환 없이 그대로)

        Args:
            df: DataFrame with sum_open_interest_value column

        Returns:
            DataFrame with feat_oi column (before scaling)
        """
        # 절대값이 아닌 전 캔들 대비 변화율(로그 차분) 사용
        # 0으로 나누기 및 log(0) 방지를 위해 아주 작은 값(1e-9) 추가
        epsilon = 1e-9
        df['feat_oi'] = np.log((df['sum_open_interest_value'] + epsilon) / 
                            (df['sum_open_interest_value'].shift(1) + epsilon))
        return df
        

    def _calculate_cvd_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cumulative Volume Delta (CVD) 특성 계산 - Rolling Window로 Stationary 보장

        Feature:
        - feat_cvd: Rolling 순 거래량 (Stationary)
          Net_Vol = Buy_Vol - Sell_Vol
          Buy_Vol = taker_buy_quote
          Sell_Vol = quote_volume - taker_buy_quote
          Net_Vol = 2 * taker_buy_quote - quote_volume
          CVD = Net_Vol.rolling(cvd_window).sum()  # NOT cumsum (unbounded)

        Note: Iteration 3에서 cumsum() → rolling().sum()으로 변경하여 stationarity 보장

        Args:
            df: DataFrame with quote_volume and taker_buy_quote columns

        Returns:
            DataFrame with feat_cvd column (before scaling)
        """
        # Net Volume 계산
        # Buy Volume = taker_buy_quote
        # Sell Volume = quote_volume - taker_buy_quote
        # Net Volume = Buy - Sell = taker_buy_quote - (quote_volume - taker_buy_quote)
        #            = 2 * taker_buy_quote - quote_volume
        net_volume = 2 * df['taker_buy_quote'] - df['quote_volume']

        # Rolling sum (stationary) - NOT cumsum (unbounded)
        # cvd_window 크기 만큼의 rolling window로 합계 계산
        df['feat_cvd'] = net_volume.rolling(window=self.cvd_window, min_periods=1).sum()

        return df

    # _calculate_whale_ratio_features() 메서드 제거됨 (Phase 3)

    def _apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 볼륨 특성에 rolling robust scaling 적용

        Robust Scaling Formula: (x - rolling_median) / rolling_IQR

        적용 대상: 3개 특성 (Phase 3: whale ratios 제거됨)
        - feat_vol, feat_oi, feat_cvd

        Args:
            df: DataFrame with feat_* columns

        Returns:
            DataFrame with scaled features
        """
        # 3개 특성에만 scaling 적용 (Phase 3: whale ratios 제거됨)
        # features_to_scale = ['feat_vol', 'feat_oi', 'feat_cvd']
        features_to_scale = ['feat_vol', 'feat_oi']

        logger.info(f"Robust scaling 적용: {len(features_to_scale)}개 특성, window={self.window_size}")

        for col in features_to_scale:
            if col in df.columns:
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
