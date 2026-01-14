"""
Whale Histogram Feature Calculator for crypto trading data

고래 거래 가격 히스토그램 계산기 - 가격 분포 특성 계산

주요 기능:
- 고래 거래 존재 여부 바이너리 플래그 (1개)
- 매수 거래 가격 히스토그램 (10개 bins)
- 매도 거래 가격 히스토그램 (10개 bins)
- reliable_total_vol (from $lookup BTCUSDT.P)으로 정규화
- [0.0, 1.0] 범위로 hard clipping (수치 폭주 방지)
- robust scaling은 Iteration 2에서 추가 예정
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Phase 2.1: Shared robust scaling utility
from .robust_scaler import robust_scale

# EUC-KR 로깅 설정 (Windows용)
import sys
if sys.platform == 'win32':
    logging.basicConfig(encoding='euc-kr')

logger = logging.getLogger(__name__)


class WhaleHistogramCalculator:
    """
    고래 거래 가격 히스토그램 계산기

    Features:
    - feat_is_whale: 고래 거래 존재 여부 (1개)
    - feat_buy_hist_0 ~ feat_buy_hist_9: 매수 거래 가격 분포 (10개)
    - feat_sell_hist_0 ~ feat_sell_hist_9: 매도 거래 가격 분포 (10개)

    Total: 21 features
    """

    # 상수 정의
    NUM_BINS = 10  # 가격 히스토그램 빈 개수
    MIN_VOLUME = 1e-3  # 최소 거래량 (0으로 나누기 방지)
    HISTOGRAM_CLIP_MIN = 0.0  # 히스토그램 최소값
    HISTOGRAM_CLIP_MAX = 1.0  # 히스토그램 최대값
    SCALED_CLIP_MIN = -10.0  # Scaled 값 최소값
    SCALED_CLIP_MAX = 10.0  # Scaled 값 최대값

    def __init__(self, window_size: int = 2016, min_iqr: float = 0.001):
        """
        히스토그램 계산기 초기화

        Args:
            window_size: Rolling window size for robust scaling (기본값: 2016 = 1 week)
            min_iqr: Minimum IQR to prevent division by zero (기본값: 0.001)
        """
        self.window_size = window_size
        self.min_iqr = min_iqr

    def calculate_features(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        DataFrame에 21개 히스토그램 특성 추가

        Args:
            df: DataFrame with required columns:
                - whale_trades_dict: {price_str: [{type: 'Buy'|'Sell', amount: float}]}
                - high: 캔들 고가
                - low: 캔들 저가
                - quote_volume: 거래량 (USD) - 기존 집계값
                - reliable_total_vol: $lookup BTCUSDT.P로 계산된 신뢰 가능한 총 거래량
            inplace: True이면 df를 직접 수정, False이면 복사본 생성 (기본값: False)

        Returns:
            특성 컬럼이 추가된 DataFrame
        """
        logger.info(f"고래 히스토그램 특성 계산 시작: {len(df)}개 캔들")

        # 원본 DataFrame 복사 (inplace=False일 때만)
        if not inplace:
            df = df.copy()

        # 21개 특성 초기화 (기본값 0.0)
        df['feat_is_whale'] = 0.0
        for i in range(self.NUM_BINS):
            df[f'feat_buy_hist_{i}'] = 0.0
            df[f'feat_sell_hist_{i}'] = 0.0

        # 각 캔들마다 히스토그램 계산
        for idx in range(len(df)):
            row = df.iloc[idx]

            whale_trades_dict = row.get('whale_trades_dict', {})

            # Binary flag (고래 거래 존재 여부)
            if whale_trades_dict:
                df.at[idx, 'feat_is_whale'] = 1.0

                low = row['low']
                high = row['high']
                quote_volume = row['quote_volume']
                reliable_total_vol = row.get('reliable_total_vol', quote_volume)

                # 가격 히스토그램 계산
                buy_hist, sell_hist = self._calculate_histogram(
                    whale_trades_dict, low, high, quote_volume, reliable_total_vol
                )

                # DataFrame에 저장
                for i in range(self.NUM_BINS):
                    df.at[idx, f'feat_buy_hist_{i}'] = buy_hist[i]
                    df.at[idx, f'feat_sell_hist_{i}'] = sell_hist[i]

        # Robust scaling 적용 (20개 히스토그램 특성)
        df = self._apply_robust_scaling(df)

        logger.info(f"고래 히스토그램 특성 계산 완료: 21개 특성 생성됨")
        return df

    def _get_safe_denominator(self, quote_volume: float, reliable_total_vol: float) -> float:
        """
        안전한 정규화 분모 계산

        max(quote_volume, reliable_total_vol, MIN_VOLUME)을 사용하여:
        - reliable_total_vol 우선 사용 (더 정확한 값)
        - quote_volume이 0이거나 작아도 안전
        - 최소값 MIN_VOLUME으로 수치 폭주 방지

        Args:
            quote_volume: 기존 집계된 거래량
            reliable_total_vol: $lookup BTCUSDT.P로 계산된 신뢰 가능한 총 거래량

        Returns:
            안전한 분모 값 (>= MIN_VOLUME)
        """
        return max(quote_volume, reliable_total_vol, self.MIN_VOLUME)

    def _calculate_histogram(
        self, whale_trades_dict: Dict, low: float, high: float,
        quote_volume: float, reliable_total_vol: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 캔들의 가격 히스토그램 계산

        Args:
            whale_trades_dict: {price_str: [{type: 'Buy'|'Sell', amount: float}]}
            low: 캔들 저가
            high: 캔들 고가
            quote_volume: 거래량 (USD) - 기존 집계값
            reliable_total_vol: $lookup BTCUSDT.P로 계산된 신뢰 가능한 총 거래량

        Returns:
            (buy_hist, sell_hist) - 각각 10개 bin의 [0.0, 1.0] 범위로 clipped된 정규화된 값
        """
        # 안전한 분모 계산 (extract method refactoring)
        safe_volume = self._get_safe_denominator(quote_volume, reliable_total_vol)

        # 빈 초기화
        buy_hist = np.zeros(self.NUM_BINS)
        sell_hist = np.zeros(self.NUM_BINS)

        # 제로 폭 캔들 처리 (low == high)
        if low == high:
            return buy_hist, sell_hist

        # 가격 빈 경계 생성 (11개 edge → 10개 bin)
        # np.linspace(low, high, 11) → [low, ..., high]
        bin_edges = np.linspace(low, high, self.NUM_BINS + 1)

        # 각 가격 레벨의 거래 처리
        for price_str, trades in whale_trades_dict.items():
            try:
                price = float(price_str)
            except ValueError:
                # 잘못된 price 문자열 무시
                continue

            # 가격이 범위 밖이면 스킵 (데이터 품질 이슈)
            if price < low or price > high:
                continue

            # 어느 빈에 속하는지 결정
            # np.digitize returns 1-based index, so subtract 1 for 0-based
            bin_idx = np.digitize(price, bin_edges) - 1

            # Clamp to valid range [0, NUM_BINS-1]
            bin_idx = min(bin_idx, self.NUM_BINS - 1)
            bin_idx = max(bin_idx, 0)

            # 해당 빈에 거래량 누적
            for trade in trades:
                amount = trade.get('amount', 0.0)
                trade_type = trade.get('type')

                if trade_type == 'Buy':
                    buy_hist[bin_idx] += amount
                elif trade_type == 'Sell':
                    sell_hist[bin_idx] += amount

        # 정규화 (reliable_total_vol 우선 사용)
        buy_hist = buy_hist / safe_volume
        sell_hist = sell_hist / safe_volume

        # Hard clipping to [0.0, 1.0] 범위 (수치 폭주 방지)
        buy_hist = np.clip(buy_hist, self.HISTOGRAM_CLIP_MIN, self.HISTOGRAM_CLIP_MAX)
        sell_hist = np.clip(sell_hist, self.HISTOGRAM_CLIP_MIN, self.HISTOGRAM_CLIP_MAX)

        return buy_hist, sell_hist

    def _apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        20개 히스토그램 특성에 rolling robust scaling 적용

        feat_buy_hist_0 ~ feat_buy_hist_9 (10개)
        feat_sell_hist_0 ~ feat_sell_hist_9 (10개)

        Args:
            df: DataFrame with histogram features

        Returns:
            Robust scaled DataFrame (in-place 수정)
        """
        # Phase 1.1: 내부 메서드는 복사하지 않음 (메모리 최적화)
        # 호출자(calculate_features)에서 이미 복사 처리됨

        # 20개 히스토그램 특성에 robust scaling 적용
        for i in range(self.NUM_BINS):
            buy_col = f'feat_buy_hist_{i}'
            sell_col = f'feat_sell_hist_{i}'

            # Robust scaling with shift(1) for temporal causality
            df[buy_col] = self._rolling_robust_scale(df[buy_col])
            df[sell_col] = self._rolling_robust_scale(df[sell_col])

        return df

    def _rolling_robust_scale(self, series: pd.Series) -> pd.Series:
        """
        Rolling window robust scaling with shift(1) for temporal causality

        Phase 2.1: Delegate to shared utility (robust_scaler.py)
        Note: 히스토그램은 추가로 hard clipping 적용 ([-10, 10])

        Args:
            series: 원본 시계열 데이터

        Returns:
            Robust scaled series ([-10, 10] 범위로 hard clipped)
        """
        # Delegate to shared utility
        result = robust_scale(series, self.window_size, self.min_iqr)
        scaled = pd.Series(result, index=series.index)

        # Hard clipping to [-10, 10] 범위 (히스토그램 전용)
        scaled = scaled.clip(lower=self.SCALED_CLIP_MIN, upper=self.SCALED_CLIP_MAX)

        return scaled
