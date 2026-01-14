"""
Batch Processor for orchestrating whale threshold calculation pipeline

Manages monthly iteration, sliding window state, and bulk MongoDB writes
"""

from datetime import datetime, timedelta
from typing import List
import logging
from pymongo.collection import Collection
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
import pandas as pd

from .mongo_aggregator import MongoAggregator
from .price_calculator import PriceFeatureCalculator
from .volume_calculator import VolumeFeatureCalculator
# Phase 6: Removed threshold-related imports
# from .sliding_window import SlidingWindowBuffer
# from .threshold_calculator import ThresholdCalculator
# from .whale_histogram_calculator import WhaleHistogramCalculator

# EUC-KR 로깅 설정 (Windows용)
import sys
if sys.platform == 'win32':
    logging.basicConfig(encoding='euc-kr')

logger = logging.getLogger(__name__)


# Phase 3.1 & Phase 6: Pure function for parallelization (module-level, picklable)
# Phase 6: Removed threshold calculation logic
def _process_month_pure(
    month_start: datetime,
    month_end: datetime,
    source_collection: Collection,
    lookback_candles: int,
    threshold_warmup: int,  # Deprecated in Phase 6, kept for API compatibility
    batch_size: int,
) -> List[UpdateOne]:
    """
    Pure function for processing a single month (no instance dependencies)

    Phase 6: Removed threshold calculation logic

    Args:
        month_start: Start of month (inclusive)
        month_end: End of month (exclusive)
        source_collection: MongoDB source collection
        lookback_candles: Context candles to fetch (default: 2141)
        threshold_warmup: DEPRECATED in Phase 6 (kept for API compatibility)
        batch_size: Not used in pure function, kept for API compatibility

    Returns:
        List of UpdateOne operations for this month
    """
    logger.info(f"Pure function 월별 처리 시작: {month_start} ~ {month_end}")

    # Create local instances (no shared state)
    aggregator = MongoAggregator(source_collection)
    # Phase 6: Removed threshold-related components
    # window = SlidingWindowBuffer(window_size=288)
    # calculator = ThresholdCalculator()
    price_calc = PriceFeatureCalculator(window_size=2016, min_iqr=0.001)
    volume_calc = VolumeFeatureCalculator(window_size=2016, min_iqr=0.001, cvd_window=2016)
    # histogram_calc = WhaleHistogramCalculator(window_size=2016, min_iqr=0.001)  # Phase 5: Removed

    # Step 1: Context fetching
    context_start = month_start - timedelta(minutes=5 * lookback_candles)
    context_candles = aggregator.aggregate_manually(context_start, month_start)

    # Step 2: Current month data
    month_candles = aggregator.aggregate_manually(month_start, month_end)

    if not month_candles:
        logger.warning(f"데이터 없음: {month_start} ~ {month_end}")
        return []

    # Step 3: Combine context + current month
    all_candles = context_candles + month_candles
    logger.info(f"전체 캔들 수: {len(all_candles)} (context: {len(context_candles)}, month: {len(month_candles)})")

    # Step 4: Convert to DataFrame
    df = _candles_to_dataframe_pure(all_candles)

    # Phase 6: Removed Pass 1 (Threshold calculation) and Pass 2 (Whale volumes calculation)
    context_len = len(context_candles)

    # Step 5-6: Calculate features
    df = price_calc.calculate_features(df, inplace=True)
    df = volume_calc.calculate_features(df, inplace=True)
    # df = histogram_calc.calculate_features(df, inplace=True)  # Phase 5: Removed

    # Step 7 (Phase 5): Fetch histogram features from aggregated collection
    # NOTE: This is a limitation of pure function - cannot access aggregated_collection
    # For now, set histogram features to 0.0
    for i in range(10):
        df[f'feat_buy_hist_shape_{i}'] = 0.0
        df[f'feat_sell_hist_shape_{i}'] = 0.0
        df[f'feat_buy_hist_strength_{i}'] = 0.0
        df[f'feat_sell_hist_strength_{i}'] = 0.0

    # Step 8: Create operations
    operations = []

    for idx in range(context_len, len(df)):
        row = df.iloc[idx]

        # Skip warm-up
        if idx < lookback_candles:
            continue

        candle_datetime = row['datetime']

        # Extract features
        features = _extract_features_pure(row)

        # Phase 6: Create document (NO threshold fields)
        doc = _create_unified_document_pure(candle_datetime, features)

        operation = UpdateOne({'_id': doc['_id']}, {'$set': doc}, upsert=True)
        operations.append(operation)

    logger.info(f"Pure function 월별 처리 완료: {month_start} ~ {month_end}, {len(operations)}개 operations")
    return operations


def _candles_to_dataframe_pure(candles: List[tuple]) -> pd.DataFrame:
    """
    Pure function version of _candles_to_dataframe

    Phase 6 cleanup: Simplified to 5-tuple (whale fields removed)
    """
    rows = []
    for dt, total_trades, ohlcv, open_interest, taker_buy_volume in candles:
        row = {
            'datetime': dt,
            'open': ohlcv['open'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'close': ohlcv['close'],
            'volume': ohlcv['volume'],
            'quote_volume': ohlcv['quote_volume'],
            'trades': total_trades,
            'sum_open_interest_value': open_interest,
            'taker_buy_quote': taker_buy_volume,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


# Phase 6: Removed _calculate_all_whale_volumes_pure (no longer needed)


def _extract_features_pure(row: pd.Series) -> dict:
    """Pure function version of _extract_features"""
    feature_cols = [col for col in row.index if col.startswith('feat_')]
    features = {col: float(row[col]) for col in feature_cols}
    return features


def _create_unified_document_pure(
    candle_datetime: datetime,
    features: dict,
) -> dict:
    """Pure function version of _create_unified_document (Phase 6: NO threshold fields)"""
    # Phase 6: ID 형식 변경 (th_ → input_)
    datetime_str = candle_datetime.strftime('%Y%m%d%H%M')
    doc_id = f"input_{datetime_str}"

    doc = {
        '_id': doc_id,
        'datetime': candle_datetime,
        'created_at': datetime.now(),
        'feature_version': 2,
    }

    doc.update(features)
    return doc


class BatchProcessor:
    """
    Orchestrates end-to-end feature calculation pipeline (Phase 6: NO threshold calculation)

    Features:
    - Monthly batch processing to manage memory
    - Context fetching for feature calculation (2141-candle lookback)
    - Bulk write accumulation for performance (default: 1000 operations)
    - Generates 59 features total: 16 price + 3 volume + 40 histogram (Phase 5)
      * Phase 4: 14 → 16 price features (added EMA40)
      * Phase 3: 5 → 3 volume features (removed whale ratios)
      * Phase 5: 21 → 40 histogram features (from aggregated collection, NO scaling)
      * Phase 6: Removed threshold calculation and whale volume filtering
    """

    # 상수 정의
    # 기존: 2016
    # 변경: 2016 (스케일링용) + 120 (MA 120용) + 5 (여유분) = 2141
    LOOKBACK_CANDLES = 2141 
    THRESHOLD_WARMUP = 288

    def __init__(
        self,
        db,
        source_collection: Collection,
        target_collection: Collection,
        aggregated_collection: Collection = None,
        batch_size: int = 1000,
    ):
        """
        Initialize batch processor

        Args:
            db: Database class instance (DBClass)
            source_collection: Source MongoDB collection (BTCUSDTP_5MinCollection)
            target_collection: Target MongoDB collection (BTCUSDTP_input)
            aggregated_collection: Optional MongoDB collection for histogram features (BTCUSDTP_5Min_Aggregated)
            batch_size: Number of UpdateOne operations to accumulate before bulk_write
        """
        self.db = db
        self.source_collection = source_collection
        self.target_collection = target_collection
        self.aggregated_collection = aggregated_collection  # Phase 5: For histogram features
        self.batch_size = batch_size

        # 기존 컴포넌트
        self.aggregator = MongoAggregator(db, source_collection, aggregated_collection)  # Phase 5: Pass aggregated_collection

        # Phase 6: Removed threshold calculation components
        # self.window = SlidingWindowBuffer(window_size=288)
        # self.calculator = ThresholdCalculator()

        # 새로운 컴포넌트: 가격 특성 계산기
        self.price_calculator = PriceFeatureCalculator(window_size=2016, min_iqr=0.001)

        # 새로운 컴포넌트: 볼륨 특성 계산기
        self.volume_calculator = VolumeFeatureCalculator(
            window_size=2016, min_iqr=0.001, cvd_window=2016
        )

        # 누적 operations
        self.pending_operations: List[UpdateOne] = []
        self.total_saved = 0

    def process_monthly_range(
        self,
        start_date: datetime,
        end_date: datetime,
        parallel: bool = False,
        max_workers: int = None
    ) -> int:
        """
        Process full date range in monthly chunks

        Phase 3.2: 병렬 처리 지원 추가

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (exclusive)
            parallel: If True, use ProcessPoolExecutor for parallel processing
            max_workers: Number of worker processes (default: CPU count)

        Returns:
            Total number of threshold documents saved
        """
        logger.info(f"배치 처리 시작: {start_date} ~ {end_date}, parallel={parallel}")

        # Phase 3.2: Parallel processing
        if parallel:
            return self._process_monthly_range_parallel(start_date, end_date, max_workers)
        else:
            # Sequential processing (original implementation)
            return self._process_monthly_range_sequential(start_date, end_date)

    def _process_monthly_range_sequential(
        self, start_date: datetime, end_date: datetime
    ) -> int:
        """
        Sequential processing (original implementation)

        Maintains sliding window state across month boundaries for accurate
        threshold calculation.
        """
        logger.info(f"순차 처리 시작: {start_date} ~ {end_date}")

        current_date = start_date

        while current_date < end_date:
            # 월 단위 종료일 계산
            if current_date.month == 12:
                month_end = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                month_end = current_date.replace(month=current_date.month + 1, day=1)

            month_end = min(month_end, end_date)

            # 한 달 처리
            self._process_single_month(current_date, month_end)

            current_date = month_end

        # 남은 operations flush
        self._flush_batch()

        logger.info(f"순차 처리 완료: 총 {self.total_saved}개 문서 저장됨")
        return self.total_saved

    def _process_monthly_range_parallel(
        self, start_date: datetime, end_date: datetime, max_workers: int = None
    ) -> int:
        """
        Parallel processing using ProcessPoolExecutor

        Phase 3.2: 병렬 처리 구현
        - 각 월을 독립적으로 처리 (pure function 사용)
        - 모든 operations를 수집한 후 bulk_write

        주의: 2141 candles context는 각 월 처리 시 자동으로 fetching됨
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        # Default max_workers = CPU count
        if max_workers is None:
            max_workers = os.cpu_count() or 4

        logger.info(f"병렬 처리 시작: {start_date} ~ {end_date}, workers={max_workers}")

        # Step 1: Generate month ranges
        month_ranges = []
        current_date = start_date

        while current_date < end_date:
            if current_date.month == 12:
                month_end = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                month_end = current_date.replace(month=current_date.month + 1, day=1)

            month_end = min(month_end, end_date)
            month_ranges.append((current_date, month_end))
            current_date = month_end

        logger.info(f"총 {len(month_ranges)}개 월을 병렬 처리합니다")

        # Step 2: Submit tasks to ProcessPoolExecutor
        all_operations = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_month = {}
            for month_start, month_end in month_ranges:
                future = executor.submit(
                    _process_month_pure,
                    month_start,
                    month_end,
                    self.source_collection,
                    self.LOOKBACK_CANDLES,
                    self.THRESHOLD_WARMUP,
                    self.batch_size,
                )
                future_to_month[future] = (month_start, month_end)

            # Collect results as they complete
            for future in as_completed(future_to_month):
                month_start, month_end = future_to_month[future]
                try:
                    operations = future.result()
                    all_operations.extend(operations)
                    logger.info(f"월 완료: {month_start} ~ {month_end}, {len(operations)}개 operations")
                except Exception as e:
                    logger.error(f"월 처리 실패: {month_start} ~ {month_end}, 오류: {str(e)}", exc_info=True)

        # Step 3: Bulk write all operations
        logger.info(f"총 {len(all_operations)}개 operations을 bulk_write합니다")

        if all_operations:
            # Write in batches
            for i in range(0, len(all_operations), self.batch_size):
                batch = all_operations[i:i + self.batch_size]
                self.pending_operations = batch
                self._flush_batch()

        logger.info(f"병렬 처리 완료: 총 {self.total_saved}개 문서 저장됨")
        return self.total_saved

    def _process_single_month(
        self, month_start: datetime, month_end: datetime
    ) -> None:
        """
        Process a single month with context fetching (Phase 6: Removed threshold logic)

        Steps:
        1. Fetch prior 2141 candles as context (~7.4 days before month_start)
        2. Fetch current month data
        3. Combine into DataFrame
        4. Calculate price features (16 feat_* columns - Phase 4: added EMA40)
        5. Calculate volume features (3 feat_* columns - Phase 3: removed whale ratios)
        6. Fetch histogram features from aggregated collection (40 feat_* columns - Phase 5)
        7. Iterate and create unified documents (59 features total, NO threshold fields)

        Phase 6 Changes:
            - Removed Pass 1: Threshold calculation
            - Removed Pass 2: Whale volume calculation
            - Removed whale-related DataFrame columns

        Args:
            month_start: Start of month (inclusive)
            month_end: End of month (exclusive)
        """
        logger.info(f"월별 처리 시작: {month_start} ~ {month_end}")

        # Step 1: Context fetching (2016 candles = ~7 days before month_start)
        context_start = month_start - timedelta(minutes=5 * self.LOOKBACK_CANDLES)
        context_candles = self.aggregator.aggregate_manually(context_start, month_start)

        # Step 2: Current month data
        month_candles = self.aggregator.aggregate_manually(month_start, month_end)

        if not month_candles:
            logger.warning(f"데이터 없음: {month_start} ~ {month_end}")
            return

        # Step 3: Combine context + current month
        all_candles = context_candles + month_candles
        logger.info(f"전체 캔들 수: {len(all_candles)} (context: {len(context_candles)}, month: {len(month_candles)})")

        # Step 4: Convert to DataFrame
        df = self._candles_to_dataframe(all_candles)

        # Phase 6: Removed threshold calculation (Pass 1 and Pass 2)
        # No longer need threshold or whale_volumes calculation
        context_len = len(context_candles)

        # Step 6: Calculate price features (adds 16 feat_* columns - Phase 4: added EMA40)
        # Phase 1.2: inplace=True로 메모리 최적화 (추가 복사 없음)
        df = self.price_calculator.calculate_features(df, inplace=True)

        # Step 7: Calculate volume features (adds 3 feat_* columns - Phase 3: removed whale ratios)
        df = self.volume_calculator.calculate_features(df, inplace=True)

        # Step 8 (Phase 5): Fetch histogram features from aggregated collection (40 features)
        # NO robust scaling - use raw values from collection
        histogram_features_dict = self.aggregator.fetch_histogram_features(df['datetime'].tolist())

        # Add histogram features to DataFrame (40 columns)
        for i in range(10):
            df[f'feat_buy_hist_shape_{i}'] = 0.0
            df[f'feat_sell_hist_shape_{i}'] = 0.0
            df[f'feat_buy_hist_strength_{i}'] = 0.0
            df[f'feat_sell_hist_strength_{i}'] = 0.0

        # Populate histogram features from fetched data
        for idx, row in df.iterrows():
            dt = row['datetime']
            if dt in histogram_features_dict:
                hist_feats = histogram_features_dict[dt]
                for feat_name, feat_value in hist_feats.items():
                    df.at[idx, f'feat_{feat_name}'] = feat_value

        logger.info("히스토그램 특성 통합 완료: 40개 특성 (NO robust scaling)")

        # Step 9: Iterate and create documents
        # Start from context_len (skip context candles in saving)
        for idx in range(context_len, len(df)):
            row = df.iloc[idx]

            # Check warm-up (need 2141 candles for price features)
            if idx < self.LOOKBACK_CANDLES:
                continue  # Skip until full warm-up

            candle_datetime = row['datetime']

            # Phase 6: Extract all features from row (16 price + 3 volume + 40 histogram = 59)
            features = self._extract_features(row)

            # Phase 6: Create unified document (NO threshold fields)
            doc = self._create_unified_document(candle_datetime, features)

            operation = UpdateOne({'_id': doc['_id']}, {'$set': doc}, upsert=True)
            self.pending_operations.append(operation)

            if len(self.pending_operations) >= self.batch_size:
                self._flush_batch()

        logger.info(f"월별 처리 완료: {month_start} ~ {month_end}")

    def _candles_to_dataframe(self, candles: List[tuple]) -> pd.DataFrame:
        """
        Convert candle tuples to DataFrame (Phase 6: Removed whale volume fields)

        Args:
            candles: List of 5-tuples (datetime, total_trades, ohlcv_dict,
                                       open_interest, taker_buy_volume)

        Returns:
            DataFrame with columns: datetime, OHLCV(6), trades, volume fields(2)
                                   Total: 10 columns (Phase 6 cleanup: whale fields removed)
        """
        rows = []
        for dt, total_trades, ohlcv, open_interest, taker_buy_volume in candles:
            row = {
                'datetime': dt,
                'open': ohlcv['open'],
                'high': ohlcv['high'],
                'low': ohlcv['low'],
                'close': ohlcv['close'],
                'volume': ohlcv['volume'],
                'quote_volume': ohlcv['quote_volume'],
                'trades': total_trades,
                'sum_open_interest_value': open_interest,
                'taker_buy_quote': taker_buy_volume,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    # Phase 6: Removed _calculate_all_whale_volumes method (no longer needed)

    def _extract_features(self, row: pd.Series) -> dict:
        """
        Extract all features (price + volume) from DataFrame row

        Returns:
            Dictionary of {feature_name: value} for all feat_* columns (19 total)
        """
        feature_cols = [col for col in row.index if col.startswith('feat_')]
        features = {col: float(row[col]) for col in feature_cols}
        return features

    def _create_unified_document(
        self,
        candle_datetime: datetime,
        features: dict,
    ) -> dict:
        """
        Create unified document with 59 features (Phase 6: NO threshold fields)

        Document structure:
        {
            '_id': 'input_YYYYMMDDHHmm',
            'datetime': datetime,
            'feat_close_ret': float,
            'feat_body': float,
            ... (16 price features - Phase 4: added EMA40)
            'feat_vol': float,
            'feat_oi': float,
            'feat_cvd': float,
            ... (3 volume features - Phase 3: removed whale ratios)
            'feat_buy_hist_shape_0': float,
            'feat_sell_hist_shape_0': float,
            ... (40 histogram features - Phase 5: from aggregated collection)
            'created_at': datetime,
            'feature_version': 2
        }

        Total: 59 feat_* features (16 price + 3 volume + 40 histogram)

        Phase 6 Changes:
            - Removed 'threshold_99_5' field
            - Removed 'total_trades_window' field
            - Changed ID format from 'th_' to 'input_' prefix

        Args:
            candle_datetime: Candle timestamp
            features: Dictionary of all features (price + volume + histogram)

        Returns:
            Document dictionary
        """
        # Phase 6: ID 형식 변경 (th_ → input_)
        datetime_str = candle_datetime.strftime('%Y%m%d%H%M')
        doc_id = f"input_{datetime_str}"

        doc = {
            '_id': doc_id,
            'datetime': candle_datetime,
            'created_at': datetime.now(),
            'feature_version': 2,  # Feature Version 2 (Iterations 0-7 완료)
        }

        # Add all features (16 price + 3 volume + 40 histogram = 59)
        doc.update(features)

        return doc

    def _flush_batch(self) -> None:
        """
        Flush pending operations to MongoDB using bulk_write

        Uses ordered=False for parallel processing
        """
        if not self.pending_operations:
            return

        try:
            logger.info(f"bulk_write 실행: {len(self.pending_operations)}개 operations")

            result = self.target_collection.bulk_write(
                self.pending_operations, ordered=False
            )

            saved_count = result.upserted_count + result.modified_count
            self.total_saved += saved_count

            logger.info(
                f"bulk_write 완료: {result.upserted_count}개 삽입, "
                f"{result.modified_count}개 수정"
            )

            # operations 초기화
            self.pending_operations = []

        except BulkWriteError as e:
            logger.error(f"bulk_write 실패: {str(e)}", exc_info=True)

            # 부분 성공 처리
            if hasattr(e, 'details'):
                details = e.details
                saved_count = details.get('nInserted', 0) + details.get('nModified', 0)
                self.total_saved += saved_count
                logger.warning(f"부분 성공: {saved_count}개 문서 저장됨")

            # operations 초기화 (실패한 것도 재시도하지 않음)
            self.pending_operations = []
