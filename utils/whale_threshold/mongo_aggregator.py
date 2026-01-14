"""
MongoDB Aggregator for fetching candle data (Phase 6 cleanup: whale logic removed)

Fetches OHLCV and volume data from BTCUSDT.P_5Min collection
"""

from datetime import datetime
from typing import List, Dict
import logging
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
import pandas as pd

# EUC-KR 로깅 설정 (Windows용)
import sys
if sys.platform == 'win32':
    logging.basicConfig(encoding='euc-kr')

logger = logging.getLogger(__name__)


class MongoAggregator:
    """
    Aggregates candle data from MongoDB (Phase 6 cleanup: whale logic removed)

    Fetches OHLCV + volume data for feature pipeline
    """

    def __init__(self, db, collection: Collection, aggregated_collection: Collection = None, max_retries: int = 3):
        """
        Initialize MongoDB aggregator

        Args:
            db: Database class instance (DBClass)
            collection: MongoDB collection (BTCUSDTP_5MinCollection)
            aggregated_collection: Optional MongoDB collection for histogram features (BTCUSDTP_5Min_Aggregated)
            max_retries: Maximum retry attempts on transient failures
        """
        self.db = db
        self.collection = collection
        self.aggregated_collection = self.db.BTCUSDTP_5Min_Aggregated  # Phase 5: For histogram features
        self.max_retries = max_retries

    def aggregate_manually(self, start_date: datetime, end_date: datetime):
        """
        5분봉 데이터 조회 (Phase 6 cleanup: whale processing removed)

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (exclusive)

        Returns:
            List of 5-tuples: (datetime, total_trades, ohlcv_dict,
                              open_interest, taker_buy_volume)
        """
        # 1. 5분봉 데이터 가져오기 (OHLCV + volume fields만 필요)
        candles_cursor = self.db.BTCUSDTP_5MinCollection.find(
            {"datetime": {"$gte": start_date, "$lt": end_date}},
            {"_id": 0}  # 모든 필드 포함
        )
        logger.info("5분봉 데이터 가져오기 완료. DataFrame 변환중")
        df_candles = pd.DataFrame(list(candles_cursor))
        if df_candles.empty:
            return []

        df_candles.set_index('datetime', inplace=True)

        # 결과를 5-tuple 형태로 변환하여 반환
        return self._convert_to_original_format(df_candles)

    def _convert_to_original_format(self, df_candles: pd.DataFrame):
        """
        DataFrame을 5-tuple 형식으로 변환 (Phase 6 cleanup: whale fields removed)

        Returns:
            List of 5-tuples: (datetime, total_trades, ohlcv_dict,
                              open_interest, taker_buy_volume)
        """
        results = []

        for dt, row in df_candles.iterrows():
            # OHLCV dict 생성
            ohlcv = {
                'open': row.get('open', 0.0),
                'high': row.get('high', 0.0),
                'low': row.get('low', 0.0),
                'close': row.get('close', 0.0),
                'volume': row.get('volume', 0.0),
                'quote_volume': row.get('quote_volume', 0.0),
            }

            results.append((
                dt,  # datetime
                row.get('trades', 0),  # total_trades
                ohlcv,  # ohlcv_dict
                row.get('sum_open_interest_value', 0.0),  # open_interest
                row.get('taker_buy_quote', 0.0),  # taker_buy_volume
            ))

        return results

    def fetch_histogram_features(self, datetimes: List[datetime]) -> Dict[datetime, Dict[str, float]]:
        """
        BTCUSDTP_5Min_Aggregated collection에서 히스토그램 특성 조회 (Phase 5)

        Args:
            datetimes: 조회할 datetime 리스트

        Returns:
            Dict[datetime, Dict[str, float]]: datetime → {feature_name: value} 매핑
            예: {datetime(2023,1,1,0,0): {'buy_hist_shape_0': 0.1, 'sell_hist_shape_0': 0.2, ...}}

        Note:
            - 40개 필드: buy_hist_shape_0~9, sell_hist_shape_0~9, buy_hist_strength_0~9, sell_hist_strength_0~9
            - NO robust scaling (collection에 저장된 그대로 사용)
        """
        if self.aggregated_collection is None:
            logger.warning("aggregated_collection이 설정되지 않았습니다. 빈 히스토그램 반환")
            return {}

        if not datetimes:
            return {}

        # MongoDB에서 datetime으로 batch 조회
        query = {'datetime': {'$in': datetimes}}
        projection = {'_id': 0, 'datetime': 1}

        # 40개 히스토그램 필드 projection
        for i in range(10):
            projection[f'buy_hist_shape_{i}'] = 1
            projection[f'sell_hist_shape_{i}'] = 1
            projection[f'buy_hist_strength_{i}'] = 1
            projection[f'sell_hist_strength_{i}'] = 1

        try:
            cursor = self.aggregated_collection.find(query, projection)
            results = {}

            for doc in cursor:
                dt = doc['datetime']
                features = {}

                # 40개 필드 추출
                for i in range(10):
                    features[f'buy_hist_shape_{i}'] = doc.get(f'buy_hist_shape_{i}', 0.0)
                    features[f'sell_hist_shape_{i}'] = doc.get(f'sell_hist_shape_{i}', 0.0)
                    features[f'buy_hist_strength_{i}'] = doc.get(f'buy_hist_strength_{i}', 0.0)
                    features[f'sell_hist_strength_{i}'] = doc.get(f'sell_hist_strength_{i}', 0.0)

                results[dt] = features

            logger.info(f"히스토그램 특성 조회 완료: {len(results)}개 datetime")
            return results

        except PyMongoError as e:
            logger.error(f"히스토그램 특성 조회 실패: {str(e)}")
            return {}
