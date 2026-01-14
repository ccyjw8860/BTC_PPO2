"""
NPY Exporter for RL Training Data

Exports 40 features from MongoDB to .npy files for reinforcement learning
"""

from datetime import datetime
from typing import List, Tuple
import logging
import sys
import os
import pandas as pd
import numpy as np

from db.dbclass import DBClass

# EUC-KR 로깅 설정 (Windows용)
if sys.platform == 'win32':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='euc-kr'
    )

logger = logging.getLogger(__name__)


class NpyExporter:
    def __init__(self, db: DBClass, batch_size: int = 10000):
        self.db = db
        self.batch_size = batch_size
        # 요구사항 1: BTCUSDTP_input2 컬렉션 사용
        self.input_collection = db.BTCUSDTP_input2 
        self.price_collection = db.BTCUSDTP_5MinCollection
        logger.info(f"NpyExporter v3 초기화 (Target: {self.input_collection.name}, batch_size={batch_size})")

    def _create_output_directory(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"출력 디렉토리 준비 완료: {output_dir}")

    def _fetch_batch(self, start_date: datetime, end_date: datetime, skip: int, limit: int) -> List[dict]:
        # 요구사항: _id, created_at, feature_version 제외
        cursor = self.input_collection.find(
            {'datetime': {'$gte': start_date, '$lt': end_date}},
            {'_id': 0, 'created_at': 0, 'feature_version': 0}
        ).sort('datetime', 1).skip(skip).limit(limit)
        return list(cursor)

    def _join_collections(self, input_docs: List[dict]) -> pd.DataFrame:
        if not input_docs: return pd.DataFrame()
        datetimes = [doc['datetime'] for doc in input_docs]
        # 요구사항 2: 원본 close 데이터 조인
        price_cursor = self.price_collection.find(
            {'datetime': {'$in': datetimes}},
            {'datetime': 1, 'close': 1, '_id': 0}
        )
        price_dict = {doc['datetime']: doc['close'] for doc in price_cursor}
        for doc in input_docs:
            doc['close'] = price_dict.get(doc['datetime'], None)
        return pd.DataFrame(input_docs)

    def _extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # 요구사항 1: 피처 개수 자동 수용 (feat_로 시작하는 모든 컬럼)
        feature_cols = sorted([col for col in df.columns if col.startswith('feat_')])
        
        # 요구사항 5: NaN 및 Inf 포함 시 해당 datetime 로그 출력 및 삭제
        # Inf를 NaN으로 치환
        df = df.replace([np.inf, -np.inf], np.nan)
        mask = df[feature_cols + ['close']].isna().any(axis=1)
        nan_rows = df[mask]
        
        if not nan_rows.empty:
            dropped_dts = nan_rows['datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()
            logger.warning(f"⚠️ 결측치 발견으로 {len(nan_rows)}개 행 제거됨. 대상 시점: {dropped_dts}")

        df_clean = df[~mask]
        X = df_clean[feature_cols].values.astype(np.float32)
        y = df_clean['close'].values.astype(np.float32)
        return X, y

    def _save_npy_arrays(self, X: np.ndarray, y: np.ndarray, prefix: str, output_dir: str) -> None:
        x_path = os.path.join(output_dir, f'{prefix}_x.npy')
        y_path = os.path.join(output_dir, f'{prefix}_y.npy')
        np.save(x_path, X)
        np.save(y_path, y)
        logger.info(f"저장 완료: {prefix} (Features: {X.shape[1]}개, Samples: {X.shape[0]}개)")

    def export_to_npy(self, train_start: datetime, train_end: datetime, test_start: datetime, test_end: datetime, output_dir: str) -> dict:
        self._create_output_directory(output_dir)
        
        # Train 데이터 변환
        logger.info(f"Train 변환: {train_start} ~ {train_end}")
        train_X, train_y = self._export_range(train_start, train_end)
        self._save_npy_arrays(train_X, train_y, 'train', output_dir)
        
        # Test 데이터 변환
        logger.info(f"Test 변환: {test_start} ~ {test_end}")
        test_X, test_y = self._export_range(test_start, test_end)
        self._save_npy_arrays(test_X, test_y, 'test', output_dir)

        # KeyError 해결을 위해 y_shape 정보 포함하여 반환
        return {
            'train_x_shape': train_X.shape,
            'train_y_shape': train_y.shape,
            'test_x_shape': test_X.shape,
            'test_y_shape': test_y.shape,
        }

    def _export_range(self, start_date: datetime, end_date: datetime) -> Tuple[np.ndarray, np.ndarray]:
        X_batches, y_batches = [], []
        skip = 0
        while True:
            batch_docs = self._fetch_batch(start_date, end_date, skip, self.batch_size)
            if not batch_docs: break
            df = self._join_collections(batch_docs)
            X_batch, y_batch = self._extract_features_and_labels(df)
            X_batches.append(X_batch)
            y_batches.append(y_batch)
            skip += self.batch_size
        
        if X_batches:
            return np.vstack(X_batches), np.concatenate(y_batches)
        return np.array([], dtype=np.float32).reshape(0, 0), np.array([], dtype=np.float32)
