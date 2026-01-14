"""
RL Data Generator for Memory-Mapped .npy Files

Provides efficient sequential access to training data for reinforcement learning
"""

from typing import Tuple
import logging
import sys
import os
import numpy as np

# EUC-KR 로깅 설정 (Windows용)
if sys.platform == 'win32':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='euc-kr'
    )

logger = logging.getLogger(__name__)


class RLDataGenerator:
    """
    RL 학습용 데이터 제너레이터 (Memory-Mapped)
    - 피처 차원을 동적으로 인식하도록 개선됨 (40 -> 59 등)
    """

    def __init__(self, mode='train', data_dir='data/npy2', seq_len=100):
        # 파라미터 검증
        if mode not in ['train', 'test']:
            raise ValueError("mode는 'train' 또는 'test'여야 합니다")

        if seq_len < 1:
            raise ValueError("seq_len은 1 이상이어야 합니다")

        # 파일 경로 설정 (데이터셋 경로 npy2 확인)
        x_path = os.path.join(data_dir, f'{mode}_x.npy')
        y_path = os.path.join(data_dir, f'{mode}_y.npy')

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Feature 파일 없음: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Price 파일 없음: {y_path}")

        # Memory-mapped loading
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')

        if len(self.X) != len(self.y):
            raise ValueError(f"X와 y의 길이 불일치: {len(self.X)} vs {len(self.y)}")

        # ✅ 수정 포인트: 하드코딩된 '40' 검증 제거 및 동적 차원 할당
        self.feature_dim = self.X.shape[1]
        self.seq_len = seq_len
        self.mode = mode

        logger.info(f"RLDataGenerator 초기화: mode={mode}, seq_len={seq_len}, feature_dim={self.feature_dim}")

    def get_num_samples(self) -> int:
        return len(self.y) - self.seq_len

    def get_feature_dim(self) -> int:
        # ✅ 저장된 차원을 반환
        return self.feature_dim

    def reset(self) -> int:
        return self.seq_len - 1

    def get_sequence(self, index: int) -> Tuple[np.ndarray, float, float]:
        min_idx = self.seq_len - 1
        max_idx = len(self.y) - 2

        if index < min_idx or index > max_idx:
            raise IndexError(f"Index {index} out of range")

        start_idx = index - self.seq_len + 1
        end_idx = index + 1

        # PyTorch/TF 연동 시 에러 방지를 위한 .copy()
        state = self.X[start_idx:end_idx].copy()
        current_price = float(self.y[index])
        next_price = float(self.y[index + 1])

        return state, current_price, next_price
