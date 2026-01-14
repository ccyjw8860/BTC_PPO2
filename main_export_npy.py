"""
Export MongoDB data to .npy files for RL training

Usage:
    python main_export_npy.py
"""

from datetime import datetime
import logging
import sys

sys.path.insert(0, 'c:\\python_project\\BTC_PPO2')

from db.dbclass import DBClass
from utils.rl_data.npy_exporter import NpyExporter

# EUC-KR 로깅 설정 (Windows용)
if sys.platform == 'win32':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='euc-kr'
    )

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("MongoDB(input2) → .npy 파일 변환 시작 (RL 학습용)")
    logger.info("=" * 80)

    try:
        # DB 연결
        db = DBClass()
        exporter = NpyExporter(db, batch_size=10000)

        # Export 실행
        result = exporter.export_to_npy(
            train_start=datetime(2022, 6, 9),
            train_end=datetime(2025, 8, 1),
            test_start=datetime(2025, 8, 1),
            test_end=datetime(2025, 10, 31),
            output_dir='data/npy2'
        )

        # 결과 출력
        logger.info("=" * 80)
        logger.info("✅ 변환 완료!")
        logger.info(f"Train X: {result['train_x_shape']}, Train y: {result['train_y_shape']}")
        logger.info(f"Test X: {result['test_x_shape']}, Test y: {result['test_y_shape']}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
