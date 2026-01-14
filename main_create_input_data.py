"""
Feature Pipeline for BTCUSDT.P Trading Data (Phase 8: NO Threshold & NO Robust Scaling)

Main entry point for computing 59 features:
- 16 Price features (EMA-based with windows 5, 20, 40, 60, 120) - RAW
- 3 Volume features (vol, oi, cvd) - RAW
- 40 Histogram features (from BTCUSDTP_5Min_Aggregated collection) - RAW
"""

from datetime import datetime
import logging
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, 'c:\\python_project\\BTC_PPO2')

from db.dbclass import DBClass
from utils.whale_threshold.batch_processor import BatchProcessor

# EUC-KR 로깅 설정 (Windows용)
if sys.platform == 'win32':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='euc-kr',
        force=True  # 기존 설정 강제 덮어쓰기
    )

logger = logging.getLogger(__name__)


def main():
    """
    메인 실행 함수 (Phase 8: NO Threshold & NO Robust Scaling)

    2022-06-01부터 2025-10-31까지의 데이터를 처리하여
    59개 특성(Raw Data)을 계산하고 BTCUSDT.P_input2 컬렉션에 저장합니다.

    Features (59 total):
    - 16 Price Features: OHLC (4), EMA (10), BB (2) - RAW (No Scaling)
    - 3 Volume Features: vol, oi, cvd - RAW (No Scaling)
    - 40 Histogram Features: from BTCUSDTP_5Min_Aggregated - RAW (No Scaling)

    Phase 8 Changes:
    - Robust Scaling REMOVED entirely (Raw data passed to VecNormalize in RL)
    - Threshold calculation REMOVED (inherited from Phase 6)
    - Target collection: BTCUSDT.P_input2 (Cleaned and repopulated)

    환경 변수:
    - USE_PARALLEL: 병렬 처리 활성화 (true/false, 기본값: false)
    - MAX_WORKERS: 병렬 프로세스 수 (기본값: CPU 코어 수)
    """
    logger.info("=" * 80)
    logger.info("Feature Pipeline Version 3.1 계산 시작 (Phase 8: NO Robust Scaling)")
    logger.info("=" * 80)
    try:
        # DB 연결
        logger.info("MongoDB 연결 중...")
        db = DBClass()
        logger.info(f"소스 컬렉션: {db.BTCUSDTP_5MinCollection.name}")
        logger.info(f"히스토그램 컬렉션: {db.BTCUSDTP_5Min_Aggregated.name}")
        logger.info(f"타겟 컬렉션: BTCUSDT.P_input2 (Raw Data)")

        # Phase 8: BTCUSDT.P_input2 컬렉션 초기화
        logger.info("")
        logger.info("Phase 8: BTCUSDT.P_input2 컬렉션 초기화 (Drop & Re-create)...")
        logger.info("  ※ Robust Scaling이 제거된 Raw Data로 다시 채웁니다.")
        db.BTCUSDTP_input2.drop()
        logger.info("✓ BTCUSDT.P_input2 컬렉션 삭제 완료")
        logger.info("")

        # BatchProcessor 생성
        # min_iqr 등의 파라미터는 내부적으로 무시되도록 Calculator 수정 필요
        processor = BatchProcessor(
            db=db,
            source_collection=db.BTCUSDTP_5MinCollection,
            target_collection=db.BTCUSDTP_input3,
            aggregated_collection=db.BTCUSDTP_5Min_Aggregated,
            batch_size=1000
        )

        # 처리 기간 설정
        start_date = datetime(2022, 6, 9)
        end_date = datetime(2025, 10, 31)

        logger.info(f"처리 기간: {start_date} ~ {end_date}")
        logger.info(f"Batch Size: 1000 operations")
        logger.info(f"Warm-up Period: 2141 candles (~7.4 days)")
        logger.info(f"Feature Config: 59 features (ALL RAW DATA)")
        logger.info(f"  - Price Features (16): EMA slopes normalized by ATR, but NO Robust Scaling")
        logger.info(f"  - Volume Features (3): Log1p/OI/CVD calculated, but NO Robust Scaling")
        logger.info(f"  - Histogram Features (40): Raw values from aggregated collection")
        logger.info(f"  ※ 중요: Robust Scaling은 강화학습 단계의 VecNormalize가 담당합니다.")
        logger.info("")

        # 병렬 처리 설정
        use_parallel = os.getenv('USE_PARALLEL', 'false').lower() == 'true'
        max_workers = int(os.getenv('MAX_WORKERS', str(os.cpu_count() or 4)))

        logger.info(f"처리 모드: {'병렬' if use_parallel else '순차'}")
        if use_parallel:
            logger.info(f"워커 프로세스: {max_workers}개")
        logger.info("")

        # 전체 범위 처리
        total_saved = processor.process_monthly_range(
            start_date,
            end_date,
            parallel=use_parallel,
            max_workers=max_workers if use_parallel else None
        )

        # 결과 출력
        logger.info("=" * 80)
        logger.info(f"✅ 처리 완료! (Phase 8)")
        logger.info(f"총 {total_saved:,}개의 Raw Feature 문서가 BTCUSDT.P_input2에 저장되었습니다.")
        logger.info(f"  - Document ID 형식: input_YYYYMMDDHHmm")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다.")
        return 1

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)