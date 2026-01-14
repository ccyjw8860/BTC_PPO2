"""
Feature Pipeline for BTCUSDT.P Trading Data (Phase 7: NO Threshold Calculation)

Main entry point for computing 59 features:
- 16 Price features (EMA-based with windows 5, 20, 40, 60, 120)
- 3 Volume features (vol, oi, cvd)
- 40 Histogram features (from BTCUSDTP_5Min_Aggregated collection, NO robust scaling)
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
    메인 실행 함수 (Phase 7: NO Threshold Calculation)

    2022-06-01부터 2025-10-31까지의 데이터를 처리하여
    59개 특성을 계산하고 BTCUSDT.P_input2 컬렉션에 저장합니다.

    Features (59 total):
    - 16 Price Features: OHLC (4), EMA (10), BB (2) with Robust Scaling
    - 3 Volume Features: vol, oi, cvd with Robust Scaling
    - 40 Histogram Features: from BTCUSDTP_5Min_Aggregated (NO robust scaling)

    Phase 7 Changes:
    - Target collection: BTCUSDT.P_input → BTCUSDT.P_input2
    - Old collection BTCUSDT.P_input is preserved (NOT dropped)
    - NO threshold calculation (removed in Phase 6)

    환경 변수:
    - USE_PARALLEL: 병렬 처리 활성화 (true/false, 기본값: false)
    - MAX_WORKERS: 병렬 프로세스 수 (기본값: CPU 코어 수)

    사용 예시:
    - 순차 처리 (기본): python main_create_input_data.py
    - 병렬 처리: set USE_PARALLEL=true && python main_create_input_data.py
    - 워커 수 지정: set MAX_WORKERS=2 && python main_create_input_data.py
    """
    logger.info("=" * 80)
    logger.info("Feature Pipeline Version 3 계산 시작 (Phase 7: NO Threshold)")
    logger.info("=" * 80)
    try:
        # DB 연결
        logger.info("MongoDB 연결 중...")
        db = DBClass()
        logger.info(f"소스 컬렉션: {db.BTCUSDTP_5MinCollection.name}")
        logger.info(f"히스토그램 컬렉션: {db.BTCUSDTP_5Min_Aggregated.name}")
        logger.info(f"타겟 컬렉션: BTCUSDT.P_input2 (NEW - Phase 7)")
        logger.info(f"보존 컬렉션: {db.BTCUSDTP_input.name} (구 버전 보존)")

        # Phase 7: BTCUSDT.P_input2 컬렉션만 삭제 (구 컬렉션은 보존)
        logger.info("")
        logger.info("Phase 7: BTCUSDT.P_input2 컬렉션 삭제 중 (clean slate)...")
        logger.info("  ※ BTCUSDT.P_input 컬렉션은 보존됩니다 (backward compatibility)")
        db.BTCUSDTP_input2.drop()
        logger.info("✓ BTCUSDT.P_input2 컬렉션 삭제 완료")
        logger.info("")

        # BatchProcessor 생성
        processor = BatchProcessor(
            db=db,                                         # Database instance
            source_collection=db.BTCUSDTP_5MinCollection,  # Source: BTCUSDT.P_5Min
            target_collection=db.BTCUSDTP_input2,          # Target: BTCUSDT.P_input2 (Phase 7)
            aggregated_collection=db.BTCUSDTP_5Min_Aggregated,  # Histogram source (Phase 5)
            batch_size=1000                                # 1000 operations per bulk_write
        )

        # 처리 기간 설정
        start_date = datetime(2022, 6, 9)
        end_date = datetime(2025, 10, 31)

        logger.info(f"처리 기간: {start_date} ~ {end_date}")
        logger.info(f"Batch Size: 1000 operations")
        logger.info(f"Warm-up Period: 2141 candles (~7.4 days)")
        logger.info(f"Feature Version 3 적용: 59 features (16 Price + 3 Volume + 40 Histogram)")
        logger.info(f"  Phase 6: Threshold 계산 제거됨 (NO whale_buy_vol, whale_sell_vol)")
        logger.info(f"  Phase 4: MA → EMA 변환 (windows: 5, 20, 40, 60, 120)")
        logger.info(f"  Phase 3: Whale ratio 특성 제거됨 (5 → 3 volume features)")
        logger.info(f"  Phase 5: Histogram 특성 BTCUSDTP_5Min_Aggregated에서 가져옴 (NO robust scaling)")
        logger.info(f"  - Price/Volume Robust Scaling (19 features, min_iqr=0.001, shift(1))")
        logger.info(f"  - Histogram Raw Values (40 features, range [0.0, 1.0])")
        logger.info(f"  - Stationary CVD (rolling window=2016)")
        logger.info(f"  - ATR-normalized EMA slopes")
        logger.info("")

        # 병렬 처리 설정 (환경 변수)
        # MongoDB collection 객체 pickle 문제로 인해 기본값을 false로 설정
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
        logger.info(f"✅ 처리 완료! (Phase 7)")
        logger.info(f"총 {total_saved:,}개의 feature 문서가 BTCUSDT.P_input2에 저장되었습니다.")
        logger.info(f"  - 각 문서: 59 features (16 price + 3 volume + 40 histogram)")
        logger.info(f"  - Document ID 형식: input_YYYYMMDDHHmm")
        logger.info(f"  - 구 컬렉션 BTCUSDT.P_input은 보존되었습니다.")
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
