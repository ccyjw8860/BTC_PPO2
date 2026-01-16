import ccxt
import pandas as pd
import time
from datetime import datetime

def download_ohlcv(symbol, start_str, end_str, timeframe='1d'):
    # 1. Bybit 객체 생성 (무기한 선물을 위해 options 설정)
    # enableRateLimit=True는 API 요청 제한을 자동으로 조절해줍니다.
    exchange = ccxt.bybit({
        'enableRateLimit': True, 
        'options': {
            'defaultType': 'swap',  # swap = perpetual (무기한 선물)
        }
    })

    # ccxt에서 Bybit USDT 무기한 선물의 심볼은 보통 'BTC/USDT:USDT' 입니다.
    # 사용자가 입력한 포맷을 ccxt 포맷으로 매핑
    if symbol == 'BTCUSDT.P':
        ccxt_symbol = 'BTC/USDT:USDT'
    else:
        ccxt_symbol = symbol

    # 2. 날짜를 타임스탬프(ms)로 변환
    since = exchange.parse8601(f'{start_str}T00:00:00Z')
    end_timestamp = exchange.parse8601(f'{end_str}T00:00:00Z')
    
    all_ohlcv = []
    
    print(f"[{symbol}] 데이터 다운로드 시작: {start_str} ~ {end_str}")

    while since < end_timestamp:
        try:
            # 한 번에 가져올 수 있는 캔들 개수 제한 (Bybit는 보통 200~1000개)
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since, limit=1000)
            
            if not ohlcv:
                print("더 이상 가져올 데이터가 없습니다.")
                break
            
            # 마지막 데이터의 시간이 시작 시간과 같으면 무한 루프 방지
            last_timestamp = ohlcv[-1][0]
            if last_timestamp == since:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # 다음 요청을 위해 마지막 캔들 시간 갱신
            since = last_timestamp + 1  # 중복 방지를 위해 약간의 오프셋을 줄 수도 있으나, fetch_ohlcv 로직에 따라 다음 캔들 시간으로 설정
            
            # 진행 상황 출력 (날짜로 변환하여 표시)
            current_date = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d')
            print(f"... {current_date} 까지 다운로드 완료 ({len(all_ohlcv)}개)")
            
            # 목표 날짜를 넘어가면 중단 (단, fetch_ohlcv가 시작점 기준이므로 루프 내부보다 후처리에서 자르는 것이 안전)
            if last_timestamp >= end_timestamp:
                break
                
        except Exception as e:
            print(f"에러 발생: {e}")
            time.sleep(5) # 에러 발생 시 잠시 대기 후 재시도
            continue

    # 3. 데이터프레임 변환 및 정리
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 4. 날짜 범위로 정확히 자르기 (API가 요청 범위보다 더 많은 데이터를 줄 수 있음)
    mask = (df['timestamp'] >= exchange.parse8601(f'{start_str}T00:00:00Z')) & \
           (df['timestamp'] <= exchange.parse8601(f'{end_str}T23:59:59Z'))
    df = df.loc[mask].reset_index(drop=True)
    
    # 보기 좋게 컬럼 정리
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    return df

# --- 실행 설정 ---
SYMBOL = 'BTCUSDT.P'
START_DATE = '2022-06-01'
END_DATE = '2025-10-31'

# 데이터 다운로드 실행
df_result = download_ohlcv(SYMBOL, START_DATE, END_DATE)

# 결과 확인
print("\n--- 다운로드 결과 (상위 5개) ---")
print(df_result.head())
print("\n--- 다운로드 결과 (하위 5개) ---")
print(df_result.tail())

# CSV 저장
filename = f"{SYMBOL}_{START_DATE}_{END_DATE}.csv"
df_result.to_csv(filename, index=False)
print(f"\n파일 저장 완료: {filename}")