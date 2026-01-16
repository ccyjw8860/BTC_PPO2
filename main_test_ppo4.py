import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Stable Baselines3 관련 라이브러리
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 프로젝트 루트 경로 추가 (필요 시 수정)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.rl_data.trading_env3 import TradingEnv
from utils.rl_data.data_generator import RLDataGenerator
from utils.models.ppo_cnn2 import Custom1DCNN  # 모델 로드 시 필요

def evaluate_and_plot(model_path="model_btc_final.zip", data_dir="data/npy2"):
    """
    학습된 모델을 테스트 데이터에서 실행하고 매매 시점과 자산 곡선을 시각화합니다.
    """
    print(f"🔄 모델 및 데이터 로드 중... (Model: {model_path})")

    # --- 1. 환경 설정 (학습 시와 동일한 파라미터 사용) ---
    SL_OPTS = [0.002, 0.005, 0.01, 0.02]
    TP_OPTS = [0.005, 0.01, 0.02, 0.04]
    WINDOW_SIZE = 1000
    FEE_RATE = 0.0
    SLIPPAGE_RATE = 0.0
    SCALING_FACTOR = 100.0

    def make_env():
        # 테스트 모드 데이터 제너레이터
        data_gen = RLDataGenerator(mode='test', data_dir=data_dir, seq_len=WINDOW_SIZE)
        
        # 테스트 환경 (max_episode_steps=0으로 설정하여 데이터 끝까지 실행)
        env = TradingEnv(
            data_generator=data_gen,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            window_size=WINDOW_SIZE,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
            # scaling_factor=SCALING_FACTOR,
            max_episode_steps=0,  # 전체 데이터 실행
            mode='test'
        )
        return env

    # 벡터 환경 생성
    env = DummyVecEnv([make_env])
    
    # VecNormalize 적용 (학습 시 norm_obs=False였으므로 동일하게 설정)
    # 훈련 모드가 아니므로 training=False, 보상 정규화도 불필요
    env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False, clip_obs=1000.0)

    # --- 2. 모델 로드 ---
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    try:
        # Custom1DCNN 클래스가 정의되어 있어야 로드 가능
        model = PPO.load(model_path, env=env, custom_objects={'Custom1DCNN': Custom1DCNN})
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    # --- 3. 시뮬레이션 실행 ---
    print("🚀 테스트 시뮬레이션 시작...")
    
    obs = env.reset()
    
    # 데이터 기록용 리스트
    prices = []
    equity_curve = []
    
    # 매매 이벤트 기록 (Step, Price)
    long_open_signals = []
    short_open_signals = []
    close_signals = []
    
    done = False
    
    # 실제 환경 객체 접근 (데이터 및 상태 확인용)
    real_env = env.venv.envs[0]
    
    # 이미 처리한 trade step을 추적하여 중복 기록 방지
    last_trade_step = -1 

    while not done:
        # 모델 예측 (Deterministic=True로 설정하여 확률적 요소 제거)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        
        info = infos[0] # VecEnv는 리스트로 반환하므로 첫 번째 환경 정보 가져오기
        
        # --- 데이터 수집 ---
        
        # 1. 현재 가격 가져오기
        # env.current_index는 step() 이후 증가된 상태이므로 -1 인덱스가 현재 step의 데이터
        current_idx = real_env.current_index - 1
        try:
            # RLDataGenerator의 y 데이터(가격)에 직접 접근
            current_price = float(real_env.data_generator.y[current_idx])
        except:
            # 인덱스 에러 등의 경우 안전하게 처리
            current_price = prices[-1] if prices else 0.0

        prices.append(current_price)
        equity_curve.append(info['equity_usd'])
        
        # 2. 매매 이벤트 추적
        trade_info = info.get('last_trade_info')
        
        if trade_info:
            trade_step = trade_info['step']
            
            # 새로운 트레이드 이벤트인 경우에만 기록
            if trade_step != last_trade_step:
                event_type = trade_info['event']
                
                if event_type == 'OPEN':
                    direction = trade_info['type'] # "LONG" or "SHORT"
                    entry_price = trade_info['entry']
                    
                    # 그래프 X축 좌표는 현재 prices 리스트의 마지막 인덱스
                    plot_idx = len(prices) - 1
                    
                    if direction == 'LONG':
                        long_open_signals.append((plot_idx, entry_price))
                    else:
                        short_open_signals.append((plot_idx, entry_price))
                        
                elif event_type == 'CLOSE':
                    exit_price = trade_info['exit']
                    plot_idx = len(prices) - 1
                    close_signals.append((plot_idx, exit_price))
                
                last_trade_step = trade_step

    total_trades = len(long_open_signals) + len(short_open_signals)
    if total_trades > 0:
        long_ratio = len(long_open_signals) / total_trades * 100
        short_ratio = len(short_open_signals) / total_trades * 100
        
        print(f"📊 매매 분석 결과")
        print(f"- 총 진입 횟수: {total_trades}회")
        print(f"- Long 진입: {len(long_open_signals)}회 ({long_ratio:.1f}%)")
        print(f"- Short 진입: {len(short_open_signals)}회 ({short_ratio:.1f}%)")
    else:
        print("매매 기록이 없습니다.")

    # --- 4. 결과 시각화 ---
    print("📊 결과 그래프 생성 중...")
    
    plt.figure(figsize=(16, 10))
    
    # 첫 번째 서브플롯: 가격 및 매매 시점
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Price', color='gray', alpha=0.5, linewidth=1)
    
    # 매매 마커 표시
    if long_open_signals:
        lx, ly = zip(*long_open_signals)
        plt.scatter(lx, ly, marker='^', color='green', s=100, label='Open Long', zorder=5)
        
    if short_open_signals:
        sx, sy = zip(*short_open_signals)
        plt.scatter(sx, sy, marker='v', color='red', s=100, label='Open Short', zorder=5)
        
    if close_signals:
        cx, cy = zip(*close_signals)
        plt.scatter(cx, cy, marker='x', color='blue', s=80, label='Close', zorder=5)
        
    plt.title('Bitcoin Futures Trading Signals (Test Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel('Price (USDT)')
    
    # 두 번째 서브플롯: 자산 곡선 (Equity Curve)
    plt.subplot(2, 1, 2)
    plt.plot(equity_curve, label='Equity (USD)', color='purple', linewidth=1.5)
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial Balance ($10,000)')
    
    # 최종 수익률 표시
    final_equity = equity_curve[-1]
    roi = ((final_equity - 10000) / 10000) * 100
    plt.title(f'Equity Curve (Final: ${final_equity:,.2f}, ROI: {roi:.2f}%)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel('Equity (USD)')
    plt.xlabel('Steps')
    
    plt.tight_layout()
    
    # 파일 저장
    output_file = 'test_trading_results.png'
    plt.savefig(output_file)
    print(f"✅ 결과가 '{output_file}'에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    # 데이터 경로가 다르다면 수정하세요 (예: data_dir="../data/processed")
    evaluate_and_plot(model_path="./checkpoints/checkpoints_CNN_seq1000_20260115_1231/btc_ppo_25600000_steps.zip", data_dir="data/npy2")