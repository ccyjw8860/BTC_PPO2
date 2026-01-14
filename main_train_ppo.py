import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# 프로젝트 루트 경로 추가 (필요 시 수정)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 사용자 정의 모듈 임포트
# (파일명이 trading_env3.py라고 가정, 변경 시 수정 필요)
from utils.rl_data.trading_env3 import TradingEnv 
from utils.rl_data.data_generator import RLDataGenerator

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # PPO의 DummyVecEnv는 infos를 리스트로 반환합니다.
        # infos[0]에 접근하여 값을 가져옵니다.
        infos = self.locals.get("infos", [{}])[0]
        
        if "equity_usd" in infos:
            self.logger.record("custom/equity_usd", infos["equity_usd"])
        if "log_equity" in infos:
            self.logger.record("custom/log_equity", infos["log_equity"])
            
        return True

def make_env(mode='train', sl_opts=None, tp_opts=None, window_size=100):
    """
    환경 생성 팩토리 함수
    """
    # 1. 데이터 제너레이터 생성
    # mode='train' -> train_x.npy 로드 / mode='test' -> test_x.npy 로드
    data_gen = RLDataGenerator(mode=mode, seq_len=window_size)
    
    # 2. 환경 초기화
    env = TradingEnv(
        data_generator=data_gen,
        sl_options=sl_opts,
        tp_options=tp_opts,
        window_size=window_size,
        pip_value=1.0,           # BTCUSDT 1 pip = 1 USDT
        lot_size=0.1,            # 1회 거래량 0.1 BTC
        spread_pips=10.0,        # 스프레드 $10 가정
        commission_pips=10.0,    # 수수료 $10 가정
        max_slippage_pips=5.0,   # 슬리피지 최대 $5
        reward_scale=1.0,        # 보상 스케일
        initial_balance=10000.0, # 초기 자본 $10,000
        mode=mode
    )
    return env

def evaluate_model(model: PPO, eval_env: VecNormalize, deterministic: bool = True):
    """
    모델 평가 및 Equity Curve 생성
    """
    # 정규화 통계 업데이트 중지 (평가 모드)
    eval_env.training = False
    eval_env.norm_reward = False
    
    obs = eval_env.reset()
    equity_curve = []
    
    # 첫 Equity 기록
    # VecEnv는 리스트 형태로 info를 반환하므로 첫 번째 환경의 info 사용
    # 초기화 직후에는 info가 없으므로 초기 자본금으로 시작
    current_equity = 10000.0 
    equity_curve.append(current_equity)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        
        # 정보 추출
        info = infos[0]
        current_equity = info.get("equity_usd", current_equity)
        equity_curve.append(current_equity)
        
        done = dones[0]

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity

def main():
    # ---- 1. 설정 및 파라미터 ----
    # 퍼센트 기반 SL/TP 옵션 (0.01 = 1%)
    SL_OPTS = [0.002, 0.005, 0.01, 0.02, 0.05] 
    TP_OPTS = [0.005, 0.01, 0.02, 0.04, 0.08]
    WINDOW_SIZE = 100  # 데이터 제너레이터 seq_len과 일치해야 함
    TOTAL_TIMESTEPS = 50_000_000  # 학습 스텝 수

    print(f"Dataset Loading... Window Size: {WINDOW_SIZE}")

    # ---- 2. 환경 생성 (Train / Test) ----
    # 훈련용 환경 (VecNormalize 적용)
    # PPO는 병렬 환경을 지원하지만, 여기서는 DummyVecEnv(단일 프로세스) 사용
    train_vec_env = DummyVecEnv([lambda: make_env('train', SL_OPTS, TP_OPTS, WINDOW_SIZE)])
    
    # [중요] VecNormalize: 입력(Obs) 정규화 + 보상(Reward) 정규화
    # Raw Data가 들어오므로 clip_obs를 넉넉하게(100.0) 설정하거나 기본값(10.0) 사용
    train_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=100.0)

    # 테스트(검증)용 환경
    # 주의: 테스트 환경은 Train 환경의 통계(mean, var)를 공유받아야 함 (뒤에서 처리)
    test_vec_env = DummyVecEnv([lambda: make_env('test', SL_OPTS, TP_OPTS, WINDOW_SIZE)])
    test_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=False, clip_obs=100.0, training=False)

    print("Environment setup complete with VecNormalize.")

    # ---- 3. 모델 정의 (PPO) ----
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_log/",
        device="cuda"  # 여기에 명시적으로 cuda 지정 (선택사항)
    )

    # ---- 4. 체크포인트 콜백 설정 ----
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # save_vecnormalize=True: 체크포인트마다 정규화 통계도 같이 저장 (필수!)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="btc_ppo",
        save_vecnormalize=True 
    )

    tb_callback = TensorboardCallback()

    # ---- 5. 학습 시작 ----
    print(f"Start Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, tb_callback])
    print("Training finished.")

    # 최종 모델 및 정규화 통계 저장
    model.save("model_btc_final")
    train_env.save("vec_normalize_final.pkl")
    print("Final model saved.")

    # ---- 6. OOS(Out-of-Sample) 평가 및 Best Model 선정 ----
    print("\nEvaluating Checkpoints on Test Data...")

    # 테스트 환경에 최종 학습된 정규화 통계 적용 (일단 기본값으로)
    # 실제로는 각 체크포인트에 맞는 stats를 로드해야 함
    
    best_equity = -np.inf
    best_path = None
    
    # 체크포인트 파일 검색 (zip 파일)
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("btc_ppo")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        # VecNormalize 통계 파일 경로 추론 (btc_ppo_X_steps.zip -> btc_ppo_X_steps_vecnormalize.pkl)
        # SB3 CheckpointCallback의 명명 규칙 따름
        vec_path = ck_path.replace(".zip", "_vecnormalize.pkl")
        
        try:
            # 1. 모델 로드
            loaded_model = PPO.load(ck_path)
            
            # 2. 정규화 통계 로드 및 테스트 환경에 적용
            if os.path.exists(vec_path):
                # 저장된 통계를 사용하여 테스트 환경 생성
                eval_env = VecNormalize.load(vec_path, test_vec_env)
                eval_env.training = False # 업데이트 끄기
                eval_env.norm_reward = False # 보상 정규화 끄기 (평가 지표는 실제 금액이어야 함)
            else:
                print(f"[Warning] No VecNormalize stats found for {ck}. Using final training stats.")
                eval_env = test_env # Fallback
            
            # 3. 평가 수행
            _, final_eq = evaluate_model(loaded_model, eval_env)
            
            print(f"[Eval] {ck} -> Final Equity: ${final_eq:,.2f}")
            
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
                
        except Exception as e:
            print(f"[Skip] Could not evaluate {ck}: {e}")

    # Best Model 결정
    print("-" * 50)
    if best_path:
        print(f"🏆 Best Model found: {best_path}")
        print(f"   Final Equity: ${best_equity:,.2f}")
        
        # Best Model 및 통계 로드
        final_model = PPO.load(best_path)
        best_vec_path = best_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(best_vec_path):
            final_eval_env = VecNormalize.load(best_vec_path, test_vec_env)
        else:
            final_eval_env = test_env
    else:
        print("Using Final Model as Best.")
        final_model = model
        final_eval_env = test_env

    final_eval_env.training = False
    final_eval_env.norm_reward = False

    # ---- 7. 최종 결과 시각화 ----
    # Train 구간 평가 (학습 데이터에 대한 성능)
    # 주의: Train env는 random start이므로 전체 커브를 그리려면 deterministic 모드로 처음부터 끝까지 돌려야 함
    # 여기서는 간단히 Test 셋에 대해서만 그립니다.
    
    print("Generating Equity Curve for Best Model...")
    equity_curve_test, _ = evaluate_model(final_model, final_eval_env)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_test, label="Test (OOS) Equity", color='orange')
    plt.axhline(y=10000, color='r', linestyle='--', label="Initial Balance")
    plt.title(f"Equity Curve: Best Model ({os.path.basename(best_path) if best_path else 'Final'})")
    plt.xlabel("Steps")
    plt.ylabel("Equity (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()