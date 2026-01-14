import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn

# 프로젝트 루트 경로 추가 (필요 시 수정)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 사용자 정의 모듈 임포트
# [주의] trading_env3.py 파일명을 사용합니다.
from utils.rl_data.trading_env3 import TradingEnv 
from utils.rl_data.data_generator import RLDataGenerator

# 1. 커스텀 1D CNN 클래스 정의
class Custom1DCNN(BaseFeaturesExtractor):
    """
    Time Series용 1D CNN Feature Extractor
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 입력 차원: (Batch, Seq_Len, N_Features) -> (Batch, N_Features, Seq_Len)으로 변환해 처리
        n_input_channels = observation_space.shape[1] # 59개 Feature
        
        self.cnn = nn.Sequential(
            # Layer 1: 세밀한 특징 추출
            nn.Conv1d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 100 -> 50
            
            # Layer 2: 조금 더 큰 패턴 추출
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 50 -> 25
            
            # Layer 3: 추세 등 거시적 특징 추출
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 25 -> 12
            
            nn.Flatten(),
        )

        # CNN 통과 후 차원 계산
        with th.no_grad():
            # 더미 데이터 생성 (Batch=1, Seq=100, Feat=59)
            sample = th.as_tensor(observation_space.sample()[None]).float()
            # PyTorch Conv1d는 (Batch, Channel, Length) 순서를 원함 -> Permute
            sample = sample.permute(0, 2, 1)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # PPO 입력 (Batch, Seq, Feat) -> (Batch, Feat, Seq)로 순서 변경
        x = observations.permute(0, 2, 1)
        x = self.cnn(x)
        return self.linear(x)


# -----------------------------------------------------------------------------
# 1. Custom Callback for TensorBoard (Train Log)
# -----------------------------------------------------------------------------
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    학습(Train) 중의 Equity, Log Equity 등을 기록합니다.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # PPO의 DummyVecEnv는 infos를 리스트로 반환합니다.
        # infos[0]에 접근하여 첫 번째 환경의 값을 기록합니다.
        infos = self.locals.get("infos", [{}])[0]
        
        if "equity_usd" in infos:
            self.logger.record("custom/equity_usd", infos["equity_usd"])
        if "log_equity" in infos:
            self.logger.record("custom/log_equity", infos["log_equity"])
            
        return True

# -----------------------------------------------------------------------------
# 2. Custom Callback for Evaluation (Eval Log) - 🟢 [신규 추가]
# -----------------------------------------------------------------------------
class CustomEvalCallback(BaseCallback):
    """
    주기적으로 Test 환경에서 모델을 평가하고, Final Equity를 TensorBoard에 기록하는 콜백
    """
    def __init__(self, eval_env, eval_freq=10000, deterministic=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        # eval_freq 주기마다 평가 수행
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # evaluate_model 함수를 재사용하여 평가 수행
            # (eval_env는 VecNormalize가 적용된 상태여야 함)
            equity_curve, final_equity = evaluate_model(self.model, self.eval_env, self.deterministic)
            
            # TensorBoard 기록 (eval 탭에 표시됨)
            self.logger.record("eval/final_equity_usd", final_equity)
            self.logger.record("eval/log_final_equity", np.log(max(final_equity, 1e-6)))
            
            # 콘솔 출력
            if self.verbose > 0:
                print(f"[CustomEval] Step {self.num_timesteps}: Final Equity = ${final_equity:,.2f}")
                
        return True

# -----------------------------------------------------------------------------
# 3. Environment Factory
# -----------------------------------------------------------------------------
def make_env(mode='train', sl_opts=None, tp_opts=None, window_size=100, max_episode_steps=2048):
    """
    환경 생성 팩토리 함수 (Binance 수수료 적용 버전)
    """
    # 1. 데이터 제너레이터 생성
    # mode='train' -> train_x.npy 로드 / mode='test' -> test_x.npy 로드
    data_gen = RLDataGenerator(mode=mode, seq_len=window_size)
    
    # 2. 환경 초기화
    # fee_rate=0.0005 (0.05%) 적용
    env = TradingEnv(
        data_generator=data_gen,
        sl_options=sl_opts,
        tp_options=tp_opts,
        window_size=window_size,
        pip_value=1.0,           # BTCUSDT 1 pip = 1 USDT
        lot_size=0.1,            # 1회 거래량 0.1 BTC
        spread_pips=10.0,        # 스프레드 비용 ($10 가정)
        fee_rate=0.0005,         # 거래 대금의 0.05% (Binance Taker)
        max_slippage_pips=5.0,   # 슬리피지 최대 $5
        reward_scale=1.0,        # 보상 스케일
        initial_balance=10000.0, # 초기 자본 $10,000
        max_episode_steps=max_episode_steps, # 에피소드 길이 제한
        mode=mode
    )
    return env

# -----------------------------------------------------------------------------
# 4. Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_model(model: PPO, eval_env: VecNormalize, deterministic: bool = True):
    """
    모델 평가 및 Equity Curve 생성
    """
    # 정규화 통계 업데이트 중지 (평가 모드)
    # 매우 중요: Test 시에는 학습 데이터의 통계(mean, var)를 고정해서 사용해야 함
    eval_env.training = False
    eval_env.norm_reward = False
    
    obs = eval_env.reset()
    equity_curve = []
    
    # 첫 Equity 기록 (초기 자본금)
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
    
    # 평가가 끝나면 다시 Training 모드로 복구할 필요는 없음 (DummyVecEnv가 별도 객체이므로)
    # 하지만 만약 동일 환경을 쓴다면 복구해야 함. 여기선 별도 eval_env를 쓰므로 괜찮음.
    
    return equity_curve, final_equity

# -----------------------------------------------------------------------------
# 5. Main Training Loop
# -----------------------------------------------------------------------------
def main():
    # ---- A. 하이퍼파라미터 설정 ----
    # 퍼센트 기반 SL/TP 옵션 (0.01 = 1%)
    SL_OPTS = [0.002, 0.005, 0.01, 0.02] 
    TP_OPTS = [0.005, 0.01, 0.02, 0.04]
    
    WINDOW_SIZE = 100          # 데이터 제너레이터 seq_len과 일치해야 함
    
    # [설정] 멀티 환경 및 에피소드 길이
    NUM_ENVS = 8               # 병렬 환경 개수 (CPU 코어 수에 맞춰 4~16 권장)
    EPISODE_LENGTH = 2048      # 에피소드 길이 (PPO n_steps와 일치 권장)
    
    TOTAL_TIMESTEPS = 50_000_000  # 총 학습 스텝 수
    ENT_COEF = 0.1
    GAMMA = 0.99
    GAE_LAMBDA = 0.9
    N_EPOCHS = 3
    CHECKPOINT_DIR = "./checkpoints_CNN"
    BATCH_SIZE = 2048

    print(f"Dataset Loading... Window Size: {WINDOW_SIZE}")
    print(f"Configuration: {NUM_ENVS} Envs, {EPISODE_LENGTH} Max Steps, Fee Rate: 0.05%")

    # ---- B. 환경 생성 (Train / Test) ----
    # 훈련용: 여러 개의 환경 생성 (List Comprehension) -> DummyVecEnv
    # (각 환경은 독립적인 랜덤 시작점을 가짐)
    env_fns = [lambda: make_env('train', SL_OPTS, TP_OPTS, WINDOW_SIZE, EPISODE_LENGTH) for _ in range(NUM_ENVS)]
    train_vec_env = DummyVecEnv(env_fns)
    
    # [중요] VecNormalize: 입력 정규화 + 보상 정규화 (학습용)
    # Raw Data가 들어오므로 clip_obs를 넉넉하게 설정
    train_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=100.0)

    # 테스트용: 단일 환경 (검증용)
    # EvalCallback에서 사용할 환경
    test_vec_env = DummyVecEnv([
            lambda: make_env(
                'test', 
                SL_OPTS, 
                TP_OPTS, 
                WINDOW_SIZE, 
                max_episode_steps=0  # 🟢 0으로 설정하면 중단 없이 끝까지 갑니다.
            )
        ])
    # 테스트 환경은 학습 환경의 통계(mean, var)를 공유받지 않고 시작하되, 
    # 실제 평가 시에는 로드된 통계를 덮어씌울 예정입니다.
    # 여기서는 일단 초기화합니다.
    test_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=False, clip_obs=100.0, training=False)

    print("Environment setup complete with VecNormalize.")

    # 2. 정책 키워드(policy_kwargs) 설정
    # MLP 대신 위에서 만든 Custom1DCNN을 사용한다고 명시
    POLICY_KWARGS = dict(
        features_extractor_class=Custom1DCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])] # CNN 뒤에 붙는 판단용 MLP
    )

    # ---- C. 모델 정의 (PPO) ----
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=POLICY_KWARGS,
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=EPISODE_LENGTH,  # 에피소드 길이와 맞춤 (버퍼 최적화)
        batch_size=BATCH_SIZE,         # 배치 사이즈 (메모리에 따라 64~4096 조절 가능)
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=0.2,
        ent_coef=ENT_COEF,           # 탐색을 위한 엔트로피 계수
        tensorboard_log="./tensorboard_log/",
        device="cuda"            # GPU 사용 명시
    )

    # ---- D. 콜백 설정 ----
    ckpt_dir = CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. CheckpointCallback: 모델 저장
    # save_freq는 전체 스텝 기준이므로 환경 수로 나누어 줍니다.
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_ENVS,
        save_path=ckpt_dir,
        name_prefix="btc_ppo",
        save_vecnormalize=True   # 정규화 통계 저장 필수!
    )
    
    # 2. TensorboardCallback: 학습 로그(Train Equity) 기록
    tb_callback = TensorboardCallback()

    # 3. 🟢 [신규] CustomEvalCallback: 평가 로그(Eval Equity) 기록
    # 체크포인트 저장 주기와 맞춰서 5만 스텝마다 평가 수행
    eval_callback = CustomEvalCallback(
        eval_env=test_env,
        eval_freq=100_000 // NUM_ENVS, 
        deterministic=True,
        verbose=1
    )
    
    # 콜백 리스트 병합
    callback_list = CallbackList([checkpoint_callback, tb_callback, eval_callback])

    # ---- E. 학습 시작 ----
    print(f"Start Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
    print("Training finished.")

    # 최종 모델 및 정규화 통계 저장
    model.save("model_btc_final")
    train_env.save("vec_normalize_final.pkl")
    print("Final model saved.")

    # ---- F. OOS(Out-of-Sample) 평가 및 Best Model 선정 ----
    print("\nEvaluating Checkpoints on Test Data...")
    
    best_equity = -np.inf
    best_path = None
    
    # 체크포인트 파일 검색
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("btc_ppo")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        # VecNormalize 통계 파일 경로 추론
        vec_path = ck_path.replace(".zip", "_vecnormalize.pkl")
        
        try:
            # 1. 모델 로드
            loaded_model = PPO.load(ck_path)
            
            # 2. 정규화 통계 로드 및 테스트 환경에 적용
            if os.path.exists(vec_path):
                # 저장된 통계를 사용하여 테스트 환경 생성
                eval_env = VecNormalize.load(vec_path, test_vec_env)
                eval_env.training = False # 업데이트 끄기
                eval_env.norm_reward = False # 보상 정규화 끄기
            else:
                # 통계 파일이 없으면 최종 학습 통계 사용 (Fallback)
                print(f"[Warning] No VecNormalize stats found for {ck}. Using final training stats.")
                eval_env = test_env 
            
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

    # ---- G. 최종 결과 시각화 (Linear & Log Scale) ----
    print("Generating Equity Curve for Best Model...")
    equity_curve_test, _ = evaluate_model(final_model, final_eval_env)

    # Subplot 생성: 위쪽은 Linear, 아래쪽은 Log Scale
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 1. Linear Scale Plot
    ax1.plot(equity_curve_test, label="Test Equity (Linear)", color='orange')
    ax1.axhline(y=10000, color='r', linestyle='--', label="Initial Balance")
    ax1.set_title(f"Equity Curve: Best Model (Linear Scale) - {os.path.basename(best_path) if best_path else 'Final'}")
    ax1.set_ylabel("Equity (USDT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Log Scale Plot
    ax2.plot(equity_curve_test, label="Test Equity (Log)", color='green')
    ax2.axhline(y=10000, color='r', linestyle='--', label="Initial Balance")
    ax2.set_yscale('log')  # Y축 로그 스케일 설정
    ax2.set_title(f"Equity Curve: Best Model (Log Scale)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Equity (Log Scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both") # 세부 눈금 표시

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()