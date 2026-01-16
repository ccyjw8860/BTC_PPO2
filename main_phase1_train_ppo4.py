import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from utils.models.ppo_cnn2 import Custom1DCNN

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.rl_data.trading_env3 import TradingEnv 
from utils.rl_data.data_generator import RLDataGenerator
from datetime import datetime

def get_current_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M")

# -----------------------------------------------------------------------------
# Callbacks (기존과 동일)
# -----------------------------------------------------------------------------
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        if "equity_usd" in infos:
            self.logger.record("custom/equity_usd", infos["equity_usd"])
        if "log_equity" in infos:
            self.logger.record("custom/log_equity", infos["log_equity"])
        if "open_position_length" in infos:
            self.logger.record("custom/open_position_length", infos["open_position_length"])
        return True

class CustomEvalCallback(BaseCallback):
    """
    평가 환경에서 모델을 테스트하고, 
    1. TensorBoard에 자산(Equity) 정보를 기록하며
    2. '평균 최종 자산(Mean Final Equity)'이 가장 높을 때 모델을 저장합니다.
    """
    def __init__(self, eval_env, check_freq: int, log_dir: str, 
                 n_eval_episodes: int = 5, 
                 best_model_save_path: str = None, 
                 verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_equity = -np.inf

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            total_equity = 0.0
            valid_episodes = 0
            
            # --- [1] 평가 루프 (n_eval_episodes 만큼 반복) ---
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                while not done:
                    # Deterministic=True로 평가 (확률적 요소 제거)
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, infos = self.eval_env.step(action)
                    
                    if done:
                        info = infos[0]
                        current_equity = 0.0
                        
                        # Info에서 equity_usd 추출
                        if 'equity_usd' in info:
                            current_equity = info['equity_usd']
                        elif 'terminal_observation' in info and 'equity_usd' in info.get('terminal_info', {}):
                            current_equity = info['terminal_info']['equity_usd']
                            
                        total_equity += current_equity
                        valid_episodes += 1
            
            # --- [2] 결과 계산 및 로깅 ---
            if valid_episodes > 0:
                # 여러 판의 '평균'을 최종 성능으로 간주 (더 안정적임)
                mean_equity = total_equity / valid_episodes
                
                # 🟢 [빠진 부분 추가] TensorBoard 기록
                # 기존 그래프와 이어지도록 태그명을 맞춰줍니다.
                self.logger.record("eval/final_equity_usd", mean_equity)
                self.logger.record("eval/log_final_equity", np.log(max(mean_equity, 1e-6)))
                
                if self.verbose > 0:
                    print(f"Eval at step {self.num_timesteps}: Mean Equity = ${mean_equity:,.2f}")

                # --- [3] Best Model 저장 로직 ---
                if self.best_model_save_path is not None:
                    if mean_equity > self.best_mean_equity:
                        if self.verbose > 0:
                            print(f"🚀 New Best Model! (Equity: ${self.best_mean_equity:,.2f} -> ${mean_equity:,.2f})")
                        
                        self.best_mean_equity = mean_equity
                        
                        # 모델 저장
                        save_path = os.path.join(self.best_model_save_path, "best_model_equity")
                        self.model.save(save_path)
                    
                    # 현재 Best Score 기록
                    self.logger.record("eval/best_equity_usd", self.best_mean_equity)

        return True

# -----------------------------------------------------------------------------
# Environment Factory
# -----------------------------------------------------------------------------
def make_env(mode='train', sl_opts=None, tp_opts=None, window_size=100, max_episode_steps=2048, fee_rate=0.0005, slippage_rate=0.0):
    data_gen = RLDataGenerator(mode=mode, seq_len=window_size)
    env = TradingEnv(
        data_generator=data_gen,
        sl_options=sl_opts,
        tp_options=tp_opts,
        window_size=window_size,
        pip_value=1.0,
        lot_size=0.1,
        spread_pips=10.0,
        fee_rate=fee_rate,          
        slippage_rate=slippage_rate, 
        reward_scale=1.0,
        initial_balance=10000.0,
        max_episode_steps=max_episode_steps,
        mode=mode
    )
    return env

def evaluate_model(model: PPO, eval_env: VecNormalize, deterministic: bool = True):
    eval_env.training = False
    eval_env.norm_reward = False
    obs = eval_env.reset()
    equity_curve = [10000.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        equity_curve.append(infos[0].get("equity_usd", equity_curve[-1]))
        done = dones[0]
    return equity_curve, float(equity_curve[-1])

# -----------------------------------------------------------------------------
# Main Fine-tuning Loop
# -----------------------------------------------------------------------------
def main():
    # 1. 경로 및 파라미터 설정
    # 윈도우 경로인 경우 r"" string 사용 권장
    PRETRAINED_PATH = r"./checkpoints/fee_rate_slippage_zero/best_last_model/best_model.zip"
    
    # [설정] 재학습 파라미터
    FEE_RATE = 0.00025       # 0.025%
    SLIPPAGE_RATE = 0.0001   # 슬리피지도 아주 살짝 (0.01%) 넣어주는 게 현실적임 (선택)
    LEARNING_RATE = 3e-5     # 기존 3e-4 -> 3e-5 (1/10 감소)
    TOTAL_TIMESTEPS = 20_000_000 # 2천만 스텝 (필요에 따라 조절)
    
    # 기타 설정
    SL_OPTS = [0.002, 0.005, 0.01, 0.02] 
    TP_OPTS = [0.005, 0.01, 0.02, 0.04]
    WINDOW_SIZE = 1000
    NUM_ENVS = 8
    EPISODE_LENGTH = 2048
    
    current_time = get_current_datetime()
    # 🟢 새로운 저장 경로 생성
    CHECKPOINT_DIR = f"./checkpoints/finetune_fee_0_025_{current_time}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"🔄 Loading Model from: {PRETRAINED_PATH}")
    print(f"⚙️ Fine-tuning Settings -> Fee: {FEE_RATE*100}%, LR: {LEARNING_RATE}")
    print(f"📂 New Checkpoint Dir: {CHECKPOINT_DIR}")

    # 2. 환경 생성 (새로운 Fee Rate 적용)
    env_fns = [lambda: make_env('train', SL_OPTS, TP_OPTS, WINDOW_SIZE, EPISODE_LENGTH, FEE_RATE, SLIPPAGE_RATE) for _ in range(NUM_ENVS)]
    train_vec_env = DummyVecEnv(env_fns)
    # NormObs=False이므로 통계 파일 로드 없이 새로 만들어도 괜찮음 (Policy 입력 분포는 동일)
    train_env = VecNormalize(train_vec_env, norm_obs=False, norm_reward=True, clip_obs=100.0)

    # 테스트 환경
    test_vec_env = DummyVecEnv([lambda: make_env('test', SL_OPTS, TP_OPTS, WINDOW_SIZE, max_episode_steps=0, fee_rate=FEE_RATE, slippage_rate=SLIPPAGE_RATE)])
    test_env = VecNormalize(test_vec_env, norm_obs=False, norm_reward=False, clip_obs=100.0, training=False)

    # 3. 모델 로드 (Fine-tuning 모드)
    # [중요] custom_objects를 통해 Optimizer의 LR 스케줄러 등을 덮어쓸 수도 있지만,
    # SB3의 load 함수에 learning_rate 인자를 주면 새로운 LR로 설정됩니다.
    try:
        model = PPO.load(
            PRETRAINED_PATH,
            env=train_env,                  # 새로운 환경(Fee 적용됨) 연결
            learning_rate=LEARNING_RATE,    # 낮춘 학습률 적용
            tensorboard_log="./tensorboard_log/", # 로그 폴더 분리
            custom_objects={'learning_rate': LEARNING_RATE} # 안전장치
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 4. 콜백 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_finetune",
        save_vecnormalize=True
    )
    tb_callback = TensorboardCallback()
    eval_callback = CustomEvalCallback(
        eval_env=test_env,
        check_freq=20000,
        log_dir="./tensorboard_log/",
        best_model_save_path=CHECKPOINT_DIR,
        verbose=1
    )
    callback_list = CallbackList([checkpoint_callback, tb_callback, eval_callback])

    # 5. 재학습 시작
    print("🚀 Starting Fine-tuning...")
    # reset_num_timesteps=False를 하면 텐서보드 스텝이 이어서 찍힙니다. 
    # True로 하면 0부터 다시 시작합니다. (새로운 로그 폴더를 쓰므로 True 추천)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, reset_num_timesteps=True)
    
    print("🏁 Fine-tuning Finished.")
    model.save(os.path.join(CHECKPOINT_DIR, "final_finetuned_model"))
    train_env.save(os.path.join(CHECKPOINT_DIR, "final_vecnormalize.pkl"))

if __name__ == "__main__":
    main()