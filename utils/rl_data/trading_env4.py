# utils/rl_data/trading_env3.py

from __future__ import annotations

import logging
import sys
import numpy as np

# Prefer gymnasium if available
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False

from .data_generator import RLDataGenerator

# EUC-KR 로깅 설정 (Windows용)
if sys.platform == 'win32':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='euc-kr'
    )

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    BTCUSDT Futures Trading Environment (Final Version)

    Features:
      - Data: Uses RLDataGenerator (.npy).
      - Action: HOLD, CLOSE, OPEN (Direction + % SL/TP).
      - Fee: Percentage-based (0.05% Taker by default).
      - Logging: Log-scale Equity included.
      - Episode: Fixed length truncation (max_episode_steps).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_generator: RLDataGenerator,
        sl_options: list[float] = [0.002, 0.005, 0.01], 
        tp_options: list[float] = [0.005, 0.01, 0.02],
        window_size: int = 100,
        pip_value: float = 1.0,         # BTCUSDT 1 pip = 1 USDT
        lot_size: float = 0.1,          # 0.1 BTC per trade
        spread_pips: float = 10.0,      # Spread Cost ($)
        fee_rate: float = 0.0005,       # 🟢 [변경] 거래 대금의 0.05% (Binance Taker)
        slippage_rate: float = 0.0005, # Random slippage up to $5
        reward_scale: float = 1.0,
        unrealized_delta_weight: float = 0.02,
        hold_reward_weight: float = 0.005,
        open_penalty_pips: float = 1.0,
        time_penalty_pips: float = 0.02,
        initial_balance: float = 10000.0,
        max_episode_steps: int = 2048,  # 🟢 [추가] 에피소드 길이 제한
        allow_flip: bool = False,
        scaling_factor: float = 100.0,       
        mode: str = 'train'
    ):
        super().__init__()

        self.data_generator = data_generator
        self.mode = mode
        
        # Validate Data Generator Seq Len
        self.max_episode_steps = max_episode_steps
        if data_generator.seq_len != window_size:
            logger.warning(f"Generator seq_len ({data_generator.seq_len}) != window_size ({window_size}). Using generator's seq_len.")
            self.window_size = data_generator.seq_len
        else:
            self.window_size = window_size

        self.pip_value = float(pip_value)
        self.lot_size = float(lot_size)
        self.scaling_factor = float(scaling_factor)
        
        # Costs
        self.spread_pips = float(spread_pips)
        self.fee_rate = float(fee_rate)       # 비율 수수료
        
        # PnL to USD conversion
        self.usd_per_pip = self.pip_value * self.lot_size

        # Reward Hyperparams
        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)

        self.initial_balance = initial_balance
        self.slippage_rate = slippage_rate
        self.allow_flip = bool(allow_flip)

        # SL/TP Options
        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided.")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)

        # --- Actions Mapping ---
        # 0: HOLD, 1: CLOSE, 2..: OPEN
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:  # 0=Short, 1=Long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))
        
        self.action_space = spaces.Discrete(len(self.action_map))
        
        # --- Observation Space ---
        self.base_num_features = self.data_generator.get_feature_dim()
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self):
        """Internal state reset"""
        self.current_index = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        
        # Position State
        self.position = 0      # 0=Flat, 1=Long, -1=Short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.open_position_length = 0
        
        # Balance
        self.equity_usd = self.initial_balance
        self.equity_curve = []
        self.last_trade_info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        
        min_idx = self.data_generator.reset()
        total_samples = self.data_generator.get_num_samples() + min_idx
        
        # Determine start index
        max_start = total_samples - self.max_episode_steps - 100
        if max_start <= min_idx:
            max_start = total_samples - 100

        if self.mode == 'train':
            self.current_index = int(self.np_random.integers(min_idx, max_start))
        else:
            self.current_index = min_idx

        obs = self._get_observation()
        
        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        # 1. Check termination
        if self.terminated or self.truncated:
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1

        # 2. Get Data
        try:
            _, current_price, next_price = self.data_generator.get_sequence(self.current_index)
        except IndexError:
            self.terminated = True
            obs = self._get_observation()
            return obs, 0.0, True, False, {}

        # 3. Process Action
        reward_pips = 0.0
        act_type, direction, sl_ratio, tp_ratio = self.action_map[int(action)]
        
        if act_type == "HOLD":
            pass
            
        elif act_type == "CLOSE":
            if self.position != 0:
                # 🟢 [수정] 현재 Long이면 매도(False), Short면 매수(True)로 청산
                is_buy = (self.position == -1) 
                exit_price = self._apply_slippage(current_price, is_buy=is_buy)
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)
                
        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(current_price, direction, sl_ratio, tp_ratio)
                self.open_position_length += 1
                reward_pips -= self.open_penalty_pips
            else:
                if self.allow_flip:
                    # 🟢 [수정] 스위칭 시 기존 포지션 청산 (Long->매도, Short->매수)
                    is_buy_close = (self.position == -1)
                    exit_price = self._apply_slippage(current_price, is_buy=is_buy_close)
                    self.open_position_length += 1
                    reward_pips += self._close_position("FLIP_CLOSE", exit_price)
                    self._open_position(current_price, direction, sl_ratio, tp_ratio)
                    reward_pips -= self.open_penalty_pips

        # 4. Intrabar Check (SL/TP)
        realized_now = self._check_sl_tp_next_price(next_price)
        if realized_now is not None:
            reward_pips += realized_now

        # 5. Reward Shaping
        if self.position != 0:
            self.time_in_trade += 1
            unreal_now = self._compute_unrealized_pips(next_price)
            delta_unreal = unreal_now - self.prev_unrealized_pips
            
            if unreal_now > 0:
                reward_pips += self.hold_reward_weight * unreal_now
            
            if self.unrealized_delta_weight != 0.0:
                reward_pips += self.unrealized_delta_weight * delta_unreal
                
            reward_pips -= self.time_penalty_pips
            self.prev_unrealized_pips = unreal_now

        # 6. Advance Time
        self.current_index += 1
        
        # 🟢 [추가] 에피소드 길이 제한
        if self.max_episode_steps > 0 and self.steps_in_episode >= self.max_episode_steps:
            self.truncated = True

        if self.current_index >= self.data_generator.get_num_samples() - 2:
            self.terminated = True
            
        # Check Bankruptcy (-50% equity)
        if self.equity_usd <= self.initial_balance * 0.5:
            self.terminated = True
            reward_pips -= 1000.0

        # 7. Finalize
        obs = self._get_observation()
        reward = float(reward_pips) * self.reward_scale
        
        # 🟢 [추가] 로그 스케일 Equity 계산
        safe_equity = max(self.equity_usd, 1e-6)
        log_equity = np.log(safe_equity)

        info = {
            "equity_usd": float(self.equity_usd),
            "log_equity": float(log_equity),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "last_trade_info": self.last_trade_info,
            "open_position_length": self.open_position_length
        }

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            return obs, reward, bool(self.terminated or self.truncated), info

    def _get_observation(self):
        try:
            state_seq, _, _ = self.data_generator.get_sequence(self.current_index)
        except IndexError:
            state_seq = np.zeros((self.window_size, self.base_num_features))

        state_seq = state_seq * self.scaling_factor

        pos_feat = float(self.position)
        time_feat = float(self.time_in_trade) / 100.0
        
        _, curr_p, _ = self.data_generator.get_sequence(self.current_index)
        unreal_pips = self._compute_unrealized_pips(curr_p)
        pnl_feat = unreal_pips / 100.0
        
        state_vec = np.array([pos_feat, time_feat, pnl_feat], dtype=np.float32)
        state_block = np.tile(state_vec, (self.window_size, 1))
        return np.hstack([state_seq, state_block]).astype(np.float32)

    # 🟢 [수정] 매개변수 이름을 ratio로 통일 (sl_pips -> sl_ratio)
    def _open_position(self, current_price, direction, sl_ratio, tp_ratio):
        # direction: 1 (Long/Buy), 0 (Short/Sell)
        is_buy = (direction == 1)
        
        # 🟢 [수정] 매수면 True, 매도면 False 전달
        entry_price = self._apply_slippage(current_price, is_buy=is_buy)
        
        self.position = 1 if direction == 1 else -1
        self.entry_price = entry_price
        
        price_delta_sl = entry_price * sl_ratio
        price_delta_tp = entry_price * tp_ratio
        
        if self.position == 1: 
            self.sl_price = entry_price - price_delta_sl
            self.tp_price = entry_price + price_delta_tp
        else: 
            self.sl_price = entry_price + price_delta_sl
            self.tp_price = entry_price - price_delta_tp
            
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        
        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_index,
            "type": "LONG" if self.position == 1 else "SHORT",
            "entry": self.entry_price,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "sl_ratio": sl_ratio,
            "tp_ratio": tp_ratio
        }

    def _close_position(self, reason, exit_price):
        if self.position == 0: return 0.0
        
        # 1. Gross PnL (USDT)
        if self.position == 1:
            gross_pnl_usd = (exit_price - self.entry_price) * self.lot_size
        else:
            gross_pnl_usd = (self.entry_price - exit_price) * self.lot_size
            
        # 🟢 [수정] 수수료 계산: (진입가 + 청산가) * 사이즈 * 비율
        entry_fee = self.entry_price * self.lot_size * self.fee_rate
        exit_fee = exit_price * self.lot_size * self.fee_rate
        total_fee_usd = entry_fee + exit_fee
        
        # Spread Cost
        spread_cost_usd = self.spread_pips * self.usd_per_pip
        
        total_cost_usd = total_fee_usd + spread_cost_usd
        net_pnl_usd = gross_pnl_usd - total_cost_usd
        
        # 3. Update Equity
        self.equity_usd += net_pnl_usd
        
        # 4. Return Reward in Pips
        net_pips = net_pnl_usd / self.usd_per_pip
        
        self.last_trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_index,
            "exit": exit_price,
            "gross_pnl_usd": float(gross_pnl_usd),
            "fee_usd": float(total_fee_usd),
            "net_pnl_usd": float(net_pnl_usd),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd)
        }
        
        # Reset
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        
        return net_pips

    def _check_sl_tp_next_price(self, next_price):
        if self.position == 0: return None
        
        sl_hit = False
        tp_hit = False
        
        if self.position == 1: # Long
            if next_price <= self.sl_price: sl_hit = True
            if next_price >= self.tp_price: tp_hit = True
        else: # Short
            if next_price >= self.sl_price: sl_hit = True
            if next_price <= self.tp_price: tp_hit = True
            
        if sl_hit:
            return self._close_position("SL_HIT", self.sl_price)
        elif tp_hit:
            return self._close_position("TP_HIT", self.tp_price)
            
        return None

    def _compute_unrealized_pips(self, current_price):
        if self.position == 0: return 0.0
        
        if self.position == 1:
            diff = current_price - self.entry_price
        else:
            diff = self.entry_price - current_price
            
        return diff / self.pip_value

    def _apply_slippage(self, price, is_buy: bool):
        """
        슬리피지 = 현재가 * 랜덤 비율 (0 ~ slippage_rate)
        항상 불리하게 적용 (Penalty)
        """
        if self.slippage_rate <= 0: return price
        
        # 0 ~ slippage_rate 사이의 랜덤 비율 추출
        rate = np.random.uniform(0, self.slippage_rate)
        slip_value = price * rate  # 가격에 비례한 슬리피지 금액($)
        
        if is_buy:
            return price + slip_value # 비싸게 삼 (불리함)
        else:
            return price - slip_value # 싸게 팜 (불리함)