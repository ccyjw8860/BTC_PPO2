"""
Trading Environment for Reinforcement Learning

BTCUSDT 선물 거래 환경 (Gym Environment)
"""

from typing import Dict, List, Tuple
import logging
import sys
import numpy as np
import gymnasium as gym

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
    BTCUSDT 선물 거래 환경 (Gym Environment)

    Features:
    - 20배 고정 레버리지
    - 초기 잔고당 25% 고정 마진 ($2,500 per entry)
    - 최대 4레이어 피라미딩 (방향별)
    - 융합 아키텍처: 시장 상태(100,40) + 에이전트 상태(4,)
    """

    def __init__(
        self,
        data_generator: RLDataGenerator,
        initial_balance: float = 10000.0,
        leverage: int = 5,
        margin_per_entry_ratio: float = 0.25,
        max_layers: int = 4,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0002,
        time_penalty_no_position: float = 0.0,
        time_penalty_with_position: float = -0.001,
        bankruptcy_penalty: float = -10.0,
        liquidation_penalty: float = -10.0,
        plus_reward_weight: float = 5,
        episode_length: int = 2000,
        mode: str = 'train',

        # Phase control flags (보상 함수 Phase 제어)
        enable_phase1_safety: bool = True,
        enable_phase2_risk_adj: bool = True,
        enable_phase3_optimization: bool = True,

        # Phase 1 parameters (안전성 및 생존)
        mdd_threshold: float = 0.05,
        mdd_penalty_coeff: float = 0.5,
        global_mdd_threshold: float = 0.10,  # [NEW] 전체 에피소드 기준 MDD 임계값 (10%)
        global_mdd_penalty_coeff: float = 1.0, # [NEW] Global MDD 페널티 계수
        liquidation_danger_threshold: float = 0.5,
        liquidation_penalty_base: float = 2.0,
        margin_usage_threshold: float = 0.75,
        margin_penalty_coeff: float = 1.0,
        pre_bankruptcy_threshold: float = 0.7,

        # Phase 2 parameters (위험 조정 수익률)
        equity_window_size: int = 50,
        sortino_penalty_coeff: float = 0.5, # [RENAME] sharpe_scaling_factor -> sortino_penalty_coeff
        dynamic_penalty_base: float = 0.02,
        dynamic_penalty_trend_coeff: float = 0.05,
        dynamic_penalty_vol_coeff: float = 0.02,
        overtrading_threshold: float = 0.02,
        asymmetric_hold_loss_coeff: float = 0.05,
        asymmetric_hold_profit_coeff: float = 0.03,

        # Phase 3 parameters (기회비용 및 복리 성장)
        opportunity_cost_threshold: float = 0.002,
        opportunity_cost_coeff: float = 0.2,
        compound_bonus_coeff: float = 0.01,
        mfe_capture_threshold: float = 0.5,
        mfe_capture_bonus_coeff: float = 0.2,
        mfe_capture_penalty_coeff: float = 0.3,
        volatility_sizing_tolerance: float = 0.2,
        volatility_sizing_bonus_coeff: float = 0.05,
        volatility_sizing_penalty_coeff: float = 0.1,
    ):
        """
        Initialize Trading Environment

        Args:
            data_generator: RLDataGenerator 인스턴스
            initial_balance: 초기 자본 ($)
            leverage: 레버리지 배율 (고정)
            margin_per_entry_ratio: 레이어당 증거금 비율 (초기 자본 대비)
            max_layers: 최대 레이어 수 (방향별)
            fee_rate: 수수료율 (0.05%)
            slippage_rate: 슬리피지율 (0.02%)
            liquidation_threshold: 강제 청산 임계값 (2.5%)
            time_penalty_no_position: 포지션 없을 때 시간 페널티
            time_penalty_with_position: 포지션 있을 때 시간 페널티
            episode_length: 에피소드 길이 (스텝 수)
            mode: 'train' 또는 'test' (random start vs sequential)
        """
        super().__init__()

        # Data generator
        self.data_generator = data_generator
        self.total_samples = data_generator.get_num_samples()

        # 자금 관리 파라미터
        self.initial_balance = initial_balance
        self.entry_equity = initial_balance
        self.leverage = leverage
        self.margin_per_entry_ratio = margin_per_entry_ratio
        self.max_layers = max_layers
        self.position_max_equity = 0.0  # 현재 포지션 내에서의 최고 Equity

        # 레이어당 포지션 크기 계산
        # 예: $10,000 * 0.25 * 20 = $50,000
        self.notional_per_layer = initial_balance * margin_per_entry_ratio * leverage

        # 관측 상태 추적 변수 (Phase 2에서 추가)
        self.position_entry_step = 0  # 진입 시점 스텝
        self.prev_scaled_pnl = 0.0  # 이전 스텝의 PnL (profit velocity 계산용)

        # 거래 비용 파라미터
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

        # 청산 및 종료 파라미터
        # 청산 임계값: 레버리지에 따라 동적으로 계산
        # - 공식: 0.5 / leverage (자산의 50% 손실 시 청산)
        # - 레버리지 5배: 10% 역행 시 청산 (0.5 / 5 = 0.1)
        # - 레버리지 20배: 2.5% 역행 시 청산 (0.5 / 20 = 0.025)
        # 예) Long 포지션에서 가격이 진입가 대비 10% 하락하면 5배 레버리지 기준 50% 손실
        self.liquidation_threshold = 0.5 / leverage
        self.time_penalty_no_position = time_penalty_no_position
        self.time_penalty_with_position = time_penalty_with_position
        self.bankruptcy_penalty = bankruptcy_penalty
        self.liquidation_penalty = liquidation_penalty
        self.plus_reward_weight = plus_reward_weight
        self.episode_length = episode_length

        # 모드 설정
        self.mode = mode

        # Phase control flags 저장
        self.enable_phase1_safety = enable_phase1_safety
        self.enable_phase2_risk_adj = enable_phase2_risk_adj
        self.enable_phase3_optimization = enable_phase3_optimization

        # Phase 1 파라미터 저장
        self.mdd_threshold = mdd_threshold
        self.mdd_penalty_coeff = mdd_penalty_coeff
        self.global_mdd_threshold = global_mdd_threshold
        self.global_mdd_penalty_coeff = global_mdd_penalty_coeff
        self.liquidation_danger_threshold = liquidation_danger_threshold
        self.liquidation_penalty_base = liquidation_penalty_base
        self.margin_usage_threshold = margin_usage_threshold
        self.margin_penalty_coeff = margin_penalty_coeff
        self.pre_bankruptcy_threshold = pre_bankruptcy_threshold

        # Phase 2 파라미터 저장
        self.equity_window_size = equity_window_size
        self.sortino_penalty_coeff = sortino_penalty_coeff
        self.dynamic_penalty_base = dynamic_penalty_base
        self.dynamic_penalty_trend_coeff = dynamic_penalty_trend_coeff
        self.dynamic_penalty_vol_coeff = dynamic_penalty_vol_coeff
        self.overtrading_threshold = overtrading_threshold
        self.asymmetric_hold_loss_coeff = asymmetric_hold_loss_coeff
        self.asymmetric_hold_profit_coeff = asymmetric_hold_profit_coeff

        # Phase 3 파라미터 저장
        self.opportunity_cost_threshold = opportunity_cost_threshold
        self.opportunity_cost_coeff = opportunity_cost_coeff
        self.compound_bonus_coeff = compound_bonus_coeff
        self.mfe_capture_threshold = mfe_capture_threshold
        self.mfe_capture_bonus_coeff = mfe_capture_bonus_coeff
        self.mfe_capture_penalty_coeff = mfe_capture_penalty_coeff
        self.volatility_sizing_tolerance = volatility_sizing_tolerance
        self.volatility_sizing_bonus_coeff = volatility_sizing_bonus_coeff
        self.volatility_sizing_penalty_coeff = volatility_sizing_penalty_coeff

        # Phase 2 추적 변수
        self.equity_history = []
        self.episode_fees_paid = 0.0

        # Phase 3 추적 변수
        self.prev_price = 0.0

        # Action space: 0=Hold, 1=Flat, 2=Long, 3=Short
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: Dict with market and agent
        # market: (seq_len, features) = (100, n_features)
        # agent: 11차원 - [pos_type, layers, pos_return, pnl, equity, liq_distance, volatility, trend, hold_duration, profit_velocity, margin_usage]
        self.observation_space = gym.spaces.Dict({
            'market': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(data_generator.seq_len, data_generator.get_feature_dim()),
                dtype=np.float32
            ),
            'agent': gym.spaces.Box(
                    low=np.array([
                        0.0,   # pos_type
                        0.0,   # layers
                        -1.0,  # pos_return
                        -1.0,  # pnl
                        0.0,   # equity
                        0.0,   # liq_distance
                        0.0,   # volatility
                        -1.0,  # trend_strength
                        0.0,   # hold_duration
                        -1.0,  # profit_velocity
                        0.0,   # margin_usage
                        -1.0   # mfe_ratio
                    ], dtype=np.float32),
                    high=np.array([
                        1.0,   # pos_type
                        1.0,   # layers
                        1.0,   # pos_return
                        1.0,   # pnl
                        2.0,   # equity
                        1.0,   # liq_distance
                        1.0,   # volatility (tanh 적용하여 0~1 범위)
                        1.0,   # trend_strength
                        1.0,   # hold_duration
                        1.0,   # profit_velocity
                        1.0,   # margin_usage
                        1.0,   # mfe_ratio
                    ], dtype=np.float32),
                    shape=(12,),  # 5 → 11로 확장
                    dtype=np.float32
                )
        })

        logger.info(f"TradingEnv 초기화: balance=${initial_balance}, leverage={leverage}x, mode={mode}")

    def reset(self, seed=None, options=None):
        """
        환경 리셋 (에피소드 시작)

        Training mode: Random start index within valid range
        Test mode: Sequential from beginning

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            observation: Dict with 'market' and 'agent' keys
            info: Additional information dict
        """
        super().reset(seed=seed)

        # 포지션 상태 초기화
        self.position_type = 0  # 0=Flat, 1=Long, 2=Short
        self.num_layers = 0
        self.layer_entries = []  # [{price, notional, fee_paid}, ...]
        self.avg_entry_price = 0.0
        self.total_notional = 0.0
        self.is_open_position_len = 0 
        self.position_max_equity = 0.0  # 현재 포지션 내에서의 최고 Equity

        # 잔고 초기화
        self.balance = self.initial_balance
        self.equity = self.initial_balance

        # 에피소드 상태 초기화
        self.step_count = 0
        self.max_equity = self.initial_balance  # 최대 자산 추적
        self.min_equity = self.initial_balance
        self.global_max_equity = self.initial_balance # [NEW] 에피소드 전체 Global Max Equity

        # 신규 상태 변수 초기화 (Phase 2)
        self.position_entry_step = 0
        self.prev_scaled_pnl = 0.0

        # Phase 2 추적 변수 초기화
        self.equity_history = []
        self.episode_fees_paid = 0.0

        # Phase 3 추적 변수 초기화
        self.prev_price = 0.0

        # 시작 가능한 최소/최대 인덱스 계산
        # min_idx: 첫 시퀀스를 만들기 위해 필요한 최소 인덱스 (seq_len - 1)
        min_idx = self.data_generator.reset() 
        
        # total_samples: 데이터 제너레이터가 가진 전체 유효 인덱스 수
        total_samples = self.total_samples + min_idx
        
        # 🟢 수정된 로직: 
        # 데이터의 끝에서 에피소드 길이만큼을 뺀 지점이 '마지막으로 시작 가능한' 위치입니다.
        # 그래야 에피소드가 진행되는 동안 Index Out of Range가 발생하지 않습니다.
        max_start_idx = total_samples - self.episode_length - 1

        # 학습과 테스트 모두에서 랜덤 시작을 원하신다면:
        if max_start_idx > min_idx:
            self.current_index = self.np_random.integers(min_idx, max_start_idx + 1)
        else:
            self.current_index = min_idx

        if self.mode == 'test':
            logger.info(f"🧪 테스트 모드: 랜덤 시작점({self.current_index})부터 평가를 시작합니다.")

            
        
        # 초기 관측 생성
        obs = self._get_observation()

        info = {
            'start_index': self.current_index,
            'initial_balance': self.initial_balance
        }

        return obs, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        관측 생성 - Robust Scaled 데이터 특성 반영 (11차원 agent state)
        """
        state, current_price, _ = self.data_generator.get_sequence(self.current_index)
        mfe_ratio = 0.0
        # 1. Position Type & 2. Num Layers (기존 동일)
        scaled_pos_type = self.position_type / 2.0
        scaled_layers = self.num_layers / self.max_layers

        # 3. Position Return: 진입가 대비 현재 수익률 (레버리지 반영)
        pos_return = 0.0
        if self.position_type != 0 and self.avg_entry_price > 0:
            # 방향성 반영 (Long=1, Short=-1)
            direction = 1 if self.position_type == 1 else -1
            # 순수 가격 변동률에 레버리지 적용
            raw_return = (current_price - self.avg_entry_price) / self.avg_entry_price * self.leverage * direction
            # -1.0(청산가 근접) ~ 1.0(수익권) 범위로 클리핑
            pos_return = np.clip(raw_return, -1.0, 1.0)

        

        # 4. Unrealized PNL (기존 계좌 대비 수익률)
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        unrealized_pnl_ratio = (unrealized_pnl / self.initial_balance) * 10.0
        scaled_pnl = np.clip(unrealized_pnl_ratio, -1.0, 1.0)

        # 5. Equity Ratio
        self.equity = self.balance + unrealized_pnl
        scaled_equity = np.clip(self.equity / self.initial_balance, 0.0, 2.0)

        # 6. 청산 거리 (Liquidation Distance)
        if self.position_type != 0:
            current_move = (current_price - self.avg_entry_price) / self.avg_entry_price
            direction = 1 if self.position_type == 1 else -1
            distance_to_liq = (self.liquidation_threshold - abs(current_move * direction))
            scaled_liq_distance = np.clip(distance_to_liq / self.liquidation_threshold, 0.0, 1.0)
            mfe_ratio = (self.position_max_equity - self.entry_equity) / self.entry_equity

            # 환경 변수로 저장 (보상 함수에서 사용)
            self.liquidation_distance_ratio = scaled_liq_distance
        else:
            scaled_liq_distance = 1.0
            self.liquidation_distance_ratio = 1.0

        # 7. 상대적 변동성 (Robust Scaled 기반)
        # state[:, 0]은 이미 Robust Scaled된 로그 수익률
        # std가 1.0 근처 = 평소 수준, 2.0 이상 = 폭발적 변동성
        scaled_volatility = np.tanh(np.std(state[-20:, 0]))

        # 8. 추세 강도 (EMA slope 5, 20, 60 활용)
        # price_calculator.py에서 EMA slope는 인덱스 4, 5, 6, 7, 8
        # feat_ema5_slope = 4, feat_ema20_slope = 5, feat_ema40_slope = 6, feat_ema60_slope = 7, feat_ema120_slope = 8
        ema5_s = state[-1, 4]
        ema20_s = state[-1, 5]
        ema60_s = state[-1, 7]
        trend_score = (ema5_s * 2.0 + ema20_s * 1.5 + ema60_s * 1.0) / 4.5
        trend_strength = np.tanh(trend_score * 2.0)

        # 9. 보유 기간 (Hold Duration)
        scaled_hold_duration = 0.0
        if self.position_type != 0:
            hold_duration = (self.step_count - self.position_entry_step) / 100.0
            scaled_hold_duration = np.tanh(hold_duration)

        # 10. 수익 변화 속도 (Profit Velocity)
        profit_velocity = 0.0
        if self.position_type != 0:
            pnl_change = scaled_pnl - self.prev_scaled_pnl
            profit_velocity = np.tanh(pnl_change * 5.0)

        # prev_scaled_pnl 업데이트 (다음 스텝용)
        self.prev_scaled_pnl = scaled_pnl

        # 11. 마진 사용률 (Margin Usage Ratio)
        margin_usage_ratio = self.num_layers / self.max_layers

        # 최종 agent state 구성 (11차원)
        agent_state = np.array([
            scaled_pos_type,      # 0
            scaled_layers,        # 1
            pos_return,           # 2
            scaled_pnl,           # 3
            scaled_equity,        # 4
            scaled_liq_distance,  # 5
            scaled_volatility,    # 6
            trend_strength,       # 7
            scaled_hold_duration, # 8
            profit_velocity,      # 9
            margin_usage_ratio,    # 10
            mfe_ratio
        ], dtype=np.float32)

        return {
            'market': state.astype(np.float32),
            'agent': agent_state
        }
        
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        미실현 손익 계산 (마진에 영향 없음)

        Long: (current_price - avg_entry) / avg_entry × total_notional
        Short: (avg_entry - current_price) / avg_entry × total_notional

        Args:
            current_price: 현재 가격

        Returns:
            미실현 손익 ($)
        """
        if self.position_type == 0:  # Flat
            return 0.0

        if self.position_type == 1:  # Long
            # 가격 상승 시 이익
            return (current_price - self.avg_entry_price) / self.avg_entry_price * self.total_notional
        else:  # Short (position_type == 2)
            # 가격 하락 시 이익
            return (self.avg_entry_price - current_price) / self.avg_entry_price * self.total_notional

    def step(self, action: int):
        """
        환경 진행 (1 step = 1 candle forward)

        Args:
            action: 0=Hold, 1=Flat, 2=Long, 3=Short

        Returns:
            observation: Dict with 'market' and 'agent'
            reward: float
            terminated: bool (episode ended)
            truncated: bool (not used, always False)
            info: dict with additional information
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. 사전 상태 저장 (reward 계산용)
        prev_equity = self.equity
        prev_pos_type = self.position_type
        prev_entry_equity = self.entry_equity
        prev_hold_len = self.is_open_position_len

        # 2. 현재 가격 취득
        _, current_price, next_price = self.data_generator.get_sequence(self.current_index)

        # 3. 액션 실행
        fee_cost = 0
        if action == 1:
            fee_cost = self._execute_flat(current_price)
        elif action == 2:
            fee_cost = self._execute_long(current_price)
        elif action == 3:
            fee_cost = self._execute_short(current_price)

        # 4. 인덱스 증가 및 상태 업데이트
        self.current_index += 1
        self.step_count += 1
        if self.position_type != 0:
            self.is_open_position_len += 1

        # 5. next_price로 equity 갱신
        unrealized_pnl = self._calculate_unrealized_pnl(next_price)
        self.equity = self.balance + unrealized_pnl
        self.max_equity = max(self.max_equity, self.equity)
        self.global_max_equity = max(self.global_max_equity, self.equity) # [NEW] Global Max 갱신
        self.min_equity = min(self.min_equity, self.equity)

        # Phase 2: equity_history 업데이트
        self.equity_history.append(self.equity)
        if len(self.equity_history) > self.equity_window_size:
            self.equity_history.pop(0)

        # Phase 2: 수수료 누적
        self.episode_fees_paid += fee_cost

        # 6. 종료 조건 체크
        terminated, liquidated, bankruptcy = self._check_episode_end(next_price)
        if liquidated or bankruptcy:
            self._execute_flat(next_price)
            self.equity = self.balance
            terminated = True

        # 7. 보상 계산 (모든 컨텍스트 전달)
        reward = self._calculate_reward(
            prev_equity=prev_equity,
            current_equity=self.equity,
            action=action,
            liquidated=liquidated,
            bankruptcy=bankruptcy,
            old_pos_type=prev_pos_type,
            old_entry_equity=prev_entry_equity,
            old_hold_len=prev_hold_len,
            fee_cost=fee_cost
        )

        # Phase 3: prev_price 업데이트 (보상 계산 후)
        self.prev_price = next_price

        # 관측 생성 (IndexError 방지)
        # terminated된 상태에서는 current_index가 데이터 범위를 벗어날 수 있으므로
        # 인덱스를 일시적으로 되돌려서 마지막 유효한 관측치를 가져옴
        if terminated:
            self.current_index -= 1
            obs = self._get_observation()
            self.current_index += 1  # 인덱스 상태 복구
        else:
            obs = self._get_observation()

        # Info dict
        info = {
            'balance': self.balance,
            'equity': self.equity,
            'position_type': self.position_type,
            'num_layers': self.num_layers,
            'unrealized_pnl': unrealized_pnl,
            'fee_paid': fee_cost,
            'liquidated': liquidated,
            'avg_entry_price': self.avg_entry_price,
            'bankruptcy': bankruptcy
        }

        # 에피소드 종료 시 추가 메트릭
        if terminated:
            return_pct = (self.equity - self.initial_balance) / self.initial_balance * 100
            max_equity_ratio = self.max_equity / self.initial_balance
            min_equity_ratio = self.min_equity / self.initial_balance
            open_position_len = self.is_open_position_len
            info['return_pct'] = return_pct
            info['max_equity_ratio'] = max_equity_ratio
            info['min_equity_ratio'] = min_equity_ratio
            info['is_liquidated'] = liquidated
            info['open_position_len'] = open_position_len
            info['is_bankruptcy'] = bankruptcy

        return obs, reward, terminated, False, info  # truncated=False

    def _execute_hold(self, current_price: float) -> float:
        """
        HOLD 액션 실행 (상태 유지)

        Args:
            current_price: 현재 가격 (사용하지 않음)

        Returns:
            fee_cost: 0.0 (수수료 없음)
        """
        return 0.0

    def _execute_flat(self, current_price: float) -> float:
        """
        FLAT 액션 실행 (모든 레이어 청산)

        Args:
            current_price: 청산 가격 기준

        Returns:
            fee_cost: 청산 수수료
        """
        if self.position_type == 0:  # Already flat
            return 0.0

        # 슬리피지 적용된 청산 가격 계산
        if self.position_type == 1:  # Long exit (sell)
            exit_price = current_price * (1 - self.slippage_rate)
        elif self.position_type == 2:  # Short exit (buy)
            exit_price = current_price * (1 + self.slippage_rate)

        # 실현 손익 계산
        pnl = self._calculate_pnl(exit_price)

        # Balance 업데이트 (손익 실현)
        self.balance += pnl

        # 청산 수수료 계산 및 차감
        exit_fee = self.total_notional * self.fee_rate
        self.balance -= exit_fee

        # 포지션 상태 초기화
        self.position_type = 0
        self.num_layers = 0
        self.layer_entries = []
        self.avg_entry_price = 0.0
        self.total_notional = 0.0
        self.equity = self.balance
        self.position_max_equity = 0 # 진입 시점의 Equity로 초기화

        return exit_fee

    def _execute_long(self, current_price: float) -> float:
        """
        LONG 액션 실행 (Long 진입 또는 레이어 추가)

        Cases:
        1. Flat → Open 1 Long layer
        2. Long + layers < 4 → Add 1 layer
        3. Long + layers == 4 → Do nothing (max reached)
        4. Short → Close all + Open 1 Long layer

        Args:
            current_price: 진입 가격 기준

        Returns:
            fee_cost: 총 수수료 (전환 시 청산 + 진입)
        """
        total_fees = 0.0

        # Case 4: Reverse from Short
        if self.position_type == 2:
            total_fees += self._execute_flat(current_price)

        # Case 3: Max layers reached
        if self.position_type == 1 and self.num_layers >= self.max_layers:
            return total_fees

        # Case 1 & 2: Add Long layer
        # 슬리피지 적용된 진입 가격 (매수 시 더 높게)
        entry_price = current_price * (1 + self.slippage_rate)

        # 포지션 크기 (레이어당 고정)
        notional = self.notional_per_layer

        # 진입 수수료 계산
        entry_fee = notional * self.fee_rate

        # VWAP 갱신
        self._update_avg_entry_price(entry_price, notional)

        # 레이어 추가
        self.layer_entries.append({
            'price': entry_price,
            'notional': notional,
            'fee_paid': entry_fee
        })

        # 상태 업데이트
        # [수정] 처음 진입하는 경우(레이어가 0일 때) 당시의 Equity 기록
        self.balance -= entry_fee
        total_fees += entry_fee
        self.equity = self.balance + self._calculate_unrealized_pnl(current_price)
        if self.num_layers == 0:
            self.entry_equity = self.equity
            self.position_entry_step = self.step_count  # 진입 시점 기록

        self.position_max_equity = self.equity  # 진입 시점의 Equity로 초기화

        self.position_type = 1  # Long
        self.num_layers += 1
        self.total_notional += notional


        return total_fees

    def _execute_short(self, current_price: float) -> float:
        """
        SHORT 액션 실행 (Short 진입 또는 레이어 추가)

        Args:
            current_price: 진입 가격 기준

        Returns:
            fee_cost: 총 수수료
        """
        total_fees = 0.0

        # Reverse from Long
        if self.position_type == 1:
            total_fees += self._execute_flat(current_price)

        # Max layers reached
        if self.position_type == 2 and self.num_layers >= self.max_layers:
            return total_fees

        # Add Short layer
        # 슬리피지 적용 (매도 시 더 낮게)
        entry_price = current_price * (1 - self.slippage_rate)
        notional = self.notional_per_layer
        entry_fee = notional * self.fee_rate

        # VWAP 갱신
        self._update_avg_entry_price(entry_price, notional)

        # 레이어 추가
        self.layer_entries.append({
            'price': entry_price,
            'notional': notional,
            'fee_paid': entry_fee
        })

        # 상태 업데이트
        # [수정] 처음 진입하는 경우(레이어가 0일 때) 당시의 Equity 기록
        self.balance -= entry_fee
        total_fees += entry_fee
        self.equity = self.balance + self._calculate_unrealized_pnl(current_price)
        if self.num_layers == 0:
            self.entry_equity = self.equity
            self.position_entry_step = self.step_count  # 진입 시점 기록

        self.position_max_equity = self.equity  # 진입 시점의 Equity로 초기화

        self.position_type = 2  # Short
        self.num_layers += 1
        self.total_notional += notional


        return total_fees

    def _update_avg_entry_price(self, new_price: float, new_notional: float):
        """
        가중평균 진입가 갱신 (VWAP)

        공식: (기존평균가 × 기존포지션크기 + 신규진입가 × 신규포지션크기) / 총포지션크기

        Args:
            new_price: 신규 진입 가격
            new_notional: 신규 포지션 크기
        """
        if self.num_layers == 0:
            # 첫 진입
            self.avg_entry_price = new_price
        else:
            # VWAP 계산
            total_old = self.avg_entry_price * self.total_notional
            total_new = new_price * new_notional
            self.avg_entry_price = (total_old + total_new) / (self.total_notional + new_notional)

    def _calculate_pnl(self, exit_price: float) -> float:
        """
        실현 손익 계산 (청산 시)

        Note: 슬리피지는 exit_price에 이미 적용되어 있음

        Args:
            exit_price: 청산 가격 (슬리피지 적용 완료)

        Returns:
            realized_pnl: 실현 손익 ($)
        """
        if self.position_type == 1:  # Long
            # 가격 상승 시 이익
            return (exit_price - self.avg_entry_price) / self.avg_entry_price * self.total_notional
        elif self.position_type == 2:  # Short (position_type == 2)
            # 가격 하락 시 이익
            return (self.avg_entry_price - exit_price) / self.avg_entry_price * self.total_notional

    def _check_episode_end(self, current_price: float) -> Tuple[bool, bool, bool]:
        """
        에피소드 종료 조건 체크 (파산 로직 강화)
        """
        is_terminated = False
        is_liquidated = False
        is_bankruptcy = False
        
        # 1. Liquidation check (가격 기반 강제 청산)
        if self._check_liquidation(current_price):
            is_terminated = True
            is_liquidated = True
            
        # 2. Bankruptcy check (잔고 기반 파산 - 추가된 로직)
        # 자산이 초기 자본의 50% 미만으로 떨어지면 즉시 종료하여 마이너스 학습 방지
        if self.equity <= self.initial_balance * 0.5:
            is_terminated = True
            is_bankruptcy = True

        
        # 3. Episode length reached
        if self.step_count >= self.episode_length:
            is_terminated = True
            if self.position_type != 0:
                self._execute_flat(current_price)
                
        # 4. Data boundary check
        if self.current_index >= self.total_samples - 2:
            is_terminated = True
            if self.position_type != 0:
                self._execute_flat(current_price)
        
        return (is_terminated, is_liquidated, is_bankruptcy)

    def _check_liquidation(self, current_price: float) -> bool:
        """
        강제 청산 조건 체크

        청산 임계값은 레버리지에 따라 동적으로 계산됨:
        - liquidation_threshold = 0.5 / leverage
        - 레버리지 5배: 10% 역행 시 청산 (자산의 50% 손실)
        - 레버리지 20배: 2.5% 역행 시 청산 (자산의 50% 손실)

        청산 조건:
        - Long 포지션: current_price <= avg_entry_price × (1 - liquidation_threshold)
          예) 레버리지 5배, 진입가 $100,000 → $90,000 이하에서 청산
        - Short 포지션: current_price >= avg_entry_price × (1 + liquidation_threshold)
          예) 레버리지 5배, 진입가 $100,000 → $110,000 이상에서 청산

        Args:
            current_price: 현재 가격

        Returns:
            liquidated: True if liquidation triggered
        """
        if self.position_type == 0:  # Flat (포지션 없음)
            return False

        if self.position_type == 1:  # Long (롱 포지션)
            # 가격이 진입가 대비 liquidation_threshold만큼 하락하면 청산
            return current_price <= self.avg_entry_price * (1 - self.liquidation_threshold)
        else:  # Short (position_type == 2, 숏 포지션)
            # 가격이 진입가 대비 liquidation_threshold만큼 상승하면 청산
            return current_price >= self.avg_entry_price * (1 + self.liquidation_threshold)

    def _calculate_reward(
        self,
        prev_equity: float,
        current_equity: float,
        action: int,
        liquidated: bool,
        bankruptcy: bool,
        old_pos_type: int,
        old_entry_equity: float,
        old_hold_len: int,
        fee_cost: float
    ) -> float:
        """
        위험 조정 수익률 기반 보상 함수 (3-Phase 통합)

        Phase 1: Safety & Survival (안전성 및 생존)
        Phase 2: Risk-Adjusted Returns & Exploitation Fixes (위험 조정 수익률 및 착취 방지)
        Phase 3: Opportunity Costs & Compound Growth (기회비용 및 복리 성장)
        """
        reward = 0.0
        obs = self._get_observation()

        # 기본 equity 변화
        equity_change = current_equity - prev_equity

        # Phase 1: Critical Safety (안전성 최우선)
        if self.enable_phase1_safety:
            reward += self._calc_equity_change_base(prev_equity, current_equity)
            reward += self._calc_mdd_penalty(current_equity)
            reward += self._calc_global_mdd_penalty(current_equity) # [NEW] Global MDD 추가
            reward += self._calc_liquidation_penalty()
            reward += self._calc_margin_usage_penalty()
            reward += self._calc_pre_bankruptcy_warning(current_equity)

        # Phase 2: Risk-Adjusted Returns (위험 조정 수익률)
        if self.enable_phase2_risk_adj:
            reward = self._apply_risk_adjusted_scaling(reward, equity_change) # [MODIFY] Sortino 적용
            reward += self._calc_dynamic_no_position_penalty(obs)
            reward += self._calc_overtrading_penalty(fee_cost)
            reward += self._calc_asymmetric_hold_rewards(current_equity)

            # 포지션 유지 페널티 (기존)
            if self.position_type != 0:
                reward += self.time_penalty_with_position

        # Phase 3: Optimization (최적화)
        if self.enable_phase3_optimization:
            _, current_price, _ = self.data_generator.get_sequence(self.current_index - 1)
            reward += self._calc_opportunity_cost(current_price)
            reward += self._calc_compound_growth_bonus(current_equity)
            reward += self._calc_mfe_capture_bonus(action, old_pos_type, current_equity, old_entry_equity, obs)
            reward += self._calc_volatility_sizing_reward(action, obs)

        # 실현 수익 보너스 (기존 유지)
        if action == 1 and old_pos_type != 0:
            realized_return_pct = (current_equity - old_entry_equity) / old_entry_equity * 100.0
            if realized_return_pct > 0:
                reward += realized_return_pct * 0.5
            else:
                reward += realized_return_pct * 0.2

        # Terminal penalties override (치명적 실패)
        if liquidated or bankruptcy:
            reward = self.bankruptcy_penalty

        return np.clip(reward, -10.0, 10.0)

    # ========================================================================
    # Phase 1: Safety & Survival (안전성 및 생존) 보상 컴포넌트
    # ========================================================================

    def _calc_equity_change_base(self, prev_equity: float, current_equity: float) -> float:
        """기본 자산 변화율 보상 (Log Return 적용)"""
        if prev_equity <= 0 or current_equity <= 0:
            return 0.0
        # [CHANGE] Linear -> Logarithmic Return
        # 자연로그 수익률 * 100
        # 예: 10000 -> 11000 (10%) => ln(1.1) * 100 = 9.53
        # 예: 10000 -> 9000 (-10%) => ln(0.9) * 100 = -10.53
        equity_change_pct = np.log(current_equity / prev_equity) * 100.0
        return equity_change_pct

    def _calc_mdd_penalty(self, current_equity: float) -> float:
        """포지션 MDD 페널티 계산"""
        if self.position_type == 0:
            return 0.0
        mdd_ratio = (self.position_max_equity - current_equity) / self.position_max_equity
        if mdd_ratio > self.mdd_threshold:
            penalty = -self.mdd_penalty_coeff * ((mdd_ratio - self.mdd_threshold) ** 2) * 50.0
            return penalty
        return 0.0

    def _calc_global_mdd_penalty(self, current_equity: float) -> float:
        """[NEW] Global MDD 페널티 (에피소드 전체 기준) - 완화된 버전"""
        if self.global_max_equity <= 0:
            return 0.0
        
        # MDD 비율 계산
        global_mdd_ratio = (self.global_max_equity - current_equity) / self.global_max_equity
        
        # [CHANGE] 페널티 대폭 완화 (회생 가능성 부여)
        
        # 1. Warning Zone (5% ~ 15%) - 가벼운 경고
        # -0.01 ~ -0.1 수준의 페널티
        if 0.05 < global_mdd_ratio <= 0.15:
            penalty = -1.0 * (global_mdd_ratio - 0.05) 
            return penalty
            
        # 2. Danger Zone (> 15%) - 점진적 페널티 강화 (그러나 Exponential 아님)
        # 15% MDD -> -0.1
        # 25% MDD -> -0.1 - (0.10 * 5.0) = -0.6
        # 35% MDD -> -0.1 - (0.20 * 5.0) = -1.1
        elif global_mdd_ratio > 0.15:
            base_penalty = -0.1
            excess_penalty = -5.0 * (global_mdd_ratio - 0.15)
            penalty = base_penalty + excess_penalty
            # 하한선 (-5.0) 설정 (Death Spiral 방지)
            return max(penalty, -5.0)
            
        return 0.0



    def _calc_liquidation_penalty(self) -> float:
        """청산 거리 기반 페널티 계산"""
        if self.position_type == 0:
            return 0.0
        liq_distance = self.liquidation_distance_ratio
        if liq_distance < self.liquidation_danger_threshold:
            penalty = -self.liquidation_penalty_base * np.exp(3.0 * (self.liquidation_danger_threshold - liq_distance))
            return penalty
        return 0.0

    def _calc_margin_usage_penalty(self) -> float:
        """마진 과다 사용 페널티"""
        if self.position_type == 0:
            return 0.0
        margin_usage = self.num_layers / self.max_layers
        if margin_usage > self.margin_usage_threshold:
            penalty = -self.margin_penalty_coeff * ((margin_usage - self.margin_usage_threshold) ** 2) * 20.0
            return penalty
        return 0.0

    def _calc_pre_bankruptcy_warning(self, current_equity: float) -> float:
        """파산 전 조기 경고 페널티"""
        equity_ratio = current_equity / self.initial_balance
        if equity_ratio < self.pre_bankruptcy_threshold:
            penalty = -0.05 * (self.pre_bankruptcy_threshold - equity_ratio) * 100.0
            return penalty
        return 0.0

    # ========================================================================
    # Phase 2: Risk-Adjusted Returns (위험 조정 수익률) 보상 컴포넌트
    # ========================================================================

    def _apply_risk_adjusted_scaling(self, base_reward: float, equity_change: float) -> float:
        """[MODIFY] Sortino Ratio 기반 Additive Penalty (하방 변동성 차감)"""
        # 데이터가 충분하지 않으면 보상 변경 없음 (단, 초기 생존 페널티는 유지)
        if len(self.equity_history) < 20:
            return base_reward
            
        # Equity 변화율 계산
        equity_changes = np.diff(self.equity_history)
        
        # 하방 변동성(Downside Deviation) 계산
        negative_changes = equity_changes[equity_changes < 0]
        
        if len(negative_changes) > 0:
            downside_std = np.std(negative_changes)
        else:
            downside_std = 0.0
            
        # 최소 노이즈 필터링
        min_std = 0.001 * self.initial_balance
        if downside_std < min_std:
            return base_reward
        
        # [CHANGE] Multiplicative -> Additive (뺄셈 방식)
        # 리스크(하방변동성)가 클수록 점수를 깎음.
        # 수익이 났든 손실이 났든, 변동성이 큰 상태는 바람직하지 않음.
        # Penalty = Downside_Std * Coeff
        
        # 스케일링 팩터 정규화 (자산 규모 대비 %로 변환해서 적용)
        # 1.0 = 자산의 1%에 해당하는 하방 표준편차당 1.0점 감점
        volatility_penalty = (downside_std / self.initial_balance) * self.sortino_penalty_coeff * 100.0
        
        # 최종 보상 = 기본 보상 - 변동성 페널티
        adjusted_reward = base_reward - volatility_penalty
        
        return adjusted_reward

    def _calc_dynamic_no_position_penalty(self, obs: Dict) -> float:
        """컨텍스트 인식 무포지션 페널티 (대폭 축소)"""
        if self.position_type != 0:
            return 0.0
            
        # [CHANGE] Remove aggressive forced trading penalty
        # 단순히 아주 작은 상수를 부여하여 Action 0(Hold)와 Action 1(Flat)의 우연한 동점을 방지
        return -0.0001

    def _calc_overtrading_penalty(self, fee_cost: float) -> float:
        """수수료 기반 과다거래 페널티"""
        penalty = 0.0
        if fee_cost > 0:
            fee_penalty = -(fee_cost / self.initial_balance) * 100.0 * 1.5
            penalty += fee_penalty
        fee_ratio = self.episode_fees_paid / self.initial_balance
        if fee_ratio > self.overtrading_threshold:
            excessive_penalty = -10.0 * ((fee_ratio - self.overtrading_threshold) ** 2) * 1000.0
            penalty += excessive_penalty
        return penalty

    def _calc_asymmetric_hold_rewards(self, current_equity: float) -> float:
        """포지션 보유 시간 기반 비대칭 보상"""
        if self.position_type == 0:
            return 0.0
        unrealized_pnl_pct = (current_equity - self.entry_equity) / self.entry_equity * 100.0
        hold_duration = self.is_open_position_len
        if unrealized_pnl_pct < 0:
            penalty = -self.asymmetric_hold_loss_coeff * np.log1p(hold_duration / 5.0)
            return penalty
        else:
            bonus = self.asymmetric_hold_profit_coeff * np.log1p(hold_duration / 10.0) * np.sqrt(unrealized_pnl_pct)
            return bonus

    # ========================================================================
    # Phase 3: Opportunity Costs & Compound Growth (기회비용 및 복리 성장)
    # ========================================================================

    def _calc_opportunity_cost(self, current_price: float) -> float:
        """놓친 거래 기회 비용 계산"""
        if self.position_type != 0 or self.prev_price == 0:
            return 0.0
        price_change_pct = (current_price - self.prev_price) / self.prev_price
        notional = self.initial_balance * 0.25 * self.leverage
        best_opportunity = max(price_change_pct * notional, -price_change_pct * notional)
        threshold = self.opportunity_cost_threshold * self.initial_balance
        if best_opportunity > threshold:
            penalty = -self.opportunity_cost_coeff * (best_opportunity / self.initial_balance) * 100.0
            return penalty
        return 0.0

    def _calc_compound_growth_bonus(self, current_equity: float) -> float:
        """복리 성장 보너스 (로그 스케일)"""
        equity_ratio = current_equity / self.initial_balance
        if equity_ratio > 1.0:
            bonus = self.compound_bonus_coeff * np.log(equity_ratio)
            return bonus
        return 0.0

    def _calc_mfe_capture_bonus(self, action: int, old_pos_type: int, current_equity: float, old_entry_equity: float, obs: Dict) -> float:
        """MFE 포착률 평가"""
        if action != 1 or old_pos_type == 0:
            return 0.0
        mfe_ratio = obs['agent'][11]
        final_return = (current_equity - old_entry_equity) / old_entry_equity
        # [NEW] MFE가 너무 작으면(0.5% 미만) 노이즈로 간주하고 스킵
        if mfe_ratio < 0.005:
            return 0.0
        capture_ratio = final_return / mfe_ratio
        if capture_ratio > self.mfe_capture_threshold:
            bonus = self.mfe_capture_bonus_coeff * (capture_ratio - self.mfe_capture_threshold)
            return bonus
        else:
            penalty = -self.mfe_capture_penalty_coeff * (self.mfe_capture_threshold - capture_ratio)
            return penalty

    def _calc_volatility_sizing_reward(self, action: int, obs: Dict) -> float:
        """변동성 기반 포지션 사이징 평가"""
        if action not in [2, 3]:
            return 0.0
        volatility = obs['agent'][6]
        margin_usage = obs['agent'][10]
        ideal_margin = 0.25 + 0.5 * (1.0 - volatility)
        sizing_error = abs(margin_usage - ideal_margin)
        if sizing_error < self.volatility_sizing_tolerance:
            bonus = self.volatility_sizing_bonus_coeff * (self.volatility_sizing_tolerance - sizing_error)
            return bonus
        else:
            penalty = -self.volatility_sizing_penalty_coeff * (sizing_error - self.volatility_sizing_tolerance)
            return penalty
