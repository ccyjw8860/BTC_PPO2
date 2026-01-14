"""
보상 컴포넌트 메서드 (trading_env2.py에 추가할 코드)

이 파일의 메서드들을 trading_env2.py의 TradingEnv 클래스에 복사하세요.
"""

# ========================================================================
# Phase 1: Safety & Survival (안전성 및 생존) 보상 컴포넌트
# ========================================================================

def _calc_equity_change_base(self, prev_equity: float, current_equity: float) -> float:
    """
    기본 자산 변화율 보상

    Returns:
        float: 백분율 스케일 보상
    """
    equity_change_pct = (current_equity - prev_equity) / self.initial_balance * 100.0
    return equity_change_pct

def _calc_mdd_penalty(self, current_equity: float) -> float:
    """
    포지션 MDD (Maximum Drawdown) 페널티 계산

    목적: 포지션 내 최고점 대비 하락률을 페널티화하여 조기 손절 유도

    Returns:
        float: 페널티 (항상 <= 0)
    """
    if self.position_type == 0:
        return 0.0

    # 현재 포지션 내 최고 equity 대비 하락률
    mdd_ratio = (self.position_max_equity - current_equity) / self.position_max_equity

    if mdd_ratio > self.mdd_threshold:  # 기본 5% 임계값
        # 제곱 페널티: 하락폭이 클수록 급격히 증가
        penalty = -self.mdd_penalty_coeff * ((mdd_ratio - self.mdd_threshold) ** 2) * 50.0
        return penalty

    return 0.0

def _calc_liquidation_penalty(self) -> float:
    """
    청산 거리 기반 페널티 계산

    목적: 청산가 근접 시 지수함수적 공포 학습

    Uses:
        self.liquidation_distance_ratio (이미 관측에 계산됨)

    Returns:
        float: 페널티 (항상 <= 0)
    """
    if self.position_type == 0:
        return 0.0

    liq_distance = self.liquidation_distance_ratio

    if liq_distance < self.liquidation_danger_threshold:  # 기본 50% 이하 = 위험 구간
        # 지수 페널티: 청산가에 가까울수록 폭발적 증가
        import numpy as np
        penalty = -self.liquidation_penalty_base * np.exp(3.0 * (self.liquidation_danger_threshold - liq_distance))
        return penalty

    return 0.0

def _calc_margin_usage_penalty(self) -> float:
    """
    마진 과다 사용 페널티

    목적: 과도한 레버리지 사용(4레이어 풀포지션) 억제

    Returns:
        float: 페널티 (항상 <= 0)
    """
    if self.position_type == 0:
        return 0.0

    margin_usage = self.num_layers / self.max_layers

    if margin_usage > self.margin_usage_threshold:  # 기본 75% 초과 (3레이어 초과)
        # 제곱 페널티
        penalty = -self.margin_penalty_coeff * ((margin_usage - self.margin_usage_threshold) ** 2) * 20.0
        return penalty

    return 0.0

def _calc_pre_bankruptcy_warning(self, current_equity: float) -> float:
    """
    파산 전 조기 경고 페널티

    목적: Equity가 70% 이하로 떨어지면 보수적 행동 유도

    Returns:
        float: 페널티 (항상 <= 0)
    """
    equity_ratio = current_equity / self.initial_balance

    if equity_ratio < self.pre_bankruptcy_threshold:  # 기본 70% (30% 손실)
        # 선형 페널티 (파산은 50%에서 -10.0이므로 보완적)
        penalty = -0.05 * (self.pre_bankruptcy_threshold - equity_ratio) * 100.0
        return penalty

    return 0.0

# ========================================================================
# Phase 2: Risk-Adjusted Returns (위험 조정 수익률) 보상 컴포넌트
# ========================================================================

def _apply_sharpe_scaling(self, base_reward: float, equity_change: float) -> float:
    """
    Sharpe-inspired 위험 조정 스케일링

    목적: 변동성 대비 수익률로 보상 정규화

    Uses:
        self.equity_history (롤링 50-window)

    Returns:
        float: 조정된 보상
    """
    if len(self.equity_history) < 20:
        # 초기 20 스텝은 기본 보상 사용
        return base_reward

    # 최근 equity 변동성 계산
    import numpy as np
    equity_std = np.std(self.equity_history[-self.equity_window_size:])

    # 최소 변동성 설정 (0으로 나누기 방지)
    min_std = 0.01 * self.initial_balance  # $100
    equity_std = max(equity_std, min_std)

    # 위험 조정 계수: 높은 수익 + 낮은 변동성 = 2x 배수
    risk_adjusted_factor = 1.0 + (equity_change / equity_std) * self.sharpe_scaling_factor
    risk_adjusted_factor = np.clip(risk_adjusted_factor, 0.5, 2.0)

    return base_reward * risk_adjusted_factor

def _calc_dynamic_no_position_penalty(self, obs) -> float:
    """
    컨텍스트 인식 무포지션 페널티

    목적: 추세 강도와 변동성에 따라 관망 페널티 조정 (lazy agent 해결)

    Args:
        obs: 현재 관측 (agent state 필요)

    Returns:
        float: 페널티 (대부분 < 0, 예외적으로 > 0)
    """
    if self.position_type != 0:
        return 0.0

    # 관측 공간에서 feature 추출
    trend_strength = abs(obs['agent'][7])  # 0~1
    volatility = obs['agent'][6]  # 0~1

    # 컨텍스트 기반 페널티 계산
    # - 강한 추세: 큰 페널티 (기회 놓침)
    # - 높은 변동성: 페널티 완화 (위험 회피 정당화)
    penalty = -self.dynamic_penalty_base - (self.dynamic_penalty_trend_coeff * trend_strength) + (self.dynamic_penalty_vol_coeff * volatility)

    return penalty

def _calc_overtrading_penalty(self, fee_cost: float) -> float:
    """
    수수료 기반 과다거래 페널티

    목적: 수수료를 심리적으로 "비싸게" 느끼게 하여 빈번한 매매 억제

    Args:
        fee_cost: 현재 스텝의 수수료 ($)

    Returns:
        float: 페널티 (항상 <= 0)
    """
    penalty = 0.0

    # 1. 즉시 수수료 페널티 (1.5배 증폭)
    if fee_cost > 0:
        fee_penalty = -(fee_cost / self.initial_balance) * 100.0 * 1.5
        penalty += fee_penalty

    # 2. 누적 수수료 기반 과다거래 페널티
    fee_ratio = self.episode_fees_paid / self.initial_balance
    if fee_ratio > self.overtrading_threshold:  # 기본 2% = 약 40회 거래 (1회당 0.05% 수수료)
        excessive_penalty = -10.0 * ((fee_ratio - self.overtrading_threshold) ** 2) * 1000.0
        penalty += excessive_penalty

    return penalty

def _calc_asymmetric_hold_rewards(self, current_equity: float) -> float:
    """
    포지션 보유 시간 기반 비대칭 보상

    목적: 수익 포지션은 길게, 손실 포지션은 빠르게 청산하도록 유도

    Returns:
        float: 보상/페널티
    """
    if self.position_type == 0:
        return 0.0

    unrealized_pnl_pct = (current_equity - self.entry_equity) / self.entry_equity * 100.0
    hold_duration = self.is_open_position_len

    import numpy as np
    if unrealized_pnl_pct < 0:
        # 손실 중: 빠른 손절 유도
        penalty = -self.asymmetric_hold_loss_coeff * np.log1p(hold_duration / 5.0)
        return penalty
    else:
        # 수익 중: 추세 추종 장려
        bonus = self.asymmetric_hold_profit_coeff * np.log1p(hold_duration / 10.0) * np.sqrt(unrealized_pnl_pct)
        return bonus

# ========================================================================
# Phase 3: Opportunity Costs & Compound Growth (기회비용 및 복리 성장) 보상 컴포넌트
# ========================================================================

def _calc_opportunity_cost(self, current_price: float) -> float:
    """
    놓친 거래 기회 비용 계산

    목적: 무포지션 상태에서 큰 가격 변동 발생 시 페널티

    Returns:
        float: 페널티 (항상 <= 0)
    """
    if self.position_type != 0 or self.prev_price == 0:
        return 0.0

    # 가격 변동률 계산
    price_change_pct = (current_price - self.prev_price) / self.prev_price

    # 1레이어 기준 잠재 수익 계산
    notional = self.initial_balance * 0.25 * self.leverage

    # Long과 Short 중 더 유리한 쪽 선택
    best_opportunity = max(
        price_change_pct * notional,  # Long
        -price_change_pct * notional   # Short
    )

    # 임계값 이상의 기회만 페널티화 (노이즈 필터)
    threshold = self.opportunity_cost_threshold * self.initial_balance  # 기본 0.2% = $20
    if best_opportunity > threshold:
        penalty = -self.opportunity_cost_coeff * (best_opportunity / self.initial_balance) * 100.0
        return penalty

    return 0.0

def _calc_compound_growth_bonus(self, current_equity: float) -> float:
    """
    복리 성장 보너스 (로그 스케일)

    목적: 로그 스케일 수익률로 기하급수적 성장 유도

    Returns:
        float: 보너스 (>= 0)
    """
    equity_ratio = current_equity / self.initial_balance

    if equity_ratio > 1.0:
        import numpy as np
        bonus = self.compound_bonus_coeff * np.log(equity_ratio)
        return bonus

    return 0.0

def _calc_mfe_capture_bonus(
    self,
    action: int,
    old_pos_type: int,
    current_equity: float,
    old_entry_equity: float,
    obs
) -> float:
    """
    MFE (Maximum Favorable Excursion) 포착률 평가

    목적: 포지션 최고점 대비 실현 수익 비율 평가 (수익 극대화 학습)
    청산 시점에만 작동

    Returns:
        float: 보너스/페널티
    """
    # 청산 액션이 아니면 0
    if action != 1 or old_pos_type == 0:
        return 0.0

    mfe_ratio = obs['agent'][11]  # 이미 계산된 MFE
    final_return = (current_equity - old_entry_equity) / old_entry_equity

    # MFE가 의미 없을 정도로 작으면 무시
    if mfe_ratio < 0.01:  # 1% 미만
        return 0.0

    # 포착률 = 실현수익 / 최대수익
    capture_ratio = final_return / mfe_ratio

    if capture_ratio > self.mfe_capture_threshold:  # 기본 50% 이상 포착
        bonus = self.mfe_capture_bonus_coeff * (capture_ratio - self.mfe_capture_threshold)
        return bonus
    else:  # 50% 미만 포착 (너무 일찍 or 너무 늦게 청산)
        penalty = -self.mfe_capture_penalty_coeff * (self.mfe_capture_threshold - capture_ratio)
        return penalty

def _calc_volatility_sizing_reward(self, action: int, obs) -> float:
    """
    변동성 기반 포지션 사이징 평가

    목적: 변동성에 맞는 적절한 레이어 수 선택 보상
    진입 시점에만 작동

    Returns:
        float: 보너스/페널티
    """
    # 진입 액션이 아니면 0
    if action not in [2, 3]:  # Long or Short
        return 0.0

    volatility = obs['agent'][6]  # 0~1
    margin_usage = obs['agent'][10]  # 0~1

    # 이상적 마진 사용률: 변동성이 낮을수록 높게
    # 낮은 vol (0.2) → ideal = 0.65
    # 중간 vol (0.5) → ideal = 0.50
    # 높은 vol (0.8) → ideal = 0.35
    ideal_margin = 0.25 + 0.5 * (1.0 - volatility)

    # 실제 마진 사용과의 차이
    sizing_error = abs(margin_usage - ideal_margin)

    if sizing_error < self.volatility_sizing_tolerance:  # 기본 20% 이내
        bonus = self.volatility_sizing_bonus_coeff * (self.volatility_sizing_tolerance - sizing_error)
        return bonus
    else:  # 오차 큼
        penalty = -self.volatility_sizing_penalty_coeff * (sizing_error - self.volatility_sizing_tolerance)
        return penalty
