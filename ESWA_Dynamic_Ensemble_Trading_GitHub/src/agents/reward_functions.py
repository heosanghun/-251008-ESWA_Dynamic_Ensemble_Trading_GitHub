"""
체제별 맞춤형 보상함수 모듈
Regime-Specific Reward Functions Module

각 시장 체제에 특화된 보상함수 설계
- Bull Market: 수익극대화 (단순 포트폴리오 수익)
- Bear Market: 손실최소화 (Sortino Ratio 기반)
- Sideways Market: 거래비용 최소화 (거래 억제)

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod
from enum import Enum


class MarketRegime(Enum):
    """시장 체제 열거형"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"


class BaseRewardFunction(ABC):
    """
    기본 보상함수 추상 클래스
    
    모든 체제별 보상함수의 기본 인터페이스
    """
    
    def __init__(self, config: Dict):
        """
        기본 보상함수 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, 
                        done: bool) -> float:
        """
        보상 계산 (추상 메서드)
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            계산된 보상
        """
        pass
    
    def get_reward_components(self, state: Dict, action: int, next_state: Dict, 
                            done: bool) -> Dict[str, float]:
        """
        보상 구성 요소 반환
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            보상 구성 요소 딕셔너리
        """
        return {
            'total_reward': self.calculate_reward(state, action, next_state, done)
        }


class BullMarketRewardFunction(BaseRewardFunction):
    """
    상승장 보상함수
    
    수익극대화에 중점을 둔 보상함수
    - 단순 포트폴리오 수익 기반
    - 상승 추세 추종 보상
    - 공격적 거래 장려
    """
    
    def __init__(self, config: Dict):
        """
        상승장 보상함수 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 보상 가중치
        self.return_weight = 1.0
        self.trend_weight = 0.3
        self.aggression_weight = 0.2
        self.risk_penalty = 0.1
        
        # 액션 가중치 (공격적 거래 장려)
        self.action_weights = {
            0: 1.2,  # Strong Buy
            1: 1.0,  # Buy
            2: 0.5,  # Hold
            3: -0.5, # Sell
            4: -1.0  # Strong Sell
        }
        
        self.logger.info("상승장 보상함수 초기화 완료")
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, 
                        done: bool) -> float:
        """
        상승장 보상 계산
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            계산된 보상
        """
        try:
            # 1. 포트폴리오 수익 보상
            portfolio_return = self._calculate_portfolio_return(state, next_state)
            return_reward = self.return_weight * portfolio_return
            
            # 2. 추세 추종 보상
            trend_reward = self._calculate_trend_following_reward(state, action, next_state)
            trend_reward = self.trend_weight * trend_reward
            
            # 3. 공격적 거래 보상
            aggression_reward = self._calculate_aggression_reward(action, portfolio_return)
            aggression_reward = self.aggression_weight * aggression_reward
            
            # 4. 위험 페널티
            risk_penalty = self._calculate_risk_penalty(state, next_state)
            risk_penalty = self.risk_penalty * risk_penalty
            
            # 총 보상
            total_reward = return_reward + trend_reward + aggression_reward - risk_penalty
            
            return float(total_reward)
            
        except Exception as e:
            self.logger.error(f"상승장 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_portfolio_return(self, state: Dict, next_state: Dict) -> float:
        """포트폴리오 수익 계산"""
        try:
            current_value = state.get('portfolio_value', 10000)
            next_value = next_state.get('portfolio_value', current_value)
            
            if current_value > 0:
                return (next_value - current_value) / current_value
            return 0.0
            
        except Exception as e:
            self.logger.error(f"포트폴리오 수익 계산 실패: {e}")
            return 0.0
    
    def _calculate_trend_following_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """추세 추종 보상 계산"""
        try:
            # 가격 변화율
            current_price = state.get('current_price', 0)
            next_price = next_state.get('current_price', current_price)
            
            if current_price > 0:
                price_change = (next_price - current_price) / current_price
            else:
                price_change = 0.0
            
            # 액션별 추세 추종 점수
            if action in [0, 1] and price_change > 0:  # 매수 + 상승
                return abs(price_change)
            elif action in [3, 4] and price_change < 0:  # 매도 + 하락
                return abs(price_change)
            elif action == 2:  # 홀드
                return 0.0
            else:  # 추세 반대 거래
                return -abs(price_change) * 0.5
            
        except Exception as e:
            self.logger.error(f"추세 추종 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_aggression_reward(self, action: int, portfolio_return: float) -> float:
        """공격적 거래 보상 계산"""
        try:
            # 액션 가중치 적용
            action_reward = self.action_weights.get(action, 0.0)
            
            # 수익이 있을 때 공격적 거래 보상 증가
            if portfolio_return > 0:
                return action_reward * (1 + portfolio_return)
            else:
                return action_reward * 0.5
            
        except Exception as e:
            self.logger.error(f"공격적 거래 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_risk_penalty(self, state: Dict, next_state: Dict) -> float:
        """위험 페널티 계산"""
        try:
            # 변동성 기반 위험 페널티
            current_price = state.get('current_price', 0)
            next_price = next_state.get('current_price', current_price)
            
            if current_price > 0:
                price_change = abs((next_price - current_price) / current_price)
                # 높은 변동성에 페널티
                return price_change * 0.1
            return 0.0
            
        except Exception as e:
            self.logger.error(f"위험 페널티 계산 실패: {e}")
            return 0.0
    
    def get_reward_components(self, state: Dict, action: int, next_state: Dict, 
                            done: bool) -> Dict[str, float]:
        """상승장 보상 구성 요소 반환"""
        try:
            portfolio_return = self._calculate_portfolio_return(state, next_state)
            trend_reward = self._calculate_trend_following_reward(state, action, next_state)
            aggression_reward = self._calculate_aggression_reward(action, portfolio_return)
            risk_penalty = self._calculate_risk_penalty(state, next_state)
            
            return {
                'portfolio_return': portfolio_return,
                'trend_following': trend_reward,
                'aggression': aggression_reward,
                'risk_penalty': risk_penalty,
                'total_reward': portfolio_return + 0.3 * trend_reward + 0.2 * aggression_reward - 0.1 * risk_penalty
            }
            
        except Exception as e:
            self.logger.error(f"보상 구성 요소 계산 실패: {e}")
            return {'total_reward': 0.0}


class BearMarketRewardFunction(BaseRewardFunction):
    """
    하락장 보상함수
    
    손실최소화에 중점을 둔 보상함수
    - Sortino Ratio 기반
    - 하방 위험 관리
    - 방어적 거래 장려
    """
    
    def __init__(self, config: Dict):
        """
        하락장 보상함수 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 보상 가중치
        self.sortino_weight = 1.0
        self.downside_protection_weight = 0.5
        self.defensive_weight = 0.3
        self.volatility_penalty = 0.2
        
        # 액션 가중치 (방어적 거래 장려)
        self.action_weights = {
            0: -0.5, # Strong Buy (위험)
            1: -0.2, # Buy (위험)
            2: 0.8,  # Hold (안전)
            3: 1.0,  # Sell (방어)
            4: 1.2   # Strong Sell (강력한 방어)
        }
        
        # Sortino Ratio 계산을 위한 변수
        self.returns_history = []
        self.max_history_length = 100
        
        self.logger.info("하락장 보상함수 초기화 완료")
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, 
                        done: bool) -> float:
        """
        하락장 보상 계산
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            계산된 보상
        """
        try:
            # 1. Sortino Ratio 기반 보상
            sortino_reward = self._calculate_sortino_reward(state, next_state)
            sortino_reward = self.sortino_weight * sortino_reward
            
            # 2. 하방 보호 보상
            downside_protection = self._calculate_downside_protection(state, action, next_state)
            downside_protection = self.downside_protection_weight * downside_protection
            
            # 3. 방어적 거래 보상
            defensive_reward = self._calculate_defensive_reward(action, state, next_state)
            defensive_reward = self.defensive_weight * defensive_reward
            
            # 4. 변동성 페널티
            volatility_penalty = self._calculate_volatility_penalty(state, next_state)
            volatility_penalty = self.volatility_penalty * volatility_penalty
            
            # 총 보상
            total_reward = sortino_reward + downside_protection + defensive_reward - volatility_penalty
            
            return float(total_reward)
            
        except Exception as e:
            self.logger.error(f"하락장 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_sortino_reward(self, state: Dict, next_state: Dict) -> float:
        """Sortino Ratio 기반 보상 계산"""
        try:
            # 포트폴리오 수익 계산
            current_value = state.get('portfolio_value', 10000)
            next_value = next_state.get('portfolio_value', current_value)
            
            if current_value > 0:
                portfolio_return = (next_value - current_value) / current_value
            else:
                portfolio_return = 0.0
            
            # 수익 히스토리 업데이트
            self.returns_history.append(portfolio_return)
            if len(self.returns_history) > self.max_history_length:
                self.returns_history.pop(0)
            
            if len(self.returns_history) < 2:
                return portfolio_return
            
            # Sortino Ratio 계산
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            
            # 하방 편차 (음수 수익만 고려)
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
            else:
                downside_deviation = 0.001  # 0으로 나누기 방지
            
            # Sortino Ratio
            if downside_deviation > 0:
                sortino_ratio = mean_return / downside_deviation
            else:
                sortino_ratio = mean_return * 100  # 높은 보상
            
            return sortino_ratio * 0.1  # 스케일링
            
        except Exception as e:
            self.logger.error(f"Sortino Ratio 계산 실패: {e}")
            return 0.0
    
    def _calculate_downside_protection(self, state: Dict, action: int, next_state: Dict) -> float:
        """하방 보호 보상 계산"""
        try:
            # 가격 변화율
            current_price = state.get('current_price', 0)
            next_price = next_state.get('current_price', current_price)
            
            if current_price > 0:
                price_change = (next_price - current_price) / current_price
            else:
                price_change = 0.0
            
            # 하락 시 방어적 액션 보상
            if price_change < -0.01:  # 1% 이상 하락
                if action in [3, 4]:  # 매도 액션
                    return abs(price_change) * 2.0  # 강한 보상
                elif action == 2:  # 홀드
                    return abs(price_change) * 0.5  # 약한 보상
                else:  # 매수 액션
                    return -abs(price_change) * 1.0  # 페널티
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"하방 보호 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_defensive_reward(self, action: int, state: Dict, next_state: Dict) -> float:
        """방어적 거래 보상 계산"""
        try:
            # 액션 가중치 적용
            action_reward = self.action_weights.get(action, 0.0)
            
            # 시장 상황에 따른 보상 조정
            current_price = state.get('current_price', 0)
            next_price = next_state.get('current_price', current_price)
            
            if current_price > 0:
                price_change = (next_price - current_price) / current_price
                
                # 하락장에서 방어적 액션 보상 증가
                if price_change < 0:
                    return action_reward * (1 + abs(price_change))
                else:
                    return action_reward * 0.5
            
            return action_reward
            
        except Exception as e:
            self.logger.error(f"방어적 거래 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_volatility_penalty(self, state: Dict, next_state: Dict) -> float:
        """변동성 페널티 계산"""
        try:
            # 가격 변동성
            current_price = state.get('current_price', 0)
            next_price = next_state.get('current_price', current_price)
            
            if current_price > 0:
                price_change = abs((next_price - current_price) / current_price)
                # 높은 변동성에 강한 페널티
                return price_change * 0.5
            return 0.0
            
        except Exception as e:
            self.logger.error(f"변동성 페널티 계산 실패: {e}")
            return 0.0


class SidewaysMarketRewardFunction(BaseRewardFunction):
    """
    횡보장 보상함수
    
    거래비용 최소화에 중점을 둔 보상함수
    - 거래 억제 보상
    - 기회비용 최소화
    - 현금 보유 장려
    """
    
    def __init__(self, config: Dict):
        """
        횡보장 보상함수 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 보상 가중치
        self.trading_cost_weight = 1.0
        self.opportunity_cost_weight = 0.5
        self.cash_holding_weight = 0.3
        self.stability_weight = 0.2
        
        # 액션 가중치 (거래 억제)
        self.action_weights = {
            0: -0.8, # Strong Buy (비용 발생)
            1: -0.5, # Buy (비용 발생)
            2: 1.0,  # Hold (비용 없음)
            3: -0.5, # Sell (비용 발생)
            4: -0.8  # Strong Sell (비용 발생)
        }
        
        # 거래 비용 설정
        self.trading_config = config.get('trading', {})
        self.buy_fee = self.trading_config.get('fees', {}).get('buy', 0.0005)
        self.sell_fee = self.trading_config.get('fees', {}).get('sell', 0.0005)
        
        self.logger.info("횡보장 보상함수 초기화 완료")
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, 
                        done: bool) -> float:
        """
        횡보장 보상 계산
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            계산된 보상
        """
        try:
            # 1. 거래 비용 페널티
            trading_cost_penalty = self._calculate_trading_cost_penalty(action, state, next_state)
            trading_cost_penalty = self.trading_cost_weight * trading_cost_penalty
            
            # 2. 기회비용 보상
            opportunity_cost_reward = self._calculate_opportunity_cost_reward(state, action, next_state)
            opportunity_cost_reward = self.opportunity_cost_weight * opportunity_cost_reward
            
            # 3. 현금 보유 보상
            cash_holding_reward = self._calculate_cash_holding_reward(action, state)
            cash_holding_reward = self.cash_holding_weight * cash_holding_reward
            
            # 4. 안정성 보상
            stability_reward = self._calculate_stability_reward(state, next_state)
            stability_reward = self.stability_weight * stability_reward
            
            # 총 보상
            total_reward = -trading_cost_penalty + opportunity_cost_reward + cash_holding_reward + stability_reward
            
            return float(total_reward)
            
        except Exception as e:
            self.logger.error(f"횡보장 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_trading_cost_penalty(self, action: int, state: Dict, next_state: Dict) -> float:
        """거래 비용 페널티 계산"""
        try:
            # 거래 액션에 따른 비용
            if action in [0, 1]:  # 매수 액션
                cost = self.buy_fee
            elif action in [3, 4]:  # 매도 액션
                cost = self.sell_fee
            else:  # 홀드
                cost = 0.0
            
            # 거래 규모에 따른 비용 조정
            portfolio_value = state.get('portfolio_value', 10000)
            if portfolio_value > 0:
                cost_ratio = cost * (portfolio_value / 10000)  # 포트폴리오 크기에 비례
            else:
                cost_ratio = cost
            
            return cost_ratio
            
        except Exception as e:
            self.logger.error(f"거래 비용 페널티 계산 실패: {e}")
            return 0.0
    
    def _calculate_opportunity_cost_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """기회비용 보상 계산"""
        try:
            # 가격 변화율
            current_price = state.get('current_price', 0)
            next_price = next_state.get('current_price', current_price)
            
            if current_price > 0:
                price_change = abs((next_price - current_price) / current_price)
            else:
                price_change = 0.0
            
            # 횡보장에서 큰 가격 변화가 없을 때 홀드 보상
            if price_change < 0.005:  # 0.5% 미만 변화
                if action == 2:  # 홀드
                    return 0.01  # 작은 보상
                else:  # 거래
                    return -0.005  # 작은 페널티
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"기회비용 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_cash_holding_reward(self, action: int, state: Dict) -> float:
        """현금 보유 보상 계산"""
        try:
            # 액션 가중치 적용
            action_reward = self.action_weights.get(action, 0.0)
            
            # 현금 비율에 따른 보상 조정
            cash_ratio = state.get('cash_ratio', 0.5)
            
            # 현금 비율이 높을 때 홀드 보상 증가
            if action == 2:  # 홀드
                return action_reward * (1 + cash_ratio)
            else:  # 거래
                return action_reward * (1 - cash_ratio)
            
        except Exception as e:
            self.logger.error(f"현금 보유 보상 계산 실패: {e}")
            return 0.0
    
    def _calculate_stability_reward(self, state: Dict, next_state: Dict) -> float:
        """안정성 보상 계산"""
        try:
            # 포트폴리오 가치 안정성
            current_value = state.get('portfolio_value', 10000)
            next_value = next_state.get('portfolio_value', current_value)
            
            if current_value > 0:
                value_change = abs((next_value - current_value) / current_value)
                # 가치 변화가 작을 때 보상
                if value_change < 0.01:  # 1% 미만 변화
                    return 0.005
                else:
                    return -value_change * 0.1
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"안정성 보상 계산 실패: {e}")
            return 0.0


class RegimeSpecificRewards:
    """
    체제별 맞춤형 보상함수 관리자
    
    시장 체제에 따라 적절한 보상함수 선택 및 적용
    """
    
    def __init__(self, config: Dict):
        """
        체제별 보상함수 관리자 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 체제별 보상함수 초기화
        self.reward_functions = {
            MarketRegime.BULL_MARKET: BullMarketRewardFunction(config),
            MarketRegime.BEAR_MARKET: BearMarketRewardFunction(config),
            MarketRegime.SIDEWAYS_MARKET: SidewaysMarketRewardFunction(config)
        }
        
        # 현재 체제
        self.current_regime = MarketRegime.SIDEWAYS_MARKET
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("체제별 맞춤형 보상함수 관리자 초기화 완료")
    
    def set_regime(self, regime: Union[str, MarketRegime]):
        """
        현재 체제 설정
        
        Args:
            regime: 시장 체제
        """
        try:
            if isinstance(regime, str):
                regime = MarketRegime(regime)
            
            if regime != self.current_regime:
                self.logger.info(f"체제 변경: {self.current_regime.value} → {regime.value}")
                self.current_regime = regime
            
        except Exception as e:
            self.logger.error(f"체제 설정 실패: {e}")
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, 
                        done: bool) -> float:
        """
        현재 체제에 맞는 보상 계산
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            계산된 보상
        """
        try:
            reward_function = self.reward_functions[self.current_regime]
            return reward_function.calculate_reward(state, action, next_state, done)
            
        except Exception as e:
            self.logger.error(f"보상 계산 실패: {e}")
            return 0.0
    
    def get_reward_components(self, state: Dict, action: int, next_state: Dict, 
                            done: bool) -> Dict[str, float]:
        """
        보상 구성 요소 반환
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            보상 구성 요소 딕셔너리
        """
        try:
            reward_function = self.reward_functions[self.current_regime]
            components = reward_function.get_reward_components(state, action, next_state, done)
            components['regime'] = self.current_regime.value
            return components
            
        except Exception as e:
            self.logger.error(f"보상 구성 요소 계산 실패: {e}")
            return {'total_reward': 0.0, 'regime': self.current_regime.value}
    
    def get_regime_info(self) -> Dict[str, str]:
        """현재 체제 정보 반환"""
        return {
            'current_regime': self.current_regime.value,
            'available_regimes': [regime.value for regime in MarketRegime]
        }


# 편의 함수들
def create_regime_specific_rewards(config: Dict) -> RegimeSpecificRewards:
    """체제별 맞춤형 보상함수 관리자 생성 편의 함수"""
    return RegimeSpecificRewards(config)


def create_reward_function(regime: Union[str, MarketRegime], config: Dict) -> BaseRewardFunction:
    """특정 체제의 보상함수 생성 편의 함수"""
    if isinstance(regime, str):
        regime = MarketRegime(regime)
    
    if regime == MarketRegime.BULL_MARKET:
        return BullMarketRewardFunction(config)
    elif regime == MarketRegime.BEAR_MARKET:
        return BearMarketRewardFunction(config)
    elif regime == MarketRegime.SIDEWAYS_MARKET:
        return SidewaysMarketRewardFunction(config)
    else:
        raise ValueError(f"지원하지 않는 체제: {regime}")
