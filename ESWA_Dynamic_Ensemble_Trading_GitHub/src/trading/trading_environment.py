"""
거래 환경 모듈
Trading Environment Module

거래 환경 구현 및 포트폴리오 관리
- 포트폴리오 상태 관리 (현금, 주식, 총 자산)
- 거래 수수료 및 슬리피지 처리
- 5개 액션 공간 (Strong Buy, Buy, Hold, Sell, Strong Sell)
- 리스크 관리 및 포지션 제한

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import math
import gymnasium as gym
from gymnasium import spaces


class ActionType(Enum):
    """거래 액션 타입"""
    STRONG_BUY = 0
    BUY = 1
    HOLD = 2
    SELL = 3
    STRONG_SELL = 4


@dataclass
class PortfolioState:
    """포트폴리오 상태"""
    cash: float
    shares: float
    total_value: float
    position_value: float
    cash_ratio: float
    shares_ratio: float


@dataclass
class TradeResult:
    """거래 결과"""
    action: int
    shares_traded: float
    cash_change: float
    fees_paid: float
    slippage_cost: float
    success: bool
    message: str


class TradingEnvironment(gym.Env):
    """
    Gymnasium 기반 거래 환경
    
    포트폴리오 관리, 거래 실행, 수수료 및 슬리피지 처리
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, config: Dict):
        """
        거래 환경 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 거래 설정
        self.trading_config = config.get('trading', {})
        
        # 초기 자본
        self.initial_capital = self.trading_config.get('initial_capital', 10000.0)
        
        # 수수료 설정
        self.fees_config = self.trading_config.get('fees', {})
        self.buy_fee_rate = self.fees_config.get('buy', 0.0005)  # 0.05%
        self.sell_fee_rate = self.fees_config.get('sell', 0.0005)  # 0.05%
        
        # 슬리피지 설정
        self.slippage_config = self.trading_config.get('slippage', {})
        self.buy_slippage_rate = self.slippage_config.get('buy', 0.0002)  # 0.02%
        self.sell_slippage_rate = self.slippage_config.get('sell', 0.0002)  # 0.02%
        
        # 액션 설정
        self.actions_config = self.trading_config.get('actions', {})
        self.position_sizes = self.actions_config.get('position_sizes', {
            0: 0.4,  # Strong Buy: 40%
            1: 0.2,  # Buy: 20%
            2: 0.0,  # Hold: 0%
            3: 0.2,  # Sell: 20%
            4: 0.4   # Strong Sell: 40%
        })
        
        # 리스크 관리
        self.risk_config = self.trading_config.get('risk_management', {})
        self.max_position_ratio = self.risk_config.get('max_position_ratio', 0.95)
        self.min_cash_ratio = self.risk_config.get('min_cash_ratio', 0.05)
        
        # 포트폴리오 상태 초기화
        self.portfolio_state = PortfolioState(
            cash=self.initial_capital,
            shares=0.0,
            total_value=self.initial_capital,
            position_value=0.0,
            cash_ratio=1.0,
            shares_ratio=0.0
        )
        
        # 거래 히스토리
        self.trade_history = []
        self.portfolio_history = []
        
        # 현재 가격
        self.current_price = 0.0
        
        # Gymnasium 환경 설정
        self.action_space = spaces.Discrete(5)  # 5개 액션 (Strong Buy, Buy, Hold, Sell, Strong Sell)
        
        # 관찰 공간 설정 (포트폴리오 상태 + 시장 데이터)
        # 포트폴리오 상태: [현금비율, 주식비율, 총자산, 수익률, 변동성]
        # 시장 데이터: [가격, 거래량, 기술적 지표 등]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # 환경 상태 초기화
        self.reset()
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Gymnasium 기반 거래 환경 초기화 완료")
        self.logger.info(f"초기 자본: ${self.initial_capital:,.2f}")
        self.logger.info(f"수수료: 매수 {self.buy_fee_rate*100:.2f}%, 매도 {self.sell_fee_rate*100:.2f}%")
        self.logger.info(f"액션 공간: {self.action_space}")
        self.logger.info(f"관찰 공간: {self.observation_space}")
        self.logger.info(f"슬리피지: 매수 {self.buy_slippage_rate*100:.2f}%, 매도 {self.sell_slippage_rate*100:.2f}%")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        환경 초기화 (Gymnasium 인터페이스)
        
        Args:
            seed: 랜덤 시드
            options: 추가 옵션
            
        Returns:
            초기 관찰값과 정보 딕셔너리
        """
        super().reset(seed=seed)
        
        # 포트폴리오 초기화
        self.portfolio = PortfolioState(
            cash=self.initial_capital,
            shares=0.0,
            total_value=self.initial_capital,
            position_value=0.0,
            cash_ratio=1.0,
            shares_ratio=0.0
        )
        
        # 거래 히스토리 초기화
        self.trade_history = []
        self.portfolio_history = [self.portfolio]
        
        # 현재 가격 초기화
        self.current_price = 100.0  # 기본 가격
        
        # 관찰값 생성
        observation = self._get_observation()
        
        info = {
            'portfolio': self.portfolio,
            'current_price': self.current_price
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        환경 스텝 실행 (Gymnasium 인터페이스)
        
        Args:
            action: 실행할 액션 (0-4)
            
        Returns:
            관찰값, 보상, 종료 여부, 잘림 여부, 정보 딕셔너리
        """
        # 액션 실행
        trade_result = self.execute_action(action)
        
        # 포트폴리오 업데이트
        self._update_portfolio()
        
        # 관찰값 생성
        observation = self._get_observation()
        
        # 보상 계산
        reward = self._calculate_reward(trade_result)
        
        # 종료 조건 확인
        terminated = self._check_termination()
        truncated = False  # 시간 제한은 별도로 관리
        
        # 정보 딕셔너리
        info = {
            'portfolio': self.portfolio,
            'trade_result': trade_result,
            'current_price': self.current_price,
            'total_value': self.portfolio.total_value
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        환경 렌더링 (Gymnasium 인터페이스)
        
        Args:
            mode: 렌더링 모드 ('human' 또는 'rgb_array')
            
        Returns:
            렌더링 결과 (rgb_array 모드인 경우)
        """
        if mode == "human":
            print(f"=== 거래 환경 상태 ===")
            print(f"현재 가격: ${self.current_price:.2f}")
            print(f"현금: ${self.portfolio.cash:.2f}")
            print(f"주식: {self.portfolio.shares:.2f}주")
            print(f"총 자산: ${self.portfolio.total_value:.2f}")
            print(f"현금 비율: {self.portfolio.cash_ratio:.2%}")
            print(f"주식 비율: {self.portfolio.shares_ratio:.2%}")
            print(f"거래 횟수: {len(self.trade_history)}")
            print("========================")
        elif mode == "rgb_array":
            # RGB 배열 반환 (구현 필요시)
            return None
        
        return None
    
    def _get_observation(self) -> np.ndarray:
        """현재 관찰값 생성"""
        try:
            # 포트폴리오 상태 (5차원)
            portfolio_obs = np.array([
                self.portfolio.cash_ratio,
                self.portfolio.shares_ratio,
                self.portfolio.total_value / self.initial_capital,  # 정규화된 총 자산
                (self.portfolio.total_value - self.initial_capital) / self.initial_capital,  # 수익률
                0.0  # 변동성 (추후 계산)
            ], dtype=np.float32)
            
            # 시장 데이터 (5차원)
            market_obs = np.array([
                self.current_price / 100.0,  # 정규화된 가격
                1.0,  # 거래량 (추후 실제 데이터 사용)
                0.0,  # 기술적 지표 1
                0.0,  # 기술적 지표 2
                0.0   # 기술적 지표 3
            ], dtype=np.float32)
            
            # 관찰값 결합
            observation = np.concatenate([portfolio_obs, market_obs])
            
            return observation
            
        except Exception as e:
            self.logger.error(f"관찰값 생성 실패: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _calculate_reward(self, trade_result: TradeResult) -> float:
        """보상 계산"""
        try:
            # 기본 보상: 포트폴리오 수익률
            portfolio_return = (self.portfolio.total_value - self.initial_capital) / self.initial_capital
            
            # 거래 비용 페널티
            cost_penalty = -(trade_result.fees_paid + trade_result.slippage_cost) / self.initial_capital
            
            # 리스크 페널티 (변동성 기반)
            risk_penalty = 0.0  # 추후 구현
            
            # 총 보상
            reward = portfolio_return + cost_penalty + risk_penalty
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"보상 계산 실패: {e}")
            return 0.0
    
    def _check_termination(self) -> bool:
        """종료 조건 확인"""
        try:
            # 자산이 10% 이하로 떨어지면 종료
            if self.portfolio.total_value < self.initial_capital * 0.1:
                return True
            
            # 최대 거래 횟수 초과시 종료
            if len(self.trade_history) > 1000:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"종료 조건 확인 실패: {e}")
            return False
    
    def _update_portfolio(self):
        """포트폴리오 상태 업데이트"""
        try:
            # 주식 포지션 가치 계산
            self.portfolio.position_value = self.portfolio.shares * self.current_price
            
            # 총 자산 계산
            self.portfolio.total_value = self.portfolio.cash + self.portfolio.position_value
            
            # 비율 계산
            if self.portfolio.total_value > 0:
                self.portfolio.cash_ratio = self.portfolio.cash / self.portfolio.total_value
                self.portfolio.shares_ratio = self.portfolio.position_value / self.portfolio.total_value
            else:
                self.portfolio.cash_ratio = 0.0
                self.portfolio.shares_ratio = 0.0
            
            # 포트폴리오 히스토리 업데이트
            self.portfolio_history.append(self.portfolio)
            
        except Exception as e:
            self.logger.error(f"포트폴리오 업데이트 실패: {e}")
    
    def update_price(self, price: float):
        """
        현재 가격 업데이트
        
        Args:
            price: 새로운 가격
        """
        try:
            self.current_price = price
            self._update_portfolio()
            
            # 포트폴리오 가치 업데이트
            self._update_portfolio_value()
            
        except Exception as e:
            self.logger.error(f"가격 업데이트 실패: {e}")
    
    def _update_portfolio_value(self):
        """포트폴리오 가치 업데이트"""
        try:
            if self.current_price > 0:
                self.portfolio_state.position_value = self.portfolio_state.shares * self.current_price
                self.portfolio_state.total_value = self.portfolio_state.cash + self.portfolio_state.position_value
                
                # 비율 계산
                if self.portfolio_state.total_value > 0:
                    self.portfolio_state.cash_ratio = self.portfolio_state.cash / self.portfolio_state.total_value
                    self.portfolio_state.shares_ratio = self.portfolio_state.position_value / self.portfolio_state.total_value
                else:
                    self.portfolio_state.cash_ratio = 1.0
                    self.portfolio_state.shares_ratio = 0.0
            
        except Exception as e:
            self.logger.error(f"포트폴리오 가치 업데이트 실패: {e}")
    
    def execute_action(self, action: int, current_price: Optional[float] = None) -> TradeResult:
        """
        거래 액션 실행
        
        Args:
            action: 거래 액션 (0-4)
            current_price: 현재 가격 (선택사항)
            
        Returns:
            거래 결과
        """
        try:
            # 가격 업데이트
            if current_price is not None:
                self.update_price(current_price)
            
            if self.current_price <= 0:
                return TradeResult(
                    action=action,
                    shares_traded=0.0,
                    cash_change=0.0,
                    fees_paid=0.0,
                    slippage_cost=0.0,
                    success=False,
                    message="유효하지 않은 가격"
                )
            
            # 액션별 거래 실행
            if action == ActionType.STRONG_BUY.value:
                return self._execute_buy_action(0.4, "Strong Buy")
            elif action == ActionType.BUY.value:
                return self._execute_buy_action(0.2, "Buy")
            elif action == ActionType.HOLD.value:
                return self._execute_hold_action()
            elif action == ActionType.SELL.value:
                return self._execute_sell_action(0.2, "Sell")
            elif action == ActionType.STRONG_SELL.value:
                return self._execute_sell_action(0.4, "Strong Sell")
            else:
                return TradeResult(
                    action=action,
                    shares_traded=0.0,
                    cash_change=0.0,
                    fees_paid=0.0,
                    slippage_cost=0.0,
                    success=False,
                    message=f"유효하지 않은 액션: {action}"
                )
            
        except Exception as e:
            self.logger.error(f"거래 액션 실행 실패: {e}")
            return TradeResult(
                action=action,
                shares_traded=0.0,
                cash_change=0.0,
                fees_paid=0.0,
                slippage_cost=0.0,
                success=False,
                message=f"거래 실행 오류: {str(e)}"
            )
    
    def _execute_buy_action(self, target_ratio: float, action_name: str) -> TradeResult:
        """
        매수 액션 실행
        
        Args:
            target_ratio: 목표 포지션 비율
            action_name: 액션 이름
            
        Returns:
            거래 결과
        """
        try:
            # 현재 포지션 비율
            current_ratio = self.portfolio_state.shares_ratio
            
            # 매수할 비율 계산
            buy_ratio = target_ratio - current_ratio
            
            if buy_ratio <= 0:
                return TradeResult(
                    action=0 if action_name == "Strong Buy" else 1,
                    shares_traded=0.0,
                    cash_change=0.0,
                    fees_paid=0.0,
                    slippage_cost=0.0,
                    success=True,
                    message=f"{action_name}: 이미 충분한 포지션 보유"
                )
            
            # 리스크 관리: 최대 포지션 비율 확인
            if target_ratio > self.max_position_ratio:
                target_ratio = self.max_position_ratio
                buy_ratio = target_ratio - current_ratio
            
            # 매수할 금액 계산
            buy_amount = self.portfolio_state.total_value * buy_ratio
            
            # 최소 현금 비율 확인
            remaining_cash = self.portfolio_state.cash - buy_amount
            if remaining_cash < self.portfolio_state.total_value * self.min_cash_ratio:
                available_cash = self.portfolio_state.cash - (self.portfolio_state.total_value * self.min_cash_ratio)
                if available_cash > 0:
                    buy_amount = available_cash
                else:
                    return TradeResult(
                        action=0 if action_name == "Strong Buy" else 1,
                        shares_traded=0.0,
                        cash_change=0.0,
                        fees_paid=0.0,
                        slippage_cost=0.0,
                        success=False,
                        message=f"{action_name}: 현금 부족 (최소 현금 비율 보장 필요)"
                    )
            
            # 슬리피지 적용된 가격
            effective_price = self.current_price * (1 + self.buy_slippage_rate)
            
            # 매수할 주식 수
            shares_to_buy = buy_amount / effective_price
            
            # 수수료 계산
            fees = buy_amount * self.buy_fee_rate
            
            # 총 비용
            total_cost = buy_amount + fees
            
            # 현금 부족 확인
            if total_cost > self.portfolio_state.cash:
                # 가능한 최대 매수
                available_cash = self.portfolio_state.cash - fees
                if available_cash > 0:
                    shares_to_buy = available_cash / effective_price
                    buy_amount = shares_to_buy * effective_price
                    total_cost = buy_amount + fees
                else:
                    return TradeResult(
                        action=0 if action_name == "Strong Buy" else 1,
                        shares_traded=0.0,
                        cash_change=0.0,
                        fees_paid=0.0,
                        slippage_cost=0.0,
                        success=False,
                        message=f"{action_name}: 수수료를 포함한 현금 부족"
                    )
            
            # 포트폴리오 업데이트
            self.portfolio_state.cash -= total_cost
            self.portfolio_state.shares += shares_to_buy
            
            # 포트폴리오 가치 업데이트
            self._update_portfolio_value()
            
            # 거래 히스토리 기록
            trade_result = TradeResult(
                action=0 if action_name == "Strong Buy" else 1,
                shares_traded=shares_to_buy,
                cash_change=-total_cost,
                fees_paid=fees,
                slippage_cost=buy_amount * self.buy_slippage_rate,
                success=True,
                message=f"{action_name}: {shares_to_buy:.4f}주 매수, ${total_cost:.2f} 지출"
            )
            
            self.trade_history.append(trade_result)
            self.portfolio_history.append(self.portfolio_state)
            
            self.logger.debug(f"{action_name} 실행: {shares_to_buy:.4f}주, ${total_cost:.2f}")
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"매수 액션 실행 실패: {e}")
            return TradeResult(
                action=0 if action_name == "Strong Buy" else 1,
                shares_traded=0.0,
                cash_change=0.0,
                fees_paid=0.0,
                slippage_cost=0.0,
                success=False,
                message=f"{action_name} 실행 오류: {str(e)}"
            )
    
    def _execute_sell_action(self, target_ratio: float, action_name: str) -> TradeResult:
        """
        매도 액션 실행
        
        Args:
            target_ratio: 목표 포지션 비율
            action_name: 액션 이름
            
        Returns:
            거래 결과
        """
        try:
            # 현재 포지션 비율
            current_ratio = self.portfolio_state.shares_ratio
            
            # 매도할 비율 계산
            sell_ratio = current_ratio - target_ratio
            
            if sell_ratio <= 0:
                return TradeResult(
                    action=3 if action_name == "Sell" else 4,
                    shares_traded=0.0,
                    cash_change=0.0,
                    fees_paid=0.0,
                    slippage_cost=0.0,
                    success=True,
                    message=f"{action_name}: 매도할 주식 없음"
                )
            
            # 매도할 주식 수
            shares_to_sell = self.portfolio_state.shares * (sell_ratio / current_ratio)
            
            # 슬리피지 적용된 가격
            effective_price = self.current_price * (1 - self.sell_slippage_rate)
            
            # 매도 금액
            sell_amount = shares_to_sell * effective_price
            
            # 수수료 계산
            fees = sell_amount * self.sell_fee_rate
            
            # 순 수령액
            net_proceeds = sell_amount - fees
            
            # 포트폴리오 업데이트
            self.portfolio_state.cash += net_proceeds
            self.portfolio_state.shares -= shares_to_sell
            
            # 포트폴리오 가치 업데이트
            self._update_portfolio_value()
            
            # 거래 히스토리 기록
            trade_result = TradeResult(
                action=3 if action_name == "Sell" else 4,
                shares_traded=-shares_to_sell,
                cash_change=net_proceeds,
                fees_paid=fees,
                slippage_cost=sell_amount * self.sell_slippage_rate,
                success=True,
                message=f"{action_name}: {shares_to_sell:.4f}주 매도, ${net_proceeds:.2f} 수령"
            )
            
            self.trade_history.append(trade_result)
            self.portfolio_history.append(self.portfolio_state)
            
            self.logger.debug(f"{action_name} 실행: {shares_to_sell:.4f}주, ${net_proceeds:.2f}")
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"매도 액션 실행 실패: {e}")
            return TradeResult(
                action=3 if action_name == "Sell" else 4,
                shares_traded=0.0,
                cash_change=0.0,
                fees_paid=0.0,
                slippage_cost=0.0,
                success=False,
                message=f"{action_name} 실행 오류: {str(e)}"
            )
    
    def _execute_hold_action(self) -> TradeResult:
        """
        홀드 액션 실행
        
        Returns:
            거래 결과
        """
        try:
            # 포트폴리오 가치 업데이트만 수행
            self._update_portfolio_value()
            
            trade_result = TradeResult(
                action=2,
                shares_traded=0.0,
                cash_change=0.0,
                fees_paid=0.0,
                slippage_cost=0.0,
                success=True,
                message="Hold: 포지션 유지"
            )
            
            self.trade_history.append(trade_result)
            self.portfolio_history.append(self.portfolio_state)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"홀드 액션 실행 실패: {e}")
            return TradeResult(
                action=2,
                shares_traded=0.0,
                cash_change=0.0,
                fees_paid=0.0,
                slippage_cost=0.0,
                success=False,
                message=f"Hold 실행 오류: {str(e)}"
            )
    
    def get_portfolio_state(self) -> PortfolioState:
        """현재 포트폴리오 상태 반환"""
        return self.portfolio_state
    
    def get_portfolio_return(self) -> float:
        """포트폴리오 수익률 계산"""
        try:
            if self.initial_capital > 0:
                return (self.portfolio_state.total_value - self.initial_capital) / self.initial_capital
            return 0.0
            
        except Exception as e:
            self.logger.error(f"포트폴리오 수익률 계산 실패: {e}")
            return 0.0
    
    def get_trade_summary(self) -> Dict[str, Union[int, float, List]]:
        """거래 요약 정보 반환"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'total_fees': 0.0,
                    'total_slippage': 0.0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'hold_trades': 0
                }
            
            total_trades = len(self.trade_history)
            successful_trades = sum(1 for trade in self.trade_history if trade.success)
            total_fees = sum(trade.fees_paid for trade in self.trade_history)
            total_slippage = sum(trade.slippage_cost for trade in self.trade_history)
            
            buy_trades = sum(1 for trade in self.trade_history if trade.action in [0, 1])
            sell_trades = sum(1 for trade in self.trade_history if trade.action in [3, 4])
            hold_trades = sum(1 for trade in self.trade_history if trade.action == 2)
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'total_fees': total_fees,
                'total_slippage': total_slippage,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'hold_trades': hold_trades,
                'success_rate': successful_trades / total_trades if total_trades > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"거래 요약 정보 생성 실패: {e}")
            return {}
    
    def reset_portfolio(self):
        """포트폴리오 초기화"""
        try:
            self.portfolio_state = PortfolioState(
                cash=self.initial_capital,
                shares=0.0,
                total_value=self.initial_capital,
                position_value=0.0,
                cash_ratio=1.0,
                shares_ratio=0.0
            )
            
            self.trade_history.clear()
            self.portfolio_history.clear()
            self.current_price = 0.0
            
            self.logger.info("포트폴리오 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"포트폴리오 초기화 실패: {e}")
    
    def get_environment_info(self) -> Dict[str, Union[float, int, Dict]]:
        """거래 환경 정보 반환"""
        try:
            return {
                'initial_capital': self.initial_capital,
                'current_portfolio_value': self.portfolio_state.total_value,
                'portfolio_return': self.get_portfolio_return(),
                'current_price': self.current_price,
                'fees_config': {
                    'buy_fee_rate': self.buy_fee_rate,
                    'sell_fee_rate': self.sell_fee_rate
                },
                'slippage_config': {
                    'buy_slippage_rate': self.buy_slippage_rate,
                    'sell_slippage_rate': self.sell_slippage_rate
                },
                'position_sizes': self.position_sizes,
                'risk_management': {
                    'max_position_ratio': self.max_position_ratio,
                    'min_cash_ratio': self.min_cash_ratio
                },
                'trade_summary': self.get_trade_summary()
            }
            
        except Exception as e:
            self.logger.error(f"거래 환경 정보 생성 실패: {e}")
            return {}


# 편의 함수들
def create_trading_environment(config: Dict) -> TradingEnvironment:
    """거래 환경 생성 편의 함수"""
    return TradingEnvironment(config)


def get_action_name(action: int) -> str:
    """액션 번호를 이름으로 변환"""
    action_names = {
        0: "Strong Buy",
        1: "Buy", 
        2: "Hold",
        3: "Sell",
        4: "Strong Sell"
    }
    return action_names.get(action, "Unknown")
