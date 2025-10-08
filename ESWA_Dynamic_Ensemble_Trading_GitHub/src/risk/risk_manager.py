"""
고급 리스크 관리 모듈
Advanced Risk Management Module

동적 포지션 크기 조절 및 체제별 리스크 관리
- 동적 포지션 크기 조절 시스템
- 체제별 차별화된 리스크 관리
- 손절매 및 익절 전략
- 포트폴리오 다각화 전략
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import math


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """리스크 지표"""
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation: float


@dataclass
class PositionSizing:
    """포지션 크기 정보"""
    base_size: float
    adjusted_size: float
    risk_adjusted_size: float
    max_size: float
    min_size: float


@dataclass
class RiskLimits:
    """리스크 한계"""
    max_position_size: float
    max_total_exposure: float
    max_drawdown_limit: float
    max_var_limit: float
    max_correlation: float


class AdvancedRiskManager:
    """
    고급 리스크 관리 메인 클래스
    
    동적 포지션 크기 조절 및 체제별 리스크 관리
    """
    
    def __init__(self, config: Dict):
        """
        고급 리스크 관리자 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 리스크 관리 설정
        self.risk_config = config.get('risk_management', {})
        
        # 기본 리스크 한계
        self.risk_limits = RiskLimits(
            max_position_size=self.risk_config.get('max_position_size', 0.2),
            max_total_exposure=self.risk_config.get('max_total_exposure', 0.5),
            max_drawdown_limit=self.risk_config.get('max_drawdown_limit', 0.15),
            max_var_limit=self.risk_config.get('max_var_limit', 0.05),
            max_correlation=self.risk_config.get('max_correlation', 0.7)
        )
        
        # 체제별 리스크 설정
        self.regime_risk_config = self.risk_config.get('regime_specific', {})
        
        # 포지션 크기 조절 설정
        self.position_sizing_config = self.risk_config.get('position_sizing', {})
        self.base_position_size = self.position_sizing_config.get('base_size', 0.1)
        self.volatility_target = self.position_sizing_config.get('volatility_target', 0.15)
        
        # 손절매/익절 설정
        self.stop_loss_config = self.risk_config.get('stop_loss', {})
        self.take_profit_config = self.risk_config.get('take_profit', {})
        
        # 리스크 지표 계산 설정
        self.risk_metrics_config = self.risk_config.get('risk_metrics', {})
        self.confidence_level = self.risk_metrics_config.get('confidence_level', 0.95)
        self.lookback_period = self.risk_metrics_config.get('lookback_period', 252)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("고급 리스크 관리자 초기화 완료")
        self.logger.info(f"최대 포지션 크기: {self.risk_limits.max_position_size:.1%}")
        self.logger.info(f"최대 총 노출: {self.risk_limits.max_total_exposure:.1%}")
        self.logger.info(f"최대 낙폭 한계: {self.risk_limits.max_drawdown_limit:.1%}")
        self.logger.info(f"기본 포지션 크기: {self.base_position_size:.1%}")
        self.logger.info(f"변동성 목표: {self.volatility_target:.1%}")
    
    def manage_portfolio_risk(self, portfolio_state: Dict, 
                            market_data: Dict, 
                            regime_info: Dict) -> Dict[str, Any]:
        """
        포트폴리오 리스크 관리
        
        Args:
            portfolio_state: 포트폴리오 상태
            market_data: 시장 데이터
            regime_info: 체제 정보
            
        Returns:
            리스크 관리 결과
        """
        try:
            self.logger.info("포트폴리오 리스크 관리 시작")
            
            # 1. 리스크 지표 계산
            risk_metrics = self._calculate_risk_metrics(portfolio_state, market_data)
            
            # 2. 리스크 레벨 평가
            risk_level = self._assess_risk_level(risk_metrics, portfolio_state)
            
            # 3. 체제별 리스크 조정
            regime_adjustment = self._get_regime_risk_adjustment(regime_info)
            
            # 4. 포지션 크기 조절
            position_sizing = self._calculate_position_sizing(
                risk_metrics, risk_level, regime_adjustment
            )
            
            # 5. 손절매/익절 신호 생성
            stop_loss_signals = self._generate_stop_loss_signals(
                portfolio_state, risk_metrics, regime_info
            )
            
            # 6. 포트폴리오 다각화 권장사항
            diversification_recommendations = self._generate_diversification_recommendations(
                portfolio_state, market_data, risk_metrics
            )
            
            # 7. 리스크 관리 결과 정리
            risk_management_result = {
                'risk_level': risk_level,
                'risk_metrics': risk_metrics,
                'position_sizing': position_sizing,
                'stop_loss_signals': stop_loss_signals,
                'diversification_recommendations': diversification_recommendations,
                'regime_adjustment': regime_adjustment,
                'risk_limits': self.risk_limits,
                'recommendations': self._generate_risk_recommendations(
                    risk_level, risk_metrics, position_sizing
                )
            }
            
            self.logger.info(f"포트폴리오 리스크 관리 완료 - 리스크 레벨: {risk_level.value}")
            
            return risk_management_result
            
        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 관리 실패: {e}")
            return self._get_default_risk_result()
    
    def _calculate_risk_metrics(self, portfolio_state: Dict, 
                              market_data: Dict) -> RiskMetrics:
        """리스크 지표 계산"""
        try:
            # 포트폴리오 수익률 계산
            returns = self._get_portfolio_returns(portfolio_state, market_data)
            
            if len(returns) < 2:
                return self._get_default_risk_metrics()
            
            # VaR 및 CVaR 계산
            var_95 = self._calculate_var(returns, self.confidence_level)
            cvar_95 = self._calculate_cvar(returns, self.confidence_level)
            
            # 최대 낙폭 계산
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # 변동성 계산
            volatility = np.std(returns) * np.sqrt(252)
            
            # 샤프 비율 계산
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # 베타 및 상관관계 계산 (간단한 구현)
            beta = 1.0  # 추후 구현
            correlation = 0.5  # 추후 구현
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation=correlation
            )
            
        except Exception as e:
            self.logger.error(f"리스크 지표 계산 실패: {e}")
            return self._get_default_risk_metrics()
    
    def _assess_risk_level(self, risk_metrics: RiskMetrics, 
                          portfolio_state: Dict) -> RiskLevel:
        """리스크 레벨 평가"""
        try:
            # 리스크 점수 계산
            risk_score = 0.0
            
            # VaR 기반 리스크 점수
            if risk_metrics.var_95 > self.risk_limits.max_var_limit:
                risk_score += 2.0
            elif risk_metrics.var_95 > self.risk_limits.max_var_limit * 0.8:
                risk_score += 1.0
            
            # 최대 낙폭 기반 리스크 점수
            if risk_metrics.max_drawdown > self.risk_limits.max_drawdown_limit:
                risk_score += 2.0
            elif risk_metrics.max_drawdown > self.risk_limits.max_drawdown_limit * 0.8:
                risk_score += 1.0
            
            # 변동성 기반 리스크 점수
            if risk_metrics.volatility > 0.3:
                risk_score += 1.0
            elif risk_metrics.volatility > 0.2:
                risk_score += 0.5
            
            # 샤프 비율 기반 리스크 점수
            if risk_metrics.sharpe_ratio < 0:
                risk_score += 1.0
            elif risk_metrics.sharpe_ratio < 0.5:
                risk_score += 0.5
            
            # 리스크 레벨 결정
            if risk_score >= 4.0:
                return RiskLevel.CRITICAL
            elif risk_score >= 2.5:
                return RiskLevel.HIGH
            elif risk_score >= 1.0:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"리스크 레벨 평가 실패: {e}")
            return RiskLevel.MEDIUM
    
    def _get_regime_risk_adjustment(self, regime_info: Dict) -> Dict[str, float]:
        """체제별 리스크 조정"""
        try:
            regime = regime_info.get('regime', 'bull_market')
            confidence = regime_info.get('confidence', 0.5)
            
            # 체제별 기본 조정 계수
            regime_adjustments = {
                'bull_market': {'position_multiplier': 1.2, 'risk_multiplier': 0.8},
                'bear_market': {'position_multiplier': 0.6, 'risk_multiplier': 1.5},
                'sideways_market': {'position_multiplier': 0.8, 'risk_multiplier': 1.2}
            }
            
            base_adjustment = regime_adjustments.get(regime, regime_adjustments['bull_market'])
            
            # 신뢰도에 따른 조정
            confidence_factor = 0.5 + confidence * 0.5  # 0.5 ~ 1.0
            
            return {
                'position_multiplier': base_adjustment['position_multiplier'] * confidence_factor,
                'risk_multiplier': base_adjustment['risk_multiplier'] * confidence_factor,
                'regime': regime,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"체제별 리스크 조정 실패: {e}")
            return {'position_multiplier': 1.0, 'risk_multiplier': 1.0, 'regime': 'unknown', 'confidence': 0.5}
    
    def _calculate_position_sizing(self, risk_metrics: RiskMetrics, 
                                 risk_level: RiskLevel, 
                                 regime_adjustment: Dict) -> PositionSizing:
        """포지션 크기 계산"""
        try:
            # 기본 포지션 크기
            base_size = self.base_position_size
            
            # 변동성 기반 조정
            volatility_adjustment = self.volatility_target / max(risk_metrics.volatility, 0.01)
            volatility_adjusted_size = base_size * volatility_adjustment
            
            # 리스크 레벨 기반 조정
            risk_adjustment = {
                RiskLevel.LOW: 1.2,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 0.7,
                RiskLevel.CRITICAL: 0.4
            }
            risk_adjusted_size = volatility_adjusted_size * risk_adjustment[risk_level]
            
            # 체제별 조정
            regime_adjusted_size = risk_adjusted_size * regime_adjustment['position_multiplier']
            
            # 최종 포지션 크기 (한계 내에서)
            final_size = min(
                max(regime_adjusted_size, 0.01),  # 최소 1%
                self.risk_limits.max_position_size  # 최대 한계
            )
            
            return PositionSizing(
                base_size=base_size,
                adjusted_size=volatility_adjusted_size,
                risk_adjusted_size=risk_adjusted_size,
                max_size=self.risk_limits.max_position_size,
                min_size=0.01
            )
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {e}")
            return PositionSizing(
                base_size=self.base_position_size,
                adjusted_size=self.base_position_size,
                risk_adjusted_size=self.base_position_size,
                max_size=self.risk_limits.max_position_size,
                min_size=0.01
            )
    
    def _generate_stop_loss_signals(self, portfolio_state: Dict, 
                                  risk_metrics: RiskMetrics, 
                                  regime_info: Dict) -> Dict[str, Any]:
        """손절매/익절 신호 생성"""
        try:
            # 손절매 신호
            stop_loss_signals = {
                'triggered': False,
                'reason': None,
                'recommended_action': 'hold'
            }
            
            # 최대 낙폭 기반 손절매
            if risk_metrics.max_drawdown > self.risk_limits.max_drawdown_limit:
                stop_loss_signals['triggered'] = True
                stop_loss_signals['reason'] = 'max_drawdown_exceeded'
                stop_loss_signals['recommended_action'] = 'reduce_position'
            
            # VaR 기반 손절매
            if risk_metrics.var_95 > self.risk_limits.max_var_limit:
                stop_loss_signals['triggered'] = True
                stop_loss_signals['reason'] = 'var_exceeded'
                stop_loss_signals['recommended_action'] = 'reduce_position'
            
            # 체제별 손절매 조정
            regime = regime_info.get('regime', 'bull_market')
            if regime == 'bear_market' and risk_metrics.max_drawdown > 0.1:
                stop_loss_signals['triggered'] = True
                stop_loss_signals['reason'] = 'bear_market_drawdown'
                stop_loss_signals['recommended_action'] = 'reduce_position'
            
            return stop_loss_signals
            
        except Exception as e:
            self.logger.error(f"손절매 신호 생성 실패: {e}")
            return {'triggered': False, 'reason': None, 'recommended_action': 'hold'}
    
    def _generate_diversification_recommendations(self, portfolio_state: Dict, 
                                                market_data: Dict, 
                                                risk_metrics: RiskMetrics) -> List[str]:
        """포트폴리오 다각화 권장사항 생성"""
        try:
            recommendations = []
            
            # 상관관계 기반 권장사항
            if risk_metrics.correlation > self.risk_limits.max_correlation:
                recommendations.append("포트폴리오 상관관계가 높습니다. 자산 다각화를 고려하세요.")
            
            # 변동성 기반 권장사항
            if risk_metrics.volatility > 0.25:
                recommendations.append("포트폴리오 변동성이 높습니다. 안정 자산 비중을 늘리세요.")
            
            # 샤프 비율 기반 권장사항
            if risk_metrics.sharpe_ratio < 0.5:
                recommendations.append("리스크 대비 수익률이 낮습니다. 투자 전략을 재검토하세요.")
            
            # VaR 기반 권장사항
            if risk_metrics.var_95 > 0.03:
                recommendations.append("VaR이 높습니다. 포지션 크기를 줄이세요.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"다각화 권장사항 생성 실패: {e}")
            return ["리스크 관리 시스템 오류로 인한 권장사항을 생성할 수 없습니다."]
    
    def _generate_risk_recommendations(self, risk_level: RiskLevel, 
                                     risk_metrics: RiskMetrics, 
                                     position_sizing: PositionSizing) -> List[str]:
        """리스크 관리 권장사항 생성"""
        try:
            recommendations = []
            
            # 리스크 레벨 기반 권장사항
            if risk_level == RiskLevel.CRITICAL:
                recommendations.append("위험 수준이 매우 높습니다. 즉시 포지션을 줄이세요.")
            elif risk_level == RiskLevel.HIGH:
                recommendations.append("위험 수준이 높습니다. 포지션 크기를 줄이고 리스크를 모니터링하세요.")
            elif risk_level == RiskLevel.MEDIUM:
                recommendations.append("위험 수준이 보통입니다. 현재 포지션을 유지하되 주의 깊게 모니터링하세요.")
            else:
                recommendations.append("위험 수준이 낮습니다. 현재 전략을 유지하세요.")
            
            # 포지션 크기 기반 권장사항
            if position_sizing.risk_adjusted_size < position_sizing.base_size * 0.5:
                recommendations.append("포지션 크기가 크게 줄어들었습니다. 시장 상황을 재평가하세요.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"리스크 권장사항 생성 실패: {e}")
            return ["리스크 관리 시스템 오류로 인한 권장사항을 생성할 수 없습니다."]
    
    def _get_portfolio_returns(self, portfolio_state: Dict, market_data: Dict) -> List[float]:
        """포트폴리오 수익률 추출"""
        try:
            # 임시 구현 - 실제로는 포트폴리오 히스토리에서 추출
            return [0.001, 0.002, -0.001, 0.003, -0.002, 0.001, 0.002, -0.001]
        except Exception as e:
            self.logger.error(f"포트폴리오 수익률 추출 실패: {e}")
            return []
    
    def _calculate_var(self, returns: List[float], confidence_level: float) -> float:
        """Value at Risk 계산"""
        try:
            if not returns:
                return 0.0
            return float(np.percentile(returns, (1 - confidence_level) * 100))
        except Exception as e:
            self.logger.error(f"VaR 계산 실패: {e}")
            return 0.0
    
    def _calculate_cvar(self, returns: List[float], confidence_level: float) -> float:
        """Conditional Value at Risk 계산"""
        try:
            if not returns:
                return 0.0
            var = self._calculate_var(returns, confidence_level)
            return float(np.mean([r for r in returns if r <= var]))
        except Exception as e:
            self.logger.error(f"CVaR 계산 실패: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """최대 낙폭 계산"""
        try:
            if not returns:
                return 0.0
            cumulative_returns = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            return float(np.min(drawdown))
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """기본 리스크 지표 반환"""
        return RiskMetrics(
            var_95=0.02,
            cvar_95=0.03,
            max_drawdown=-0.05,
            volatility=0.15,
            sharpe_ratio=0.5,
            beta=1.0,
            correlation=0.3
        )
    
    def _get_default_risk_result(self) -> Dict[str, Any]:
        """기본 리스크 관리 결과 반환"""
        return {
            'risk_level': RiskLevel.MEDIUM,
            'risk_metrics': self._get_default_risk_metrics(),
            'position_sizing': PositionSizing(
                base_size=self.base_position_size,
                adjusted_size=self.base_position_size,
                risk_adjusted_size=self.base_position_size,
                max_size=self.risk_limits.max_position_size,
                min_size=0.01
            ),
            'stop_loss_signals': {'triggered': False, 'reason': None, 'recommended_action': 'hold'},
            'diversification_recommendations': [],
            'regime_adjustment': {'position_multiplier': 1.0, 'risk_multiplier': 1.0},
            'risk_limits': self.risk_limits,
            'recommendations': ["기본 리스크 관리 설정을 사용합니다."]
        }


# 편의 함수들
def create_risk_manager(config: Dict) -> AdvancedRiskManager:
    """고급 리스크 관리자 생성 편의 함수"""
    return AdvancedRiskManager(config)


def manage_portfolio_risk(portfolio_state: Dict, market_data: Dict, 
                         regime_info: Dict, config: Dict) -> Dict[str, Any]:
    """포트폴리오 리스크 관리 편의 함수"""
    risk_manager = create_risk_manager(config)
    return risk_manager.manage_portfolio_risk(portfolio_state, market_data, regime_info)
