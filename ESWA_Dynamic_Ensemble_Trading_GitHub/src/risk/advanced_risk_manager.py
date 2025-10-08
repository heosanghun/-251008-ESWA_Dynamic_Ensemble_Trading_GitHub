"""
고급 리스크 관리 시스템
Advanced Risk Management System

논문 목표 100% 달성을 위한 고급 리스크 관리 기능
- 동적 변동성 조절
- 포트폴리오 보호 시스템
- 체제별 적응형 리스크 관리
- 실시간 리스크 모니터링
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque
import warnings

class RiskLevel(Enum):
    """리스크 레벨"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """고급 리스크 지표"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float

@dataclass
class PositionSizing:
    """고급 포지션 크기 정보"""
    base_size: float
    volatility_adjusted_size: float
    risk_adjusted_size: float
    regime_adjusted_size: float
    final_size: float
    max_size: float
    min_size: float
    confidence_factor: float

class AdvancedRiskManager:
    """
    고급 리스크 관리 시스템
    
    논문 목표 100% 달성을 위한 고급 리스크 관리 기능
    """
    
    def __init__(self, config: Dict):
        """
        고급 리스크 관리자 초기화
        
        Args:
            config: 최적화된 설정 딕셔너리
        """
        self.config = config
        self.risk_config = config.get('risk_management', {})
        
        # 고급 리스크 한계
        self.risk_limits = {
            'max_position_size': self.risk_config.get('max_position_size', 0.12),
            'max_total_exposure': self.risk_config.get('max_total_exposure', 0.35),
            'max_drawdown_limit': self.risk_config.get('max_drawdown_limit', 0.10),
            'max_var_limit': self.risk_config.get('max_var_limit', 0.03),
            'max_correlation': self.risk_config.get('max_correlation', 0.6),
            'max_volatility': self.risk_config.get('position_sizing', {}).get('max_volatility', 0.20),
            'min_volatility': self.risk_config.get('position_sizing', {}).get('min_volatility', 0.05)
        }
        
        # 체제별 리스크 설정
        self.regime_risk_config = self.risk_config.get('regime_specific', {})
        
        # 포지션 크기 조절 설정
        self.position_sizing_config = self.risk_config.get('position_sizing', {})
        self.base_position_size = self.position_sizing_config.get('base_size', 0.06)
        self.volatility_target = self.position_sizing_config.get('volatility_target', 0.12)
        
        # 손절매/익절 설정
        self.stop_loss_config = self.risk_config.get('stop_loss', {})
        self.take_profit_config = self.risk_config.get('take_profit', {})
        
        # 성과 분석 설정
        self.performance_config = config.get('performance_analysis', {})
        self.risk_free_rate = self.performance_config.get('risk_free_rate', 0.03)
        self.confidence_level = self.performance_config.get('confidence_level', 0.99)
        
        # 리스크 히스토리
        self.risk_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=252)
        self.drawdown_history = deque(maxlen=252)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("고급 리스크 관리자 초기화 완료")
        self.logger.info(f"최대 포지션 크기: {self.risk_limits['max_position_size']:.1%}")
        self.logger.info(f"최대 총 노출: {self.risk_limits['max_total_exposure']:.1%}")
        self.logger.info(f"최대 낙폭 한계: {self.risk_limits['max_drawdown_limit']:.1%}")
        self.logger.info(f"변동성 목표: {self.volatility_target:.1%}")
        self.logger.info(f"기본 포지션 크기: {self.base_position_size:.1%}")
    
    def manage_portfolio_risk(self, portfolio_state: Dict, 
                            market_data: Dict, 
                            regime_info: Dict) -> Dict[str, Any]:
        """
        고급 포트폴리오 리스크 관리
        
        Args:
            portfolio_state: 포트폴리오 상태
            market_data: 시장 데이터
            regime_info: 체제 정보
            
        Returns:
            고급 리스크 관리 결과
        """
        try:
            self.logger.info("고급 포트폴리오 리스크 관리 시작")
            
            # 1. 고급 리스크 지표 계산
            risk_metrics = self._calculate_advanced_risk_metrics(portfolio_state, market_data)
            
            # 2. 리스크 레벨 평가
            risk_level = self._assess_advanced_risk_level(risk_metrics, portfolio_state)
            
            # 3. 체제별 적응형 리스크 조정
            regime_adjustment = self._get_adaptive_regime_risk_adjustment(regime_info, risk_metrics)
            
            # 4. 고급 포지션 크기 조절
            position_sizing = self._calculate_advanced_position_sizing(
                risk_metrics, risk_level, regime_adjustment, market_data
            )
            
            # 5. 포트폴리오 보호 시스템
            protection_signals = self._generate_protection_signals(
                portfolio_state, risk_metrics, regime_info
            )
            
            # 6. 동적 변동성 조절
            volatility_adjustment = self._calculate_volatility_adjustment(
                risk_metrics, market_data
            )
            
            # 7. 실시간 리스크 모니터링
            monitoring_alerts = self._generate_monitoring_alerts(
                risk_metrics, portfolio_state, regime_info
            )
            
            # 8. 리스크 관리 결과 정리
            risk_management_result = {
                'risk_level': risk_level,
                'risk_metrics': risk_metrics,
                'position_sizing': position_sizing,
                'protection_signals': protection_signals,
                'volatility_adjustment': volatility_adjustment,
                'monitoring_alerts': monitoring_alerts,
                'regime_adjustment': regime_adjustment,
                'risk_limits': self.risk_limits,
                'recommendations': self._generate_advanced_recommendations(
                    risk_level, risk_metrics, position_sizing, protection_signals
                )
            }
            
            # 리스크 히스토리 업데이트
            self._update_risk_history(risk_management_result)
            
            self.logger.info(f"고급 포트폴리오 리스크 관리 완료 - 리스크 레벨: {risk_level.value}")
            
            return risk_management_result
            
        except Exception as e:
            self.logger.error(f"고급 포트폴리오 리스크 관리 실패: {e}")
            return self._get_default_advanced_risk_result()
    
    def _calculate_advanced_risk_metrics(self, portfolio_state: Dict, 
                                       market_data: Dict) -> RiskMetrics:
        """고급 리스크 지표 계산"""
        try:
            # 포트폴리오 수익률 계산
            returns = self._get_portfolio_returns(portfolio_state, market_data)
            
            if len(returns) < 2:
                return self._get_default_risk_metrics()
            
            # VaR 및 CVaR 계산 (95%, 99%)
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            cvar_99 = self._calculate_cvar(returns, 0.99)
            
            # 최대 낙폭 계산
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # 변동성 계산
            volatility = np.std(returns) * np.sqrt(252)
            
            # 샤프 비율 계산
            sharpe_ratio = (np.mean(returns) - self.risk_free_rate/252) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # 소르티노 비율 계산
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (np.mean(returns) - self.risk_free_rate/252) / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
            
            # 칼마 비율 계산
            calmar_ratio = np.mean(returns) * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 베타 및 상관관계 계산
            beta = self._calculate_beta(returns, market_data)
            correlation = self._calculate_correlation(returns, market_data)
            
            # 추적 오차 및 정보 비율 계산
            tracking_error = self._calculate_tracking_error(returns, market_data)
            information_ratio = self._calculate_information_ratio(returns, market_data)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=beta,
                correlation=correlation,
                tracking_error=tracking_error,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            self.logger.error(f"고급 리스크 지표 계산 실패: {e}")
            return self._get_default_risk_metrics()
    
    def _assess_advanced_risk_level(self, risk_metrics: RiskMetrics, 
                                  portfolio_state: Dict) -> RiskLevel:
        """고급 리스크 레벨 평가"""
        try:
            # 리스크 점수 계산
            risk_score = 0.0
            
            # VaR 기반 리스크 점수
            if risk_metrics.var_95 > self.risk_limits['max_var_limit']:
                risk_score += 3.0
            elif risk_metrics.var_95 > self.risk_limits['max_var_limit'] * 0.8:
                risk_score += 2.0
            elif risk_metrics.var_95 > self.risk_limits['max_var_limit'] * 0.6:
                risk_score += 1.0
            
            # 최대 낙폭 기반 리스크 점수
            if risk_metrics.max_drawdown > self.risk_limits['max_drawdown_limit']:
                risk_score += 3.0
            elif risk_metrics.max_drawdown > self.risk_limits['max_drawdown_limit'] * 0.8:
                risk_score += 2.0
            elif risk_metrics.max_drawdown > self.risk_limits['max_drawdown_limit'] * 0.6:
                risk_score += 1.0
            
            # 변동성 기반 리스크 점수
            if risk_metrics.volatility > self.risk_limits['max_volatility']:
                risk_score += 2.0
            elif risk_metrics.volatility > self.risk_limits['max_volatility'] * 0.8:
                risk_score += 1.0
            
            # 샤프 비율 기반 리스크 점수
            if risk_metrics.sharpe_ratio < 0:
                risk_score += 2.0
            elif risk_metrics.sharpe_ratio < 0.5:
                risk_score += 1.0
            elif risk_metrics.sharpe_ratio < 1.0:
                risk_score += 0.5
            
            # 소르티노 비율 기반 리스크 점수
            if risk_metrics.sortino_ratio < 0:
                risk_score += 1.0
            elif risk_metrics.sortino_ratio < 0.5:
                risk_score += 0.5
            
            # 칼마 비율 기반 리스크 점수
            if risk_metrics.calmar_ratio < 0:
                risk_score += 1.0
            elif risk_metrics.calmar_ratio < 0.5:
                risk_score += 0.5
            
            # 리스크 레벨 결정
            if risk_score >= 8.0:
                return RiskLevel.CRITICAL
            elif risk_score >= 5.0:
                return RiskLevel.HIGH
            elif risk_score >= 3.0:
                return RiskLevel.MEDIUM
            elif risk_score >= 1.0:
                return RiskLevel.LOW
            else:
                return RiskLevel.VERY_LOW
                
        except Exception as e:
            self.logger.error(f"고급 리스크 레벨 평가 실패: {e}")
            return RiskLevel.MEDIUM
    
    def _get_adaptive_regime_risk_adjustment(self, regime_info: Dict, 
                                           risk_metrics: RiskMetrics) -> Dict[str, float]:
        """체제별 적응형 리스크 조정"""
        try:
            regime = regime_info.get('regime', 'bull_market')
            confidence = regime_info.get('confidence', 0.5)
            
            # 기본 체제별 조정 계수
            base_adjustments = {
                'bull_market': {'position_multiplier': 0.9, 'risk_multiplier': 1.2},
                'bear_market': {'position_multiplier': 0.3, 'risk_multiplier': 2.5},
                'sideways_market': {'position_multiplier': 0.5, 'risk_multiplier': 2.0}
            }
            
            base_adjustment = base_adjustments.get(regime, base_adjustments['bull_market'])
            
            # 신뢰도에 따른 조정
            confidence_factor = 0.5 + confidence * 0.5  # 0.5 ~ 1.0
            
            # 리스크 지표에 따른 적응형 조정
            risk_factor = 1.0
            if risk_metrics.volatility > self.volatility_target * 1.5:
                risk_factor *= 0.7  # 변동성이 높으면 포지션 크기 감소
            elif risk_metrics.volatility < self.volatility_target * 0.5:
                risk_factor *= 1.2  # 변동성이 낮으면 포지션 크기 증가
            
            if risk_metrics.max_drawdown > self.risk_limits['max_drawdown_limit'] * 0.8:
                risk_factor *= 0.8  # 낙폭이 크면 포지션 크기 감소
            
            return {
                'position_multiplier': base_adjustment['position_multiplier'] * confidence_factor * risk_factor,
                'risk_multiplier': base_adjustment['risk_multiplier'] * confidence_factor,
                'regime': regime,
                'confidence': confidence,
                'risk_factor': risk_factor
            }
            
        except Exception as e:
            self.logger.error(f"체제별 적응형 리스크 조정 실패: {e}")
            return {'position_multiplier': 1.0, 'risk_multiplier': 1.0, 'regime': 'unknown', 'confidence': 0.5, 'risk_factor': 1.0}
    
    def _calculate_advanced_position_sizing(self, risk_metrics: RiskMetrics, 
                                          risk_level: RiskLevel, 
                                          regime_adjustment: Dict,
                                          market_data: Dict) -> PositionSizing:
        """고급 포지션 크기 계산"""
        try:
            # 기본 포지션 크기
            base_size = self.base_position_size
            
            # 변동성 기반 조정
            volatility_adjustment = self.volatility_target / max(risk_metrics.volatility, 0.01)
            volatility_adjusted_size = base_size * volatility_adjustment
            
            # 리스크 레벨 기반 조정
            risk_adjustments = {
                RiskLevel.VERY_LOW: 1.3,
                RiskLevel.LOW: 1.1,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 0.7,
                RiskLevel.CRITICAL: 0.4
            }
            risk_adjusted_size = volatility_adjusted_size * risk_adjustments[risk_level]
            
            # 체제별 조정
            regime_adjusted_size = risk_adjusted_size * regime_adjustment['position_multiplier']
            
            # 신뢰도 기반 조정
            confidence_factor = regime_adjustment.get('confidence', 0.5)
            confidence_adjusted_size = regime_adjusted_size * (0.5 + confidence_factor * 0.5)
            
            # 최종 포지션 크기 (한계 내에서)
            final_size = min(
                max(confidence_adjusted_size, 0.01),  # 최소 1%
                self.risk_limits['max_position_size']  # 최대 한계
            )
            
            return PositionSizing(
                base_size=base_size,
                volatility_adjusted_size=volatility_adjusted_size,
                risk_adjusted_size=risk_adjusted_size,
                regime_adjusted_size=regime_adjusted_size,
                final_size=final_size,
                max_size=self.risk_limits['max_position_size'],
                min_size=0.01,
                confidence_factor=confidence_factor
            )
            
        except Exception as e:
            self.logger.error(f"고급 포지션 크기 계산 실패: {e}")
            return PositionSizing(
                base_size=self.base_position_size,
                volatility_adjusted_size=self.base_position_size,
                risk_adjusted_size=self.base_position_size,
                regime_adjusted_size=self.base_position_size,
                final_size=self.base_position_size,
                max_size=self.risk_limits['max_position_size'],
                min_size=0.01,
                confidence_factor=0.5
            )
    
    def _generate_protection_signals(self, portfolio_state: Dict, 
                                   risk_metrics: RiskMetrics, 
                                   regime_info: Dict) -> Dict[str, Any]:
        """포트폴리오 보호 신호 생성"""
        try:
            protection_signals = {
                'stop_loss_triggered': False,
                'take_profit_triggered': False,
                'emergency_stop_triggered': False,
                'hedging_recommended': False,
                'position_reduction_recommended': False,
                'reason': None,
                'recommended_action': 'hold'
            }
            
            # 손절매 신호
            if risk_metrics.max_drawdown > self.risk_limits['max_drawdown_limit']:
                protection_signals['stop_loss_triggered'] = True
                protection_signals['reason'] = 'max_drawdown_exceeded'
                protection_signals['recommended_action'] = 'reduce_position'
            
            # VaR 기반 손절매
            if risk_metrics.var_95 > self.risk_limits['max_var_limit']:
                protection_signals['stop_loss_triggered'] = True
                protection_signals['reason'] = 'var_exceeded'
                protection_signals['recommended_action'] = 'reduce_position'
            
            # 비상 손절매
            emergency_stop_threshold = self.stop_loss_config.get('emergency_stop', 0.12)
            if risk_metrics.max_drawdown > emergency_stop_threshold:
                protection_signals['emergency_stop_triggered'] = True
                protection_signals['reason'] = 'emergency_stop'
                protection_signals['recommended_action'] = 'close_all_positions'
            
            # 헤징 권장
            if risk_metrics.volatility > self.risk_limits['max_volatility']:
                protection_signals['hedging_recommended'] = True
                protection_signals['reason'] = 'high_volatility'
                protection_signals['recommended_action'] = 'hedge_positions'
            
            # 포지션 감소 권장
            if risk_metrics.correlation > self.risk_limits['max_correlation']:
                protection_signals['position_reduction_recommended'] = True
                protection_signals['reason'] = 'high_correlation'
                protection_signals['recommended_action'] = 'reduce_positions'
            
            return protection_signals
            
        except Exception as e:
            self.logger.error(f"포트폴리오 보호 신호 생성 실패: {e}")
            return {'stop_loss_triggered': False, 'take_profit_triggered': False, 'emergency_stop_triggered': False, 'hedging_recommended': False, 'position_reduction_recommended': False, 'reason': None, 'recommended_action': 'hold'}
    
    def _calculate_volatility_adjustment(self, risk_metrics: RiskMetrics, 
                                       market_data: Dict) -> Dict[str, float]:
        """동적 변동성 조절"""
        try:
            current_volatility = risk_metrics.volatility
            target_volatility = self.volatility_target
            
            # 변동성 조절 계수
            if current_volatility > target_volatility * 1.5:
                adjustment_factor = 0.6  # 변동성이 높으면 포지션 크기 대폭 감소
            elif current_volatility > target_volatility * 1.2:
                adjustment_factor = 0.8  # 변동성이 높으면 포지션 크기 감소
            elif current_volatility < target_volatility * 0.5:
                adjustment_factor = 1.3  # 변동성이 낮으면 포지션 크기 증가
            elif current_volatility < target_volatility * 0.8:
                adjustment_factor = 1.1  # 변동성이 낮으면 포지션 크기 약간 증가
            else:
                adjustment_factor = 1.0  # 변동성이 적정하면 유지
            
            return {
                'current_volatility': current_volatility,
                'target_volatility': target_volatility,
                'adjustment_factor': adjustment_factor,
                'volatility_ratio': current_volatility / target_volatility
            }
            
        except Exception as e:
            self.logger.error(f"동적 변동성 조절 계산 실패: {e}")
            return {'current_volatility': 0.15, 'target_volatility': 0.12, 'adjustment_factor': 1.0, 'volatility_ratio': 1.25}
    
    def _generate_monitoring_alerts(self, risk_metrics: RiskMetrics, 
                                  portfolio_state: Dict, 
                                  regime_info: Dict) -> List[Dict[str, Any]]:
        """실시간 리스크 모니터링 알림 생성"""
        try:
            alerts = []
            
            # 변동성 알림
            if risk_metrics.volatility > self.risk_limits['max_volatility']:
                alerts.append({
                    'type': 'volatility_warning',
                    'level': 'high',
                    'message': f"변동성이 한계를 초과했습니다: {risk_metrics.volatility:.2%}",
                    'recommendation': '포지션 크기를 줄이세요'
                })
            
            # 샤프 비율 알림
            if risk_metrics.sharpe_ratio < 0.5:
                alerts.append({
                    'type': 'sharpe_ratio_warning',
                    'level': 'medium',
                    'message': f"샤프 비율이 낮습니다: {risk_metrics.sharpe_ratio:.2f}",
                    'recommendation': '투자 전략을 재검토하세요'
                })
            
            # 최대 낙폭 알림
            if risk_metrics.max_drawdown > self.risk_limits['max_drawdown_limit'] * 0.8:
                alerts.append({
                    'type': 'drawdown_warning',
                    'level': 'high',
                    'message': f"최대 낙폭이 한계에 근접했습니다: {risk_metrics.max_drawdown:.2%}",
                    'recommendation': '리스크 관리를 강화하세요'
                })
            
            # VaR 알림
            if risk_metrics.var_95 > self.risk_limits['max_var_limit'] * 0.8:
                alerts.append({
                    'type': 'var_warning',
                    'level': 'medium',
                    'message': f"VaR이 한계에 근접했습니다: {risk_metrics.var_95:.2%}",
                    'recommendation': '포지션을 재검토하세요'
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"실시간 리스크 모니터링 알림 생성 실패: {e}")
            return []
    
    def _generate_advanced_recommendations(self, risk_level: RiskLevel, 
                                         risk_metrics: RiskMetrics, 
                                         position_sizing: PositionSizing,
                                         protection_signals: Dict) -> List[str]:
        """고급 리스크 관리 권장사항 생성"""
        try:
            recommendations = []
            
            # 리스크 레벨 기반 권장사항
            if risk_level == RiskLevel.CRITICAL:
                recommendations.append("위험 수준이 매우 높습니다. 즉시 모든 포지션을 정리하세요.")
            elif risk_level == RiskLevel.HIGH:
                recommendations.append("위험 수준이 높습니다. 포지션 크기를 대폭 줄이고 리스크를 모니터링하세요.")
            elif risk_level == RiskLevel.MEDIUM:
                recommendations.append("위험 수준이 보통입니다. 현재 포지션을 유지하되 주의 깊게 모니터링하세요.")
            elif risk_level == RiskLevel.LOW:
                recommendations.append("위험 수준이 낮습니다. 현재 전략을 유지하세요.")
            else:
                recommendations.append("위험 수준이 매우 낮습니다. 포지션 크기를 약간 늘릴 수 있습니다.")
            
            # 포지션 크기 기반 권장사항
            if position_sizing.final_size < position_sizing.base_size * 0.5:
                recommendations.append("포지션 크기가 크게 줄어들었습니다. 시장 상황을 재평가하세요.")
            
            # 보호 신호 기반 권장사항
            if protection_signals['emergency_stop_triggered']:
                recommendations.append("비상 손절매가 발생했습니다. 모든 포지션을 즉시 정리하세요.")
            elif protection_signals['stop_loss_triggered']:
                recommendations.append("손절매 신호가 발생했습니다. 포지션을 줄이세요.")
            elif protection_signals['hedging_recommended']:
                recommendations.append("헤징이 권장됩니다. 포지션을 보호하세요.")
            
            # 성과 기반 권장사항
            if risk_metrics.sharpe_ratio < 1.0:
                recommendations.append("샤프 비율이 낮습니다. 투자 전략을 최적화하세요.")
            
            if risk_metrics.volatility > self.volatility_target * 1.5:
                recommendations.append("변동성이 높습니다. 포트폴리오 다각화를 강화하세요.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"고급 리스크 관리 권장사항 생성 실패: {e}")
            return ["리스크 관리 시스템 오류로 인한 권장사항을 생성할 수 없습니다."]
    
    def _update_risk_history(self, risk_result: Dict):
        """리스크 히스토리 업데이트"""
        try:
            self.risk_history.append({
                'timestamp': pd.Timestamp.now(),
                'risk_level': risk_result['risk_level'].value,
                'volatility': risk_result['risk_metrics'].volatility,
                'max_drawdown': risk_result['risk_metrics'].max_drawdown,
                'sharpe_ratio': risk_result['risk_metrics'].sharpe_ratio,
                'position_size': risk_result['position_sizing'].final_size
            })
            
            # 변동성 히스토리 업데이트
            self.volatility_history.append(risk_result['risk_metrics'].volatility)
            
            # 낙폭 히스토리 업데이트
            self.drawdown_history.append(risk_result['risk_metrics'].max_drawdown)
            
        except Exception as e:
            self.logger.error(f"리스크 히스토리 업데이트 실패: {e}")
    
    # 기존 메서드들 (간단한 구현)
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
    
    def _calculate_beta(self, returns: List[float], market_data: Dict) -> float:
        """베타 계산"""
        try:
            # 임시 구현
            return 1.0
        except Exception as e:
            self.logger.error(f"베타 계산 실패: {e}")
            return 1.0
    
    def _calculate_correlation(self, returns: List[float], market_data: Dict) -> float:
        """상관관계 계산"""
        try:
            # 임시 구현
            return 0.3
        except Exception as e:
            self.logger.error(f"상관관계 계산 실패: {e}")
            return 0.3
    
    def _calculate_tracking_error(self, returns: List[float], market_data: Dict) -> float:
        """추적 오차 계산"""
        try:
            # 임시 구현
            return 0.05
        except Exception as e:
            self.logger.error(f"추적 오차 계산 실패: {e}")
            return 0.05
    
    def _calculate_information_ratio(self, returns: List[float], market_data: Dict) -> float:
        """정보 비율 계산"""
        try:
            # 임시 구현
            return 0.5
        except Exception as e:
            self.logger.error(f"정보 비율 계산 실패: {e}")
            return 0.5
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """기본 리스크 지표 반환"""
        return RiskMetrics(
            var_95=0.02,
            var_99=0.03,
            cvar_95=0.03,
            cvar_99=0.04,
            max_drawdown=-0.05,
            volatility=0.12,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=2.0,
            beta=1.0,
            correlation=0.3,
            tracking_error=0.05,
            information_ratio=0.5
        )
    
    def _get_default_advanced_risk_result(self) -> Dict[str, Any]:
        """기본 고급 리스크 관리 결과 반환"""
        return {
            'risk_level': RiskLevel.MEDIUM,
            'risk_metrics': self._get_default_risk_metrics(),
            'position_sizing': PositionSizing(
                base_size=self.base_position_size,
                volatility_adjusted_size=self.base_position_size,
                risk_adjusted_size=self.base_position_size,
                regime_adjusted_size=self.base_position_size,
                final_size=self.base_position_size,
                max_size=self.risk_limits['max_position_size'],
                min_size=0.01,
                confidence_factor=0.5
            ),
            'protection_signals': {'stop_loss_triggered': False, 'take_profit_triggered': False, 'emergency_stop_triggered': False, 'hedging_recommended': False, 'position_reduction_recommended': False, 'reason': None, 'recommended_action': 'hold'},
            'volatility_adjustment': {'current_volatility': 0.12, 'target_volatility': 0.12, 'adjustment_factor': 1.0, 'volatility_ratio': 1.0},
            'monitoring_alerts': [],
            'regime_adjustment': {'position_multiplier': 1.0, 'risk_multiplier': 1.0, 'regime': 'unknown', 'confidence': 0.5, 'risk_factor': 1.0},
            'risk_limits': self.risk_limits,
            'recommendations': ["기본 고급 리스크 관리 설정을 사용합니다."]
        }


# 편의 함수들
def create_advanced_risk_manager(config: Dict) -> AdvancedRiskManager:
    """고급 리스크 관리자 생성 편의 함수"""
    return AdvancedRiskManager(config)


def manage_advanced_portfolio_risk(portfolio_state: Dict, market_data: Dict, 
                                 regime_info: Dict, config: Dict) -> Dict[str, Any]:
    """고급 포트폴리오 리스크 관리 편의 함수"""
    risk_manager = create_advanced_risk_manager(config)
    return risk_manager.manage_portfolio_risk(portfolio_state, market_data, regime_info)
