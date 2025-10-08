"""
동적 포지션 크기 조절 시스템
Dynamic Position Sizing System

VaR, CVaR 기반 동적 포지션 크기 조절 및 리스크 관리
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .var_calculator import VaRCalculator
from .cvar_calculator import CVaRCalculator

logger = logging.getLogger(__name__)

class DynamicPositionSizing:
    """동적 포지션 크기 조절 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        동적 포지션 크기 조절 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 리스크 관리 파라미터
        self.max_position_size = config.get('max_position_size', 0.12)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.08)
        self.volatility_target = config.get('volatility_target', 0.12)
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # VaR/CVaR 계산기 초기화
        self.var_calculator = VaRCalculator(
            confidence_level=self.confidence_level,
            lookback_period=252
        )
        self.cvar_calculator = CVaRCalculator(
            confidence_level=self.confidence_level,
            lookback_period=252
        )
        
        # 포지션 크기 조절 파라미터
        self.base_position_size = config.get('base_position_size', 0.06)
        self.risk_scaling_factor = config.get('risk_scaling_factor', 2.0)
        self.volatility_scaling_factor = config.get('volatility_scaling_factor', 1.5)
        
        logger.info(f"동적 포지션 크기 조절 시스템 초기화 완료")
        logger.info(f"최대 포지션 크기: {self.max_position_size:.1%}")
        logger.info(f"최대 낙폭 한계: {self.max_drawdown_limit:.1%}")
        logger.info(f"변동성 목표: {self.volatility_target:.1%}")
    
    def calculate_var_based_position_size(self, returns: np.ndarray, 
                                        current_portfolio_value: float,
                                        target_var_limit: float = 0.02) -> Dict[str, Any]:
        """
        VaR 기반 포지션 크기 계산
        
        Args:
            returns: 수익률 배열
            current_portfolio_value: 현재 포트폴리오 가치
            target_var_limit: 목표 VaR 한계 (기본값: 2%)
            
        Returns:
            포지션 크기 결과
        """
        try:
            logger.info("VaR 기반 포지션 크기 계산 시작")
            
            # VaR 계산
            var_result = self.var_calculator.calculate_comprehensive_var(
                returns, current_portfolio_value
            )
            
            if 'error' in var_result:
                return {'error': var_result['error']}
            
            current_var = var_result['average_var_percentage']
            
            # VaR 기반 포지션 크기 조절
            if current_var > 0:
                # 목표 VaR 대비 현재 VaR 비율
                var_ratio = target_var_limit / current_var
                
                # 포지션 크기 조절 (VaR가 높으면 포지션 크기 감소)
                adjusted_position_size = self.base_position_size * min(var_ratio, 1.0)
            else:
                adjusted_position_size = self.base_position_size
            
            # 최대 포지션 크기 제한
            final_position_size = min(adjusted_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'var_based_adjustment': var_ratio if current_var > 0 else 1.0,
                'current_var': current_var,
                'target_var': target_var_limit,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'method': 'var_based'
            }
            
            logger.info(f"VaR 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"VaR 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_cvar_based_position_size(self, returns: np.ndarray, 
                                         current_portfolio_value: float,
                                         target_cvar_limit: float = 0.025) -> Dict[str, Any]:
        """
        CVaR 기반 포지션 크기 계산
        
        Args:
            returns: 수익률 배열
            current_portfolio_value: 현재 포트폴리오 가치
            target_cvar_limit: 목표 CVaR 한계 (기본값: 2.5%)
            
        Returns:
            포지션 크기 결과
        """
        try:
            logger.info("CVaR 기반 포지션 크기 계산 시작")
            
            # CVaR 계산
            cvar_result = self.cvar_calculator.calculate_comprehensive_cvar(
                returns, current_portfolio_value
            )
            
            if 'error' in cvar_result:
                return {'error': cvar_result['error']}
            
            current_cvar = cvar_result['average_cvar_percentage']
            
            # CVaR 기반 포지션 크기 조절
            if current_cvar > 0:
                # 목표 CVaR 대비 현재 CVaR 비율
                cvar_ratio = target_cvar_limit / current_cvar
                
                # 포지션 크기 조절 (CVaR가 높으면 포지션 크기 감소)
                adjusted_position_size = self.base_position_size * min(cvar_ratio, 1.0)
            else:
                adjusted_position_size = self.base_position_size
            
            # 최대 포지션 크기 제한
            final_position_size = min(adjusted_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'cvar_based_adjustment': cvar_ratio if current_cvar > 0 else 1.0,
                'current_cvar': current_cvar,
                'target_cvar': target_cvar_limit,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'method': 'cvar_based'
            }
            
            logger.info(f"CVaR 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"CVaR 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_volatility_based_position_size(self, returns: np.ndarray,
                                               current_volatility: float) -> Dict[str, Any]:
        """
        변동성 기반 포지션 크기 계산
        
        Args:
            returns: 수익률 배열
            current_volatility: 현재 변동성
            
        Returns:
            포지션 크기 결과
        """
        try:
            logger.info("변동성 기반 포지션 크기 계산 시작")
            
            # 변동성 기반 포지션 크기 조절
            if current_volatility > 0:
                # 목표 변동성 대비 현재 변동성 비율
                volatility_ratio = self.volatility_target / current_volatility
                
                # 포지션 크기 조절 (변동성이 높으면 포지션 크기 감소)
                adjusted_position_size = self.base_position_size * min(volatility_ratio, self.volatility_scaling_factor)
            else:
                adjusted_position_size = self.base_position_size
            
            # 최대 포지션 크기 제한
            final_position_size = min(adjusted_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'volatility_adjustment': volatility_ratio if current_volatility > 0 else 1.0,
                'current_volatility': current_volatility,
                'target_volatility': self.volatility_target,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'method': 'volatility_based'
            }
            
            logger.info(f"변동성 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"변동성 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_drawdown_based_position_size(self, current_drawdown: float,
                                             portfolio_value: float,
                                             peak_value: float) -> Dict[str, Any]:
        """
        낙폭 기반 포지션 크기 계산
        
        Args:
            current_drawdown: 현재 낙폭
            portfolio_value: 현재 포트폴리오 가치
            peak_value: 최고점 포트폴리오 가치
            
        Returns:
            포지션 크기 결과
        """
        try:
            logger.info("낙폭 기반 포지션 크기 계산 시작")
            
            # 낙폭 기반 포지션 크기 조절
            if current_drawdown < 0:
                # 낙폭이 클수록 포지션 크기 감소
                drawdown_ratio = abs(current_drawdown) / self.max_drawdown_limit
                
                # 낙폭이 한계에 가까울수록 포지션 크기 대폭 감소
                if drawdown_ratio >= 0.8:  # 80% 이상
                    adjustment_factor = 0.1  # 10%로 대폭 감소
                elif drawdown_ratio >= 0.6:  # 60% 이상
                    adjustment_factor = 0.3  # 30%로 감소
                elif drawdown_ratio >= 0.4:  # 40% 이상
                    adjustment_factor = 0.6  # 60%로 감소
                else:
                    adjustment_factor = 1.0  # 유지
                
                adjusted_position_size = self.base_position_size * adjustment_factor
            else:
                adjusted_position_size = self.base_position_size
            
            # 최대 포지션 크기 제한
            final_position_size = min(adjusted_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'drawdown_adjustment': adjustment_factor if current_drawdown < 0 else 1.0,
                'current_drawdown': current_drawdown,
                'max_drawdown_limit': self.max_drawdown_limit,
                'drawdown_ratio': drawdown_ratio if current_drawdown < 0 else 0.0,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'method': 'drawdown_based'
            }
            
            logger.info(f"낙폭 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"낙폭 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_comprehensive_position_size(self, returns: np.ndarray,
                                            current_portfolio_value: float,
                                            current_volatility: float,
                                            current_drawdown: float,
                                            peak_value: float,
                                            weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        종합 포지션 크기 계산 (모든 방법 통합)
        
        Args:
            returns: 수익률 배열
            current_portfolio_value: 현재 포트폴리오 가치
            current_volatility: 현재 변동성
            current_drawdown: 현재 낙폭
            peak_value: 최고점 포트폴리오 가치
            weights: 각 방법의 가중치 (선택사항)
            
        Returns:
            종합 포지션 크기 결과
        """
        try:
            logger.info("종합 포지션 크기 계산 시작")
            
            # 기본 가중치 설정
            if weights is None:
                weights = {
                    'var_based': 0.3,
                    'cvar_based': 0.3,
                    'volatility_based': 0.2,
                    'drawdown_based': 0.2
                }
            
            # 각 방법별 포지션 크기 계산
            var_result = self.calculate_var_based_position_size(returns, current_portfolio_value)
            cvar_result = self.calculate_cvar_based_position_size(returns, current_portfolio_value)
            volatility_result = self.calculate_volatility_based_position_size(returns, current_volatility)
            drawdown_result = self.calculate_drawdown_based_position_size(current_drawdown, current_portfolio_value, peak_value)
            
            # 결과 수집
            individual_results = {
                'var_based': var_result,
                'cvar_based': cvar_result,
                'volatility_based': volatility_result,
                'drawdown_based': drawdown_result
            }
            
            # 가중 평균 포지션 크기 계산
            weighted_position_size = 0.0
            total_weight = 0.0
            
            for method, result in individual_results.items():
                if 'error' not in result and 'position_size' in result:
                    weight = weights.get(method, 0.0)
                    weighted_position_size += result['position_size'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_position_size = weighted_position_size / total_weight
            else:
                final_position_size = self.base_position_size
            
            # 최대 포지션 크기 제한
            final_position_size = min(final_position_size, self.max_position_size)
            
            # 최소 포지션 크기 보장 (너무 작아지지 않도록)
            min_position_size = self.base_position_size * 0.1
            final_position_size = max(final_position_size, min_position_size)
            
            comprehensive_result = {
                'final_position_size': final_position_size,
                'individual_results': individual_results,
                'weights': weights,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'min_position_size': min_position_size,
                'method': 'comprehensive'
            }
            
            logger.info(f"종합 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"종합 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def get_position_sizing_signal(self, returns: np.ndarray,
                                 current_portfolio_value: float,
                                 current_volatility: float,
                                 current_drawdown: float,
                                 peak_value: float) -> Dict[str, Any]:
        """
        포지션 크기 조절 신호 생성
        
        Args:
            returns: 수익률 배열
            current_portfolio_value: 현재 포트폴리오 가치
            current_volatility: 현재 변동성
            current_drawdown: 현재 낙폭
            peak_value: 최고점 포트폴리오 가치
            
        Returns:
            포지션 크기 조절 신호
        """
        try:
            logger.info("포지션 크기 조절 신호 생성 시작")
            
            # 종합 포지션 크기 계산
            position_result = self.calculate_comprehensive_position_size(
                returns, current_portfolio_value, current_volatility, 
                current_drawdown, peak_value
            )
            
            if 'error' in position_result:
                return {'error': position_result['error']}
            
            final_position_size = position_result['final_position_size']
            
            # 신호 생성
            signal = {
                'action': 'adjust_position',
                'recommended_position_size': final_position_size,
                'position_size_percentage': final_position_size * 100,
                'risk_level': self._assess_risk_level(current_volatility, current_drawdown),
                'adjustment_reason': self._get_adjustment_reason(position_result),
                'timestamp': datetime.now().isoformat(),
                'details': position_result
            }
            
            # 긴급 신호 확인
            if current_drawdown < -self.max_drawdown_limit * 0.8:
                signal['action'] = 'emergency_reduce'
                signal['emergency_reason'] = '최대 낙폭 한계 근접'
            elif current_volatility > self.volatility_target * 2:
                signal['action'] = 'reduce_position'
                signal['adjustment_reason'] = '변동성 급증'
            
            logger.info(f"포지션 크기 조절 신호 생성 완료: {signal['action']}")
            
            return signal
            
        except Exception as e:
            logger.error(f"포지션 크기 조절 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _assess_risk_level(self, current_volatility: float, current_drawdown: float) -> str:
        """리스크 수준 평가"""
        try:
            risk_score = 0
            
            # 변동성 평가
            if current_volatility > self.volatility_target * 1.5:
                risk_score += 3
            elif current_volatility > self.volatility_target:
                risk_score += 1
            
            # 낙폭 평가
            if current_drawdown < -self.max_drawdown_limit * 0.75:
                risk_score += 3
            elif current_drawdown < -self.max_drawdown_limit * 0.5:
                risk_score += 1
            
            # 리스크 수준 결정
            if risk_score >= 5:
                return 'high'
            elif risk_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"리스크 수준 평가 실패: {e}")
            return 'unknown'
    
    def _get_adjustment_reason(self, position_result: Dict[str, Any]) -> str:
        """조절 이유 설명"""
        try:
            reasons = []
            
            individual_results = position_result.get('individual_results', {})
            
            # VaR 기반 조절
            var_result = individual_results.get('var_based', {})
            if var_result.get('var_based_adjustment', 1.0) < 0.8:
                reasons.append('VaR 한계 초과')
            
            # CVaR 기반 조절
            cvar_result = individual_results.get('cvar_based', {})
            if cvar_result.get('cvar_based_adjustment', 1.0) < 0.8:
                reasons.append('CVaR 한계 초과')
            
            # 변동성 기반 조절
            volatility_result = individual_results.get('volatility_based', {})
            if volatility_result.get('volatility_adjustment', 1.0) < 0.8:
                reasons.append('변동성 목표 초과')
            
            # 낙폭 기반 조절
            drawdown_result = individual_results.get('drawdown_based', {})
            if drawdown_result.get('drawdown_adjustment', 1.0) < 0.8:
                reasons.append('낙폭 한계 근접')
            
            if not reasons:
                return '정상 범위'
            else:
                return ', '.join(reasons)
                
        except Exception as e:
            logger.error(f"조절 이유 설명 생성 실패: {e}")
            return '알 수 없음'

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("동적 포지션 크기 조절 시스템 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {
            'max_position_size': 0.12,
            'max_drawdown_limit': 0.08,
            'volatility_target': 0.12,
            'confidence_level': 0.95,
            'base_position_size': 0.06,
            'risk_scaling_factor': 2.0,
            'volatility_scaling_factor': 1.5
        }
        
        # 테스트 데이터 생성
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 252)
        
        # 동적 포지션 크기 조절 시스템 생성
        position_sizing = DynamicPositionSizing(config)
        
        # 종합 포지션 크기 계산
        result = position_sizing.calculate_comprehensive_position_size(
            returns=test_returns,
            current_portfolio_value=10000,
            current_volatility=0.15,
            current_drawdown=-0.03,
            peak_value=10300
        )
        
        print(f"테스트 결과:")
        print(f"  최종 포지션 크기: {result['final_position_size']:.4f} ({result['final_position_size']*100:.2f}%)")
        print(f"  기본 포지션 크기: {result['base_position_size']:.4f}")
        print(f"  최대 포지션 크기: {result['max_position_size']:.4f}")
        
        # 개별 방법별 결과
        print(f"\n개별 방법별 결과:")
        for method, method_result in result['individual_results'].items():
            if 'error' not in method_result:
                print(f"  {method}: {method_result['position_size']:.4f} ({method_result['position_size']*100:.2f}%)")
        
        # 포지션 크기 조절 신호 생성
        signal = position_sizing.get_position_sizing_signal(
            returns=test_returns,
            current_portfolio_value=10000,
            current_volatility=0.15,
            current_drawdown=-0.03,
            peak_value=10300
        )
        
        print(f"\n포지션 크기 조절 신호:")
        print(f"  액션: {signal['action']}")
        print(f"  권장 포지션 크기: {signal['recommended_position_size']:.4f}")
        print(f"  리스크 수준: {signal['risk_level']}")
        print(f"  조절 이유: {signal['adjustment_reason']}")
        
        print(f"\n[성공] 동적 포지션 크기 조절 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 동적 포지션 크기 조절 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
