"""
변동성 기반 포지션 크기 조절 시스템
Volatility-Based Position Sizing System

실시간 변동성 모니터링 및 포지션 크기 자동 조절
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .volatility_calculator import VolatilityCalculator
from .dynamic_position_sizing import DynamicPositionSizing

logger = logging.getLogger(__name__)

class VolatilityBasedSizing:
    """변동성 기반 포지션 크기 조절 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        변동성 기반 포지션 크기 조절 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 변동성 관리 파라미터
        self.volatility_target = config.get('volatility_target', 0.12)
        self.volatility_tolerance = config.get('volatility_tolerance', 0.02)
        self.max_volatility = config.get('max_volatility', 0.30)
        self.min_volatility = config.get('min_volatility', 0.05)
        
        # 포지션 크기 파라미터
        self.base_position_size = config.get('base_position_size', 0.06)
        self.max_position_size = config.get('max_position_size', 0.12)
        self.min_position_size = config.get('min_position_size', 0.01)
        
        # 변동성 기반 조절 파라미터
        self.volatility_scaling_factor = config.get('volatility_scaling_factor', 1.5)
        self.volatility_adjustment_speed = config.get('volatility_adjustment_speed', 0.1)
        
        # 변동성 계산기 초기화
        self.volatility_calculator = VolatilityCalculator()
        
        # 동적 포지션 크기 조절 시스템 초기화
        self.dynamic_sizing = DynamicPositionSizing(config)
        
        logger.info(f"변동성 기반 포지션 크기 조절 시스템 초기화 완료")
        logger.info(f"변동성 목표: {self.volatility_target:.1%}")
        logger.info(f"변동성 허용 범위: ±{self.volatility_tolerance:.1%}")
        logger.info(f"최대 변동성: {self.max_volatility:.1%}")
        logger.info(f"최소 변동성: {self.min_volatility:.1%}")
    
    def calculate_volatility_adjusted_position_size(self, returns: np.ndarray,
                                                  current_volatility: float,
                                                  target_volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        변동성 조정 포지션 크기 계산
        
        Args:
            returns: 수익률 배열
            current_volatility: 현재 변동성
            target_volatility: 목표 변동성 (기본값: 설정값 사용)
            
        Returns:
            변동성 조정 포지션 크기 결과
        """
        try:
            logger.info("변동성 조정 포지션 크기 계산 시작")
            
            if target_volatility is None:
                target_volatility = self.volatility_target
            
            # 변동성 비율 계산
            if current_volatility > 0:
                volatility_ratio = target_volatility / current_volatility
            else:
                volatility_ratio = 1.0
            
            # 포지션 크기 조절
            if volatility_ratio > 1.0:
                # 현재 변동성이 목표보다 낮음 -> 포지션 크기 증가 가능
                adjustment_factor = min(volatility_ratio, self.volatility_scaling_factor)
            else:
                # 현재 변동성이 목표보다 높음 -> 포지션 크기 감소
                adjustment_factor = volatility_ratio
            
            # 조정된 포지션 크기 계산
            adjusted_position_size = self.base_position_size * adjustment_factor
            
            # 최대/최소 포지션 크기 제한
            final_position_size = np.clip(adjusted_position_size, self.min_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'volatility_ratio': volatility_ratio,
                'adjustment_factor': adjustment_factor,
                'current_volatility': current_volatility,
                'target_volatility': target_volatility,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'min_position_size': self.min_position_size,
                'method': 'volatility_adjusted'
            }
            
            logger.info(f"변동성 조정 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"변동성 조정 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_volatility_regime_position_size(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        변동성 체제 기반 포지션 크기 계산
        
        Args:
            returns: 수익률 배열
            
        Returns:
            변동성 체제 기반 포지션 크기 결과
        """
        try:
            logger.info("변동성 체제 기반 포지션 크기 계산 시작")
            
            # 변동성 체제 감지
            regime_result = self.volatility_calculator.detect_volatility_regime(
                returns, 
                low_threshold=self.min_volatility,
                high_threshold=self.max_volatility
            )
            
            if 'error' in regime_result:
                return {'error': regime_result['error']}
            
            current_volatility = regime_result['current_volatility']
            regime = regime_result['regime']
            trend = regime_result['trend']
            
            # 체제별 포지션 크기 조절
            if regime == 'low_volatility':
                # 낮은 변동성 -> 포지션 크기 증가
                regime_multiplier = 1.2
                regime_description = "낮은 변동성으로 포지션 크기 증가"
            elif regime == 'high_volatility':
                # 높은 변동성 -> 포지션 크기 감소
                regime_multiplier = 0.6
                regime_description = "높은 변동성으로 포지션 크기 감소"
            else:
                # 정상 변동성 -> 기본 크기 유지
                regime_multiplier = 1.0
                regime_description = "정상 변동성으로 기본 포지션 크기 유지"
            
            # 트렌드별 추가 조절
            if trend == 'increasing':
                # 변동성 증가 중 -> 추가 감소
                trend_multiplier = 0.8
                trend_description = "변동성 증가로 추가 감소"
            elif trend == 'decreasing':
                # 변동성 감소 중 -> 추가 증가
                trend_multiplier = 1.1
                trend_description = "변동성 감소로 추가 증가"
            else:
                # 안정 -> 변화 없음
                trend_multiplier = 1.0
                trend_description = "변동성 안정으로 변화 없음"
            
            # 최종 조정 계수
            total_multiplier = regime_multiplier * trend_multiplier
            
            # 포지션 크기 계산
            adjusted_position_size = self.base_position_size * total_multiplier
            
            # 최대/최소 포지션 크기 제한
            final_position_size = np.clip(adjusted_position_size, self.min_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'regime': regime,
                'trend': trend,
                'current_volatility': current_volatility,
                'regime_multiplier': regime_multiplier,
                'trend_multiplier': trend_multiplier,
                'total_multiplier': total_multiplier,
                'regime_description': regime_description,
                'trend_description': trend_description,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'min_position_size': self.min_position_size,
                'method': 'volatility_regime_based'
            }
            
            logger.info(f"변동성 체제 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"변동성 체제 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_volatility_forecast_position_size(self, returns: np.ndarray,
                                                  forecast_horizon: int = 5) -> Dict[str, Any]:
        """
        변동성 예측 기반 포지션 크기 계산
        
        Args:
            returns: 수익률 배열
            forecast_horizon: 예측 기간 (일)
            
        Returns:
            변동성 예측 기반 포지션 크기 결과
        """
        try:
            logger.info(f"변동성 예측 기반 포지션 크기 계산 시작 (예측 기간: {forecast_horizon}일)")
            
            # 변동성 예측
            forecast_result = self.volatility_calculator.calculate_volatility_forecast(
                returns, forecast_horizon
            )
            
            if 'error' in forecast_result:
                return {'error': forecast_result['error']}
            
            current_volatility = forecast_result['current_volatility']
            forecast_volatility = forecast_result['forecast_volatility']
            upper_bound = forecast_result['upper_bound']
            lower_bound = forecast_result['lower_bound']
            
            # 예측 변동성 기반 포지션 크기 조절
            if forecast_volatility > self.volatility_target:
                # 예측 변동성이 목표보다 높음 -> 포지션 크기 감소
                forecast_ratio = self.volatility_target / forecast_volatility
                forecast_description = "예측 변동성 높음으로 포지션 크기 감소"
            else:
                # 예측 변동성이 목표보다 낮음 -> 포지션 크기 증가 가능
                forecast_ratio = min(forecast_volatility / self.volatility_target, self.volatility_scaling_factor)
                forecast_description = "예측 변동성 낮음으로 포지션 크기 증가"
            
            # 불확실성 고려 (예측 구간 폭)
            uncertainty = (upper_bound - lower_bound) / forecast_volatility if forecast_volatility > 0 else 0
            uncertainty_adjustment = 1.0 - min(uncertainty * 0.5, 0.3)  # 불확실성이 높으면 포지션 크기 감소
            
            # 최종 조정 계수
            total_adjustment = forecast_ratio * uncertainty_adjustment
            
            # 포지션 크기 계산
            adjusted_position_size = self.base_position_size * total_adjustment
            
            # 최대/최소 포지션 크기 제한
            final_position_size = np.clip(adjusted_position_size, self.min_position_size, self.max_position_size)
            
            result = {
                'position_size': final_position_size,
                'current_volatility': current_volatility,
                'forecast_volatility': forecast_volatility,
                'forecast_horizon': forecast_horizon,
                'forecast_ratio': forecast_ratio,
                'uncertainty': uncertainty,
                'uncertainty_adjustment': uncertainty_adjustment,
                'total_adjustment': total_adjustment,
                'forecast_description': forecast_description,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'min_position_size': self.min_position_size,
                'method': 'volatility_forecast_based'
            }
            
            logger.info(f"변동성 예측 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"변동성 예측 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_comprehensive_volatility_position_size(self, returns: np.ndarray,
                                                       weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        종합 변동성 기반 포지션 크기 계산 (모든 방법 통합)
        
        Args:
            returns: 수익률 배열
            weights: 각 방법의 가중치 (선택사항)
            
        Returns:
            종합 변동성 기반 포지션 크기 결과
        """
        try:
            logger.info("종합 변동성 기반 포지션 크기 계산 시작")
            
            # 기본 가중치 설정
            if weights is None:
                weights = {
                    'volatility_adjusted': 0.4,
                    'volatility_regime': 0.3,
                    'volatility_forecast': 0.3
                }
            
            # 현재 변동성 계산
            current_vol_result = self.volatility_calculator.calculate_comprehensive_volatility(returns)
            if 'error' in current_vol_result:
                return {'error': current_vol_result['error']}
            
            current_volatility = current_vol_result['average_volatility']
            
            # 각 방법별 포지션 크기 계산
            adjusted_result = self.calculate_volatility_adjusted_position_size(returns, current_volatility)
            regime_result = self.calculate_volatility_regime_position_size(returns)
            forecast_result = self.calculate_volatility_forecast_position_size(returns)
            
            # 결과 수집
            individual_results = {
                'volatility_adjusted': adjusted_result,
                'volatility_regime': regime_result,
                'volatility_forecast': forecast_result
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
            
            # 최대/최소 포지션 크기 제한
            final_position_size = np.clip(final_position_size, self.min_position_size, self.max_position_size)
            
            comprehensive_result = {
                'final_position_size': final_position_size,
                'current_volatility': current_volatility,
                'individual_results': individual_results,
                'weights': weights,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'min_position_size': self.min_position_size,
                'method': 'comprehensive_volatility_based'
            }
            
            logger.info(f"종합 변동성 기반 포지션 크기 계산 완료: {final_position_size:.4f} ({final_position_size*100:.2f}%)")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"종합 변동성 기반 포지션 크기 계산 실패: {e}")
            return {'error': str(e)}
    
    def get_volatility_sizing_signal(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        변동성 기반 포지션 크기 조절 신호 생성
        
        Args:
            returns: 수익률 배열
            
        Returns:
            변동성 기반 포지션 크기 조절 신호
        """
        try:
            logger.info("변동성 기반 포지션 크기 조절 신호 생성 시작")
            
            # 종합 변동성 기반 포지션 크기 계산
            position_result = self.calculate_comprehensive_volatility_position_size(returns)
            
            if 'error' in position_result:
                return {'error': position_result['error']}
            
            final_position_size = position_result['final_position_size']
            current_volatility = position_result['current_volatility']
            
            # 신호 생성
            signal = {
                'action': 'adjust_position',
                'recommended_position_size': final_position_size,
                'position_size_percentage': final_position_size * 100,
                'current_volatility': current_volatility,
                'volatility_target': self.volatility_target,
                'volatility_deviation': abs(current_volatility - self.volatility_target),
                'risk_level': self._assess_volatility_risk_level(current_volatility),
                'adjustment_reason': self._get_volatility_adjustment_reason(position_result),
                'timestamp': datetime.now().isoformat(),
                'details': position_result
            }
            
            # 긴급 신호 확인
            if current_volatility > self.max_volatility:
                signal['action'] = 'emergency_reduce'
                signal['emergency_reason'] = '최대 변동성 한계 초과'
            elif current_volatility < self.min_volatility:
                signal['action'] = 'increase_position'
                signal['adjustment_reason'] = '최소 변동성 이하로 포지션 증가 가능'
            
            logger.info(f"변동성 기반 포지션 크기 조절 신호 생성 완료: {signal['action']}")
            
            return signal
            
        except Exception as e:
            logger.error(f"변동성 기반 포지션 크기 조절 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _assess_volatility_risk_level(self, current_volatility: float) -> str:
        """변동성 리스크 수준 평가"""
        try:
            if current_volatility > self.max_volatility:
                return 'extreme_high'
            elif current_volatility > self.volatility_target + self.volatility_tolerance:
                return 'high'
            elif current_volatility < self.volatility_target - self.volatility_tolerance:
                return 'low'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"변동성 리스크 수준 평가 실패: {e}")
            return 'unknown'
    
    def _get_volatility_adjustment_reason(self, position_result: Dict[str, Any]) -> str:
        """변동성 조절 이유 설명"""
        try:
            reasons = []
            
            current_volatility = position_result.get('current_volatility', 0)
            volatility_target = self.volatility_target
            
            # 변동성 목표 대비 편차
            if current_volatility > volatility_target + self.volatility_tolerance:
                reasons.append('변동성 목표 초과')
            elif current_volatility < volatility_target - self.volatility_tolerance:
                reasons.append('변동성 목표 미달')
            
            # 개별 방법별 조절 이유
            individual_results = position_result.get('individual_results', {})
            
            # 변동성 체제 기반 조절
            regime_result = individual_results.get('volatility_regime', {})
            if regime_result.get('regime') == 'high_volatility':
                reasons.append('높은 변동성 체제')
            elif regime_result.get('regime') == 'low_volatility':
                reasons.append('낮은 변동성 체제')
            
            # 변동성 예측 기반 조절
            forecast_result = individual_results.get('volatility_forecast', {})
            if forecast_result.get('forecast_volatility', 0) > volatility_target:
                reasons.append('변동성 증가 예측')
            
            if not reasons:
                return '정상 범위'
            else:
                return ', '.join(reasons)
                
        except Exception as e:
            logger.error(f"변동성 조절 이유 설명 생성 실패: {e}")
            return '알 수 없음'

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("변동성 기반 포지션 크기 조절 시스템 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {
            'volatility_target': 0.12,
            'volatility_tolerance': 0.02,
            'max_volatility': 0.30,
            'min_volatility': 0.05,
            'base_position_size': 0.06,
            'max_position_size': 0.12,
            'min_position_size': 0.01,
            'volatility_scaling_factor': 1.5,
            'volatility_adjustment_speed': 0.1
        }
        
        # 테스트 데이터 생성
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 252)
        
        # 변동성 기반 포지션 크기 조절 시스템 생성
        volatility_sizing = VolatilityBasedSizing(config)
        
        # 종합 변동성 기반 포지션 크기 계산
        result = volatility_sizing.calculate_comprehensive_volatility_position_size(test_returns)
        
        print(f"테스트 결과:")
        print(f"  최종 포지션 크기: {result['final_position_size']:.4f} ({result['final_position_size']*100:.2f}%)")
        print(f"  현재 변동성: {result['current_volatility']:.4f} ({result['current_volatility']*100:.2f}%)")
        print(f"  변동성 목표: {volatility_sizing.volatility_target:.4f} ({volatility_sizing.volatility_target*100:.2f}%)")
        
        # 개별 방법별 결과
        print(f"\n개별 방법별 결과:")
        for method, method_result in result['individual_results'].items():
            if 'error' not in method_result:
                print(f"  {method}: {method_result['position_size']:.4f} ({method_result['position_size']*100:.2f}%)")
        
        # 변동성 기반 포지션 크기 조절 신호 생성
        signal = volatility_sizing.get_volatility_sizing_signal(test_returns)
        
        print(f"\n변동성 기반 포지션 크기 조절 신호:")
        print(f"  액션: {signal['action']}")
        print(f"  권장 포지션 크기: {signal['recommended_position_size']:.4f}")
        print(f"  리스크 수준: {signal['risk_level']}")
        print(f"  조절 이유: {signal['adjustment_reason']}")
        
        print(f"\n[성공] 변동성 기반 포지션 크기 조절 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 변동성 기반 포지션 크기 조절 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
