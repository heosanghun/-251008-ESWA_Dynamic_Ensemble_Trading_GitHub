"""
변동성 계산기
Volatility Calculator

실시간 변동성 계산 및 모니터링을 위한 고급 변동성 분석 도구
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VolatilityCalculator:
    """변동성 계산기 클래스"""
    
    def __init__(self, lookback_period: int = 252, annualization_factor: int = 252):
        """
        변동성 계산기 초기화
        
        Args:
            lookback_period: 백테스팅 기간 (기본값: 252일)
            annualization_factor: 연간화 인자 (기본값: 252일)
        """
        self.lookback_period = lookback_period
        self.annualization_factor = annualization_factor
        
        logger.info(f"변동성 계산기 초기화 완료")
        logger.info(f"백테스팅 기간: {lookback_period}일")
        logger.info(f"연간화 인자: {annualization_factor}")
    
    def calculate_simple_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """
        단순 변동성 계산 (표준편차 기반)
        
        Args:
            returns: 수익률 배열
            
        Returns:
            변동성 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'volatility': 0.0, 'method': 'simple'}
            
            # 단순 표준편차 계산
            volatility = np.std(returns) * np.sqrt(self.annualization_factor)
            
            result = {
                'volatility': volatility,
                'method': 'simple',
                'sample_size': len(returns),
                'mean_return': np.mean(returns),
                'std_return': np.std(returns)
            }
            
            logger.info(f"단순 변동성 계산 완료: {volatility:.4f} ({volatility*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"단순 변동성 계산 실패: {e}")
            return {'volatility': 0.0, 'method': 'simple', 'error': str(e)}
    
    def calculate_ewma_volatility(self, returns: np.ndarray, lambda_param: float = 0.94) -> Dict[str, float]:
        """
        EWMA (Exponentially Weighted Moving Average) 변동성 계산
        
        Args:
            returns: 수익률 배열
            lambda_param: EWMA 감쇠 인자 (기본값: 0.94)
            
        Returns:
            EWMA 변동성 결과
        """
        try:
            if len(returns) == 0:
                return {'volatility': 0.0, 'method': 'ewma'}
            
            # EWMA 가중치 계산
            weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))])
            weights = weights[::-1]  # 최신 데이터에 높은 가중치
            weights = weights / np.sum(weights)  # 정규화
            
            # 가중 평균 계산
            mean_return = np.average(returns, weights=weights)
            
            # 가중 분산 계산
            weighted_variance = np.average((returns - mean_return) ** 2, weights=weights)
            volatility = np.sqrt(weighted_variance * self.annualization_factor)
            
            result = {
                'volatility': volatility,
                'method': 'ewma',
                'lambda_param': lambda_param,
                'sample_size': len(returns),
                'mean_return': mean_return,
                'weighted_variance': weighted_variance
            }
            
            logger.info(f"EWMA 변동성 계산 완료: {volatility:.4f} ({volatility*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"EWMA 변동성 계산 실패: {e}")
            return {'volatility': 0.0, 'method': 'ewma', 'error': str(e)}
    
    def calculate_garch_volatility(self, returns: np.ndarray, 
                                 p: int = 1, q: int = 1) -> Dict[str, float]:
        """
        GARCH 변동성 계산 (간단한 구현)
        
        Args:
            returns: 수익률 배열
            p: GARCH 모델의 ARCH 항 수
            q: GARCH 모델의 GARCH 항 수
            
        Returns:
            GARCH 변동성 결과
        """
        try:
            if len(returns) < max(p, q) + 10:
                return {'volatility': 0.0, 'method': 'garch', 'error': 'Insufficient data'}
            
            # 간단한 GARCH(1,1) 모델 구현
            # 실제로는 arch 라이브러리를 사용하는 것이 좋음
            
            # 초기값 설정
            omega = 0.0001  # 상수항
            alpha = 0.1     # ARCH 계수
            beta = 0.85     # GARCH 계수
            
            # 조건부 분산 계산
            conditional_variance = np.zeros(len(returns))
            conditional_variance[0] = np.var(returns[:10])  # 초기값
            
            for t in range(1, len(returns)):
                conditional_variance[t] = omega + alpha * (returns[t-1] ** 2) + beta * conditional_variance[t-1]
            
            # 최근 변동성 계산
            recent_volatility = np.sqrt(conditional_variance[-1] * self.annualization_factor)
            
            result = {
                'volatility': recent_volatility,
                'method': 'garch',
                'p': p,
                'q': q,
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'conditional_variance': conditional_variance[-1],
                'sample_size': len(returns)
            }
            
            logger.info(f"GARCH 변동성 계산 완료: {recent_volatility:.4f} ({recent_volatility*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"GARCH 변동성 계산 실패: {e}")
            return {'volatility': 0.0, 'method': 'garch', 'error': str(e)}
    
    def calculate_realized_volatility(self, returns: np.ndarray, 
                                    window_size: int = 20) -> Dict[str, float]:
        """
        Realized Volatility 계산 (고빈도 데이터 기반)
        
        Args:
            returns: 수익률 배열
            window_size: 윈도우 크기
            
        Returns:
            Realized Volatility 결과
        """
        try:
            if len(returns) < window_size:
                return {'volatility': 0.0, 'method': 'realized', 'error': 'Insufficient data'}
            
            # 최근 윈도우의 수익률 제곱합
            recent_returns = returns[-window_size:]
            realized_variance = np.sum(recent_returns ** 2)
            realized_volatility = np.sqrt(realized_variance * self.annualization_factor / window_size)
            
            result = {
                'volatility': realized_volatility,
                'method': 'realized',
                'window_size': window_size,
                'realized_variance': realized_variance,
                'sample_size': len(returns)
            }
            
            logger.info(f"Realized Volatility 계산 완료: {realized_volatility:.4f} ({realized_volatility*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Realized Volatility 계산 실패: {e}")
            return {'volatility': 0.0, 'method': 'realized', 'error': str(e)}
    
    def calculate_parkinson_volatility(self, high_prices: np.ndarray, 
                                     low_prices: np.ndarray) -> Dict[str, float]:
        """
        Parkinson Volatility 계산 (고가-저가 기반)
        
        Args:
            high_prices: 고가 배열
            low_prices: 저가 배열
            
        Returns:
            Parkinson Volatility 결과
        """
        try:
            if len(high_prices) != len(low_prices) or len(high_prices) == 0:
                return {'volatility': 0.0, 'method': 'parkinson', 'error': 'Invalid data'}
            
            # Parkinson 공식: σ² = (1/4ln2) * Σ(ln(H/L))²
            log_hl_ratio = np.log(high_prices / low_prices)
            parkinson_variance = np.mean(log_hl_ratio ** 2) / (4 * np.log(2))
            parkinson_volatility = np.sqrt(parkinson_variance * self.annualization_factor)
            
            result = {
                'volatility': parkinson_volatility,
                'method': 'parkinson',
                'parkinson_variance': parkinson_variance,
                'sample_size': len(high_prices)
            }
            
            logger.info(f"Parkinson Volatility 계산 완료: {parkinson_volatility:.4f} ({parkinson_volatility*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Parkinson Volatility 계산 실패: {e}")
            return {'volatility': 0.0, 'method': 'parkinson', 'error': str(e)}
    
    def calculate_comprehensive_volatility(self, returns: np.ndarray,
                                         high_prices: Optional[np.ndarray] = None,
                                         low_prices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        종합 변동성 계산 (모든 방법 사용)
        
        Args:
            returns: 수익률 배열
            high_prices: 고가 배열 (선택사항)
            low_prices: 저가 배열 (선택사항)
            
        Returns:
            종합 변동성 결과
        """
        try:
            logger.info("종합 변동성 계산 시작")
            
            # 각 방법별 변동성 계산
            simple_vol = self.calculate_simple_volatility(returns)
            ewma_vol = self.calculate_ewma_volatility(returns)
            garch_vol = self.calculate_garch_volatility(returns)
            realized_vol = self.calculate_realized_volatility(returns)
            
            # Parkinson 변동성 (고가-저가 데이터가 있는 경우)
            parkinson_vol = None
            if high_prices is not None and low_prices is not None:
                parkinson_vol = self.calculate_parkinson_volatility(high_prices, low_prices)
            
            # 결과 통합
            volatility_results = {
                'simple': simple_vol,
                'ewma': ewma_vol,
                'garch': garch_vol,
                'realized': realized_vol
            }
            
            if parkinson_vol is not None:
                volatility_results['parkinson'] = parkinson_vol
            
            # 평균 변동성 계산
            valid_volatilities = [result['volatility'] for result in volatility_results.values() 
                                if 'volatility' in result and result['volatility'] > 0]
            
            if valid_volatilities:
                average_volatility = np.mean(valid_volatilities)
                median_volatility = np.median(valid_volatilities)
                std_volatility = np.std(valid_volatilities)
            else:
                average_volatility = 0.0
                median_volatility = 0.0
                std_volatility = 0.0
            
            comprehensive_result = {
                'individual_results': volatility_results,
                'average_volatility': average_volatility,
                'median_volatility': median_volatility,
                'std_volatility': std_volatility,
                'sample_size': len(returns),
                'annualization_factor': self.annualization_factor
            }
            
            logger.info(f"종합 변동성 계산 완료")
            logger.info(f"평균 변동성: {average_volatility:.4f} ({average_volatility*100:.2f}%)")
            logger.info(f"중간값 변동성: {median_volatility:.4f} ({median_volatility*100:.2f}%)")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"종합 변동성 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_volatility_forecast(self, returns: np.ndarray, 
                                    forecast_horizon: int = 1) -> Dict[str, Any]:
        """
        변동성 예측
        
        Args:
            returns: 수익률 배열
            forecast_horizon: 예측 기간 (일)
            
        Returns:
            변동성 예측 결과
        """
        try:
            logger.info(f"변동성 예측 시작 (예측 기간: {forecast_horizon}일)")
            
            # 현재 변동성 계산
            current_vol = self.calculate_comprehensive_volatility(returns)
            
            if 'error' in current_vol:
                return {'error': current_vol['error']}
            
            current_volatility = current_vol['average_volatility']
            
            # 간단한 변동성 예측 모델 (평균 회귀)
            # 실제로는 더 정교한 모델을 사용해야 함
            
            # 변동성의 평균 회귵 속도
            mean_reversion_speed = 0.1
            long_term_volatility = 0.15  # 장기 평균 변동성
            
            # 예측 변동성 계산
            forecast_volatility = current_volatility + mean_reversion_speed * (long_term_volatility - current_volatility)
            
            # 예측 구간 계산 (신뢰구간)
            confidence_interval = 0.95
            z_score = stats.norm.ppf((1 + confidence_interval) / 2)
            
            # 예측 오차 (간단한 추정)
            forecast_error = current_vol['std_volatility'] * np.sqrt(forecast_horizon)
            
            lower_bound = forecast_volatility - z_score * forecast_error
            upper_bound = forecast_volatility + z_score * forecast_error
            
            result = {
                'current_volatility': current_volatility,
                'forecast_volatility': forecast_volatility,
                'forecast_horizon': forecast_horizon,
                'confidence_interval': confidence_interval,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'forecast_error': forecast_error,
                'mean_reversion_speed': mean_reversion_speed,
                'long_term_volatility': long_term_volatility
            }
            
            logger.info(f"변동성 예측 완료: {forecast_volatility:.4f} ({forecast_volatility*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"변동성 예측 실패: {e}")
            return {'error': str(e)}
    
    def detect_volatility_regime(self, returns: np.ndarray, 
                               low_threshold: float = 0.1, 
                               high_threshold: float = 0.25) -> Dict[str, Any]:
        """
        변동성 체제 감지
        
        Args:
            returns: 수익률 배열
            low_threshold: 낮은 변동성 임계값
            high_threshold: 높은 변동성 임계값
            
        Returns:
            변동성 체제 감지 결과
        """
        try:
            logger.info("변동성 체제 감지 시작")
            
            # 현재 변동성 계산
            current_vol = self.calculate_comprehensive_volatility(returns)
            
            if 'error' in current_vol:
                return {'error': current_vol['error']}
            
            current_volatility = current_vol['average_volatility']
            
            # 변동성 체제 분류
            if current_volatility < low_threshold:
                regime = 'low_volatility'
                regime_description = '낮은 변동성'
            elif current_volatility > high_threshold:
                regime = 'high_volatility'
                regime_description = '높은 변동성'
            else:
                regime = 'normal_volatility'
                regime_description = '정상 변동성'
            
            # 변동성 체제 변화 감지
            if len(returns) >= 50:
                # 최근 25일과 이전 25일의 변동성 비교
                recent_vol = self.calculate_simple_volatility(returns[-25:])
                previous_vol = self.calculate_simple_volatility(returns[-50:-25])
                
                vol_change = (recent_vol['volatility'] - previous_vol['volatility']) / previous_vol['volatility']
                
                if vol_change > 0.2:
                    trend = 'increasing'
                    trend_description = '변동성 증가'
                elif vol_change < -0.2:
                    trend = 'decreasing'
                    trend_description = '변동성 감소'
                else:
                    trend = 'stable'
                    trend_description = '변동성 안정'
            else:
                vol_change = 0.0
                trend = 'unknown'
                trend_description = '데이터 부족'
            
            result = {
                'current_volatility': current_volatility,
                'regime': regime,
                'regime_description': regime_description,
                'trend': trend,
                'trend_description': trend_description,
                'volatility_change': vol_change,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                'sample_size': len(returns)
            }
            
            logger.info(f"변동성 체제 감지 완료: {regime} ({regime_description})")
            
            return result
            
        except Exception as e:
            logger.error(f"변동성 체제 감지 실패: {e}")
            return {'error': str(e)}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("변동성 계산기 테스트")
        print("=" * 60)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 252)
        test_high = np.random.uniform(100, 110, 252)
        test_low = np.random.uniform(90, 100, 252)
        
        # 변동성 계산기 생성
        vol_calculator = VolatilityCalculator()
        
        # 종합 변동성 계산
        result = vol_calculator.calculate_comprehensive_volatility(
            test_returns, test_high, test_low
        )
        
        print(f"테스트 결과:")
        print(f"  평균 변동성: {result['average_volatility']:.4f} ({result['average_volatility']*100:.2f}%)")
        print(f"  중간값 변동성: {result['median_volatility']:.4f} ({result['median_volatility']*100:.2f}%)")
        
        # 개별 방법별 결과
        print(f"\n개별 방법별 결과:")
        for method, method_result in result['individual_results'].items():
            if 'error' not in method_result:
                print(f"  {method}: {method_result['volatility']:.4f} ({method_result['volatility']*100:.2f}%)")
        
        # 변동성 예측
        forecast = vol_calculator.calculate_volatility_forecast(test_returns, forecast_horizon=5)
        if 'error' not in forecast:
            print(f"\n변동성 예측 (5일):")
            print(f"  현재 변동성: {forecast['current_volatility']:.4f}")
            print(f"  예측 변동성: {forecast['forecast_volatility']:.4f}")
            print(f"  예측 구간: [{forecast['lower_bound']:.4f}, {forecast['upper_bound']:.4f}]")
        
        # 변동성 체제 감지
        regime = vol_calculator.detect_volatility_regime(test_returns)
        if 'error' not in regime:
            print(f"\n변동성 체제:")
            print(f"  체제: {regime['regime']} ({regime['regime_description']})")
            print(f"  트렌드: {regime['trend']} ({regime['trend_description']})")
        
        print(f"\n[성공] 변동성 계산기 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 변동성 계산기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
