"""
VaR (Value at Risk) 계산기
Value at Risk Calculator

Historical, Parametric, Monte Carlo 방법을 통한 VaR 계산
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VaRCalculator:
    """VaR 계산기 클래스"""
    
    def __init__(self, confidence_level: float = 0.95, lookback_period: int = 252):
        """
        VaR 계산기 초기화
        
        Args:
            confidence_level: 신뢰수준 (기본값: 0.95)
            lookback_period: 백테스팅 기간 (기본값: 252일)
        """
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.alpha = 1 - confidence_level
        
        logger.info(f"VaR 계산기 초기화 완료")
        logger.info(f"신뢰수준: {confidence_level:.1%}")
        logger.info(f"백테스팅 기간: {lookback_period}일")
    
    def calculate_historical_var(self, returns: np.ndarray, portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Historical VaR 계산
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            
        Returns:
            VaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'var': 0.0, 'var_percentage': 0.0, 'method': 'historical'}
            
            # Historical VaR 계산
            var_percentage = np.percentile(returns, self.alpha * 100)
            var_amount = portfolio_value * abs(var_percentage)
            
            result = {
                'var': var_amount,
                'var_percentage': abs(var_percentage),
                'method': 'historical',
                'confidence_level': self.confidence_level,
                'sample_size': len(returns)
            }
            
            logger.info(f"Historical VaR 계산 완료: {var_percentage:.4f} ({var_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Historical VaR 계산 실패: {e}")
            return {'var': 0.0, 'var_percentage': 0.0, 'method': 'historical', 'error': str(e)}
    
    def calculate_parametric_var(self, returns: np.ndarray, portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Parametric VaR 계산 (정규분포 가정)
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            
        Returns:
            VaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'var': 0.0, 'var_percentage': 0.0, 'method': 'parametric'}
            
            # 평균과 표준편차 계산
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 정규분포 가정하에 VaR 계산
            z_score = stats.norm.ppf(self.alpha)
            var_percentage = abs(mean_return + z_score * std_return)
            var_amount = portfolio_value * var_percentage
            
            result = {
                'var': var_amount,
                'var_percentage': var_percentage,
                'method': 'parametric',
                'confidence_level': self.confidence_level,
                'mean_return': mean_return,
                'std_return': std_return,
                'z_score': z_score
            }
            
            logger.info(f"Parametric VaR 계산 완료: {var_percentage:.4f} ({var_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Parametric VaR 계산 실패: {e}")
            return {'var': 0.0, 'var_percentage': 0.0, 'method': 'parametric', 'error': str(e)}
    
    def calculate_monte_carlo_var(self, returns: np.ndarray, portfolio_value: float = 1.0, 
                                n_simulations: int = 10000) -> Dict[str, float]:
        """
        Monte Carlo VaR 계산
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            n_simulations: 시뮬레이션 횟수
            
        Returns:
            VaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'var': 0.0, 'var_percentage': 0.0, 'method': 'monte_carlo'}
            
            # 수익률의 평균과 표준편차 계산
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Monte Carlo 시뮬레이션
            np.random.seed(42)  # 재현 가능한 결과를 위해
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # VaR 계산
            var_percentage = np.percentile(simulated_returns, self.alpha * 100)
            var_amount = portfolio_value * abs(var_percentage)
            
            result = {
                'var': var_amount,
                'var_percentage': abs(var_percentage),
                'method': 'monte_carlo',
                'confidence_level': self.confidence_level,
                'n_simulations': n_simulations,
                'mean_return': mean_return,
                'std_return': std_return
            }
            
            logger.info(f"Monte Carlo VaR 계산 완료: {var_percentage:.4f} ({var_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Monte Carlo VaR 계산 실패: {e}")
            return {'var': 0.0, 'var_percentage': 0.0, 'method': 'monte_carlo', 'error': str(e)}
    
    def calculate_ewma_var(self, returns: np.ndarray, portfolio_value: float = 1.0, 
                          lambda_param: float = 0.94) -> Dict[str, float]:
        """
        EWMA (Exponentially Weighted Moving Average) VaR 계산
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            lambda_param: EWMA 감쇠 인자
            
        Returns:
            VaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'var': 0.0, 'var_percentage': 0.0, 'method': 'ewma'}
            
            # EWMA 가중치 계산
            weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))])
            weights = weights[::-1]  # 최신 데이터에 높은 가중치
            
            # 가중 평균과 가중 표준편차 계산
            mean_return = np.average(returns, weights=weights)
            
            # 가중 분산 계산
            weighted_variance = np.average((returns - mean_return) ** 2, weights=weights)
            std_return = np.sqrt(weighted_variance)
            
            # VaR 계산
            z_score = stats.norm.ppf(self.alpha)
            var_percentage = abs(mean_return + z_score * std_return)
            var_amount = portfolio_value * var_percentage
            
            result = {
                'var': var_amount,
                'var_percentage': var_percentage,
                'method': 'ewma',
                'confidence_level': self.confidence_level,
                'lambda_param': lambda_param,
                'mean_return': mean_return,
                'std_return': std_return,
                'z_score': z_score
            }
            
            logger.info(f"EWMA VaR 계산 완료: {var_percentage:.4f} ({var_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"EWMA VaR 계산 실패: {e}")
            return {'var': 0.0, 'var_percentage': 0.0, 'method': 'ewma', 'error': str(e)}
    
    def calculate_comprehensive_var(self, returns: np.ndarray, portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        종합 VaR 계산 (모든 방법 사용)
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            
        Returns:
            종합 VaR 결과
        """
        try:
            logger.info("종합 VaR 계산 시작")
            
            # 각 방법별 VaR 계산
            historical_var = self.calculate_historical_var(returns, portfolio_value)
            parametric_var = self.calculate_parametric_var(returns, portfolio_value)
            monte_carlo_var = self.calculate_monte_carlo_var(returns, portfolio_value)
            ewma_var = self.calculate_ewma_var(returns, portfolio_value)
            
            # 결과 통합
            var_results = {
                'historical': historical_var,
                'parametric': parametric_var,
                'monte_carlo': monte_carlo_var,
                'ewma': ewma_var
            }
            
            # 평균 VaR 계산
            var_percentages = [result['var_percentage'] for result in var_results.values() 
                             if 'var_percentage' in result and result['var_percentage'] > 0]
            
            if var_percentages:
                average_var_percentage = np.mean(var_percentages)
                average_var_amount = portfolio_value * average_var_percentage
            else:
                average_var_percentage = 0.0
                average_var_amount = 0.0
            
            comprehensive_result = {
                'individual_results': var_results,
                'average_var': average_var_amount,
                'average_var_percentage': average_var_percentage,
                'confidence_level': self.confidence_level,
                'portfolio_value': portfolio_value,
                'sample_size': len(returns)
            }
            
            logger.info(f"종합 VaR 계산 완료")
            logger.info(f"평균 VaR: {average_var_percentage:.4f} ({average_var_percentage*100:.2f}%)")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"종합 VaR 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_portfolio_var(self, asset_returns: Dict[str, np.ndarray], 
                              portfolio_weights: Dict[str, float], 
                              portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        포트폴리오 VaR 계산
        
        Args:
            asset_returns: 자산별 수익률 딕셔너리
            portfolio_weights: 포트폴리오 가중치
            portfolio_value: 포트폴리오 가치
            
        Returns:
            포트폴리오 VaR 결과
        """
        try:
            logger.info("포트폴리오 VaR 계산 시작")
            
            # 포트폴리오 수익률 계산
            assets = list(asset_returns.keys())
            min_length = min(len(asset_returns[asset]) for asset in assets)
            
            # 동일한 길이로 자르기
            aligned_returns = {asset: asset_returns[asset][:min_length] for asset in assets}
            
            # 가중 포트폴리오 수익률 계산
            portfolio_returns = np.zeros(min_length)
            for asset in assets:
                weight = portfolio_weights.get(asset, 0.0)
                portfolio_returns += weight * aligned_returns[asset]
            
            # 포트폴리오 VaR 계산
            portfolio_var = self.calculate_comprehensive_var(portfolio_returns, portfolio_value)
            
            # 개별 자산 VaR도 계산
            individual_vars = {}
            for asset in assets:
                individual_vars[asset] = self.calculate_comprehensive_var(
                    aligned_returns[asset], portfolio_value * portfolio_weights.get(asset, 0.0)
                )
            
            result = {
                'portfolio_var': portfolio_var,
                'individual_vars': individual_vars,
                'portfolio_weights': portfolio_weights,
                'portfolio_value': portfolio_value,
                'assets': assets
            }
            
            logger.info(f"포트폴리오 VaR 계산 완료")
            
            return result
            
        except Exception as e:
            logger.error(f"포트폴리오 VaR 계산 실패: {e}")
            return {'error': str(e)}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("VaR 계산기 테스트")
        print("=" * 60)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 252)  # 1년 데이터
        
        # VaR 계산기 생성
        var_calculator = VaRCalculator(confidence_level=0.95)
        
        # 종합 VaR 계산
        result = var_calculator.calculate_comprehensive_var(test_returns, portfolio_value=10000)
        
        print(f"테스트 결과:")
        print(f"  평균 VaR: {result['average_var']:.2f}")
        print(f"  VaR 비율: {result['average_var_percentage']:.4f} ({result['average_var_percentage']*100:.2f}%)")
        
        # 개별 방법별 결과
        print(f"\n개별 방법별 결과:")
        for method, method_result in result['individual_results'].items():
            print(f"  {method}: {method_result['var_percentage']:.4f} ({method_result['var_percentage']*100:.2f}%)")
        
        print(f"\n[성공] VaR 계산기 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] VaR 계산기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
