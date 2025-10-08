"""
CVaR (Conditional Value at Risk) 계산기
Conditional Value at Risk Calculator

CVaR 계산 및 포지션 크기 조절을 위한 고급 리스크 관리
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CVaRCalculator:
    """CVaR 계산기 클래스"""
    
    def __init__(self, confidence_level: float = 0.95, lookback_period: int = 252):
        """
        CVaR 계산기 초기화
        
        Args:
            confidence_level: 신뢰수준 (기본값: 0.95)
            lookback_period: 백테스팅 기간 (기본값: 252일)
        """
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.alpha = 1 - confidence_level
        
        logger.info(f"CVaR 계산기 초기화 완료")
        logger.info(f"신뢰수준: {confidence_level:.1%}")
        logger.info(f"백테스팅 기간: {lookback_period}일")
    
    def calculate_historical_cvar(self, returns: np.ndarray, portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Historical CVaR 계산
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            
        Returns:
            CVaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'historical'}
            
            # VaR 임계값 계산
            var_threshold = np.percentile(returns, self.alpha * 100)
            
            # VaR 임계값 이하의 손실들만 선택
            tail_losses = returns[returns <= var_threshold]
            
            if len(tail_losses) == 0:
                cvar_percentage = abs(var_threshold)
            else:
                cvar_percentage = abs(np.mean(tail_losses))
            
            cvar_amount = portfolio_value * cvar_percentage
            
            result = {
                'cvar': cvar_amount,
                'cvar_percentage': cvar_percentage,
                'var_threshold': var_threshold,
                'method': 'historical',
                'confidence_level': self.confidence_level,
                'tail_losses_count': len(tail_losses),
                'sample_size': len(returns)
            }
            
            logger.info(f"Historical CVaR 계산 완료: {cvar_percentage:.4f} ({cvar_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Historical CVaR 계산 실패: {e}")
            return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'historical', 'error': str(e)}
    
    def calculate_parametric_cvar(self, returns: np.ndarray, portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Parametric CVaR 계산 (정규분포 가정)
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            
        Returns:
            CVaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'parametric'}
            
            # 평균과 표준편차 계산
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 정규분포 가정하에 CVaR 계산
            z_score = stats.norm.ppf(self.alpha)
            
            # 정규분포의 CVaR 공식
            phi_z = stats.norm.pdf(z_score)
            cvar_percentage = abs(mean_return - std_return * (phi_z / self.alpha))
            cvar_amount = portfolio_value * cvar_percentage
            
            result = {
                'cvar': cvar_amount,
                'cvar_percentage': cvar_percentage,
                'method': 'parametric',
                'confidence_level': self.confidence_level,
                'mean_return': mean_return,
                'std_return': std_return,
                'z_score': z_score,
                'phi_z': phi_z
            }
            
            logger.info(f"Parametric CVaR 계산 완료: {cvar_percentage:.4f} ({cvar_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Parametric CVaR 계산 실패: {e}")
            return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'parametric', 'error': str(e)}
    
    def calculate_monte_carlo_cvar(self, returns: np.ndarray, portfolio_value: float = 1.0, 
                                 n_simulations: int = 10000) -> Dict[str, float]:
        """
        Monte Carlo CVaR 계산
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            n_simulations: 시뮬레이션 횟수
            
        Returns:
            CVaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'monte_carlo'}
            
            # 수익률의 평균과 표준편차 계산
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Monte Carlo 시뮬레이션
            np.random.seed(42)
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # VaR 임계값 계산
            var_threshold = np.percentile(simulated_returns, self.alpha * 100)
            
            # VaR 임계값 이하의 손실들만 선택
            tail_losses = simulated_returns[simulated_returns <= var_threshold]
            
            if len(tail_losses) == 0:
                cvar_percentage = abs(var_threshold)
            else:
                cvar_percentage = abs(np.mean(tail_losses))
            
            cvar_amount = portfolio_value * cvar_percentage
            
            result = {
                'cvar': cvar_amount,
                'cvar_percentage': cvar_percentage,
                'var_threshold': var_threshold,
                'method': 'monte_carlo',
                'confidence_level': self.confidence_level,
                'n_simulations': n_simulations,
                'tail_losses_count': len(tail_losses),
                'mean_return': mean_return,
                'std_return': std_return
            }
            
            logger.info(f"Monte Carlo CVaR 계산 완료: {cvar_percentage:.4f} ({cvar_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Monte Carlo CVaR 계산 실패: {e}")
            return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'monte_carlo', 'error': str(e)}
    
    def calculate_ewma_cvar(self, returns: np.ndarray, portfolio_value: float = 1.0, 
                           lambda_param: float = 0.94) -> Dict[str, float]:
        """
        EWMA CVaR 계산
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            lambda_param: EWMA 감쇠 인자
            
        Returns:
            CVaR 결과 딕셔너리
        """
        try:
            if len(returns) == 0:
                return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'ewma'}
            
            # EWMA 가중치 계산
            weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))])
            weights = weights[::-1]
            
            # 가중 평균과 가중 표준편차 계산
            mean_return = np.average(returns, weights=weights)
            weighted_variance = np.average((returns - mean_return) ** 2, weights=weights)
            std_return = np.sqrt(weighted_variance)
            
            # 정규분포 가정하에 CVaR 계산
            z_score = stats.norm.ppf(self.alpha)
            phi_z = stats.norm.pdf(z_score)
            cvar_percentage = abs(mean_return - std_return * (phi_z / self.alpha))
            cvar_amount = portfolio_value * cvar_percentage
            
            result = {
                'cvar': cvar_amount,
                'cvar_percentage': cvar_percentage,
                'method': 'ewma',
                'confidence_level': self.confidence_level,
                'lambda_param': lambda_param,
                'mean_return': mean_return,
                'std_return': std_return,
                'z_score': z_score,
                'phi_z': phi_z
            }
            
            logger.info(f"EWMA CVaR 계산 완료: {cvar_percentage:.4f} ({cvar_percentage*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"EWMA CVaR 계산 실패: {e}")
            return {'cvar': 0.0, 'cvar_percentage': 0.0, 'method': 'ewma', 'error': str(e)}
    
    def calculate_comprehensive_cvar(self, returns: np.ndarray, portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        종합 CVaR 계산 (모든 방법 사용)
        
        Args:
            returns: 수익률 배열
            portfolio_value: 포트폴리오 가치
            
        Returns:
            종합 CVaR 결과
        """
        try:
            logger.info("종합 CVaR 계산 시작")
            
            # 각 방법별 CVaR 계산
            historical_cvar = self.calculate_historical_cvar(returns, portfolio_value)
            parametric_cvar = self.calculate_parametric_cvar(returns, portfolio_value)
            monte_carlo_cvar = self.calculate_monte_carlo_cvar(returns, portfolio_value)
            ewma_cvar = self.calculate_ewma_cvar(returns, portfolio_value)
            
            # 결과 통합
            cvar_results = {
                'historical': historical_cvar,
                'parametric': parametric_cvar,
                'monte_carlo': monte_carlo_cvar,
                'ewma': ewma_cvar
            }
            
            # 평균 CVaR 계산
            cvar_percentages = [result['cvar_percentage'] for result in cvar_results.values() 
                              if 'cvar_percentage' in result and result['cvar_percentage'] > 0]
            
            if cvar_percentages:
                average_cvar_percentage = np.mean(cvar_percentages)
                average_cvar_amount = portfolio_value * average_cvar_percentage
            else:
                average_cvar_percentage = 0.0
                average_cvar_amount = 0.0
            
            comprehensive_result = {
                'individual_results': cvar_results,
                'average_cvar': average_cvar_amount,
                'average_cvar_percentage': average_cvar_percentage,
                'confidence_level': self.confidence_level,
                'portfolio_value': portfolio_value,
                'sample_size': len(returns)
            }
            
            logger.info(f"종합 CVaR 계산 완료")
            logger.info(f"평균 CVaR: {average_cvar_percentage:.4f} ({average_cvar_percentage*100:.2f}%)")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"종합 CVaR 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_portfolio_cvar(self, asset_returns: Dict[str, np.ndarray], 
                               portfolio_weights: Dict[str, float], 
                               portfolio_value: float = 1.0) -> Dict[str, Any]:
        """
        포트폴리오 CVaR 계산
        
        Args:
            asset_returns: 자산별 수익률 딕셔너리
            portfolio_weights: 포트폴리오 가중치
            portfolio_value: 포트폴리오 가치
            
        Returns:
            포트폴리오 CVaR 결과
        """
        try:
            logger.info("포트폴리오 CVaR 계산 시작")
            
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
            
            # 포트폴리오 CVaR 계산
            portfolio_cvar = self.calculate_comprehensive_cvar(portfolio_returns, portfolio_value)
            
            # 개별 자산 CVaR도 계산
            individual_cvars = {}
            for asset in assets:
                individual_cvars[asset] = self.calculate_comprehensive_cvar(
                    aligned_returns[asset], portfolio_value * portfolio_weights.get(asset, 0.0)
                )
            
            result = {
                'portfolio_cvar': portfolio_cvar,
                'individual_cvars': individual_cvars,
                'portfolio_weights': portfolio_weights,
                'portfolio_value': portfolio_value,
                'assets': assets
            }
            
            logger.info(f"포트폴리오 CVaR 계산 완료")
            
            return result
            
        except Exception as e:
            logger.error(f"포트폴리오 CVaR 계산 실패: {e}")
            return {'error': str(e)}
    
    def calculate_risk_contribution(self, asset_returns: Dict[str, np.ndarray], 
                                  portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        자산별 리스크 기여도 계산
        
        Args:
            asset_returns: 자산별 수익률 딕셔너리
            portfolio_weights: 포트폴리오 가중치
            
        Returns:
            리스크 기여도 결과
        """
        try:
            logger.info("리스크 기여도 계산 시작")
            
            assets = list(asset_returns.keys())
            min_length = min(len(asset_returns[asset]) for asset in assets)
            
            # 동일한 길이로 자르기
            aligned_returns = {asset: asset_returns[asset][:min_length] for asset in assets}
            
            # 포트폴리오 수익률 계산
            portfolio_returns = np.zeros(min_length)
            for asset in assets:
                weight = portfolio_weights.get(asset, 0.0)
                portfolio_returns += weight * aligned_returns[asset]
            
            # 포트폴리오 CVaR 계산
            portfolio_cvar = self.calculate_comprehensive_cvar(portfolio_returns)
            portfolio_cvar_pct = portfolio_cvar['average_cvar_percentage']
            
            # 자산별 리스크 기여도 계산
            risk_contributions = {}
            total_contribution = 0.0
            
            for asset in assets:
                # 해당 자산의 가중치를 0.01 증가시켜서 CVaR 변화 측정
                original_weight = portfolio_weights.get(asset, 0.0)
                new_weights = portfolio_weights.copy()
                new_weights[asset] = original_weight + 0.01
                
                # 가중치 정규화
                total_weight = sum(new_weights.values())
                normalized_weights = {k: v/total_weight for k, v in new_weights.items()}
                
                # 새로운 포트폴리오 수익률 계산
                new_portfolio_returns = np.zeros(min_length)
                for a in assets:
                    weight = normalized_weights.get(a, 0.0)
                    new_portfolio_returns += weight * aligned_returns[a]
                
                # 새로운 CVaR 계산
                new_cvar = self.calculate_comprehensive_cvar(new_portfolio_returns)
                new_cvar_pct = new_cvar['average_cvar_percentage']
                
                # 리스크 기여도 계산
                risk_contribution = (new_cvar_pct - portfolio_cvar_pct) / 0.01
                risk_contributions[asset] = {
                    'risk_contribution': risk_contribution,
                    'weight': original_weight,
                    'contribution_percentage': risk_contribution * original_weight / portfolio_cvar_pct if portfolio_cvar_pct > 0 else 0
                }
                total_contribution += risk_contribution * original_weight
            
            result = {
                'portfolio_cvar_percentage': portfolio_cvar_pct,
                'risk_contributions': risk_contributions,
                'total_contribution': total_contribution,
                'portfolio_weights': portfolio_weights
            }
            
            logger.info(f"리스크 기여도 계산 완료")
            
            return result
            
        except Exception as e:
            logger.error(f"리스크 기여도 계산 실패: {e}")
            return {'error': str(e)}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("CVaR 계산기 테스트")
        print("=" * 60)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        test_returns = np.random.normal(0.001, 0.02, 252)
        
        # CVaR 계산기 생성
        cvar_calculator = CVaRCalculator(confidence_level=0.95)
        
        # 종합 CVaR 계산
        result = cvar_calculator.calculate_comprehensive_cvar(test_returns, portfolio_value=10000)
        
        print(f"테스트 결과:")
        print(f"  평균 CVaR: {result['average_cvar']:.2f}")
        print(f"  CVaR 비율: {result['average_cvar_percentage']:.4f} ({result['average_cvar_percentage']*100:.2f}%)")
        
        # 개별 방법별 결과
        print(f"\n개별 방법별 결과:")
        for method, method_result in result['individual_results'].items():
            print(f"  {method}: {method_result['cvar_percentage']:.4f} ({method_result['cvar_percentage']*100:.2f}%)")
        
        print(f"\n[성공] CVaR 계산기 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] CVaR 계산기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
