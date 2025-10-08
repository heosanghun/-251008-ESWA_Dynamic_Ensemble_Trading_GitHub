"""
성과 지표 계산 및 분석 모듈
Performance Metrics Module

성과 지표 계산 및 분석
- 누적 수익률, CAGR, 샤프 비율, 최대 낙폭
- 승률, 수익 팩터, 변동성 분석
- 리스크 조정 수익률 지표
- 체제별 성과 비교 분석

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class PerformanceMetrics:
    """성과 지표 데이터 클래스"""
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    volatility: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    cvar_95: float


class PerformanceAnalyzer:
    """
    성과 분석기 메인 클래스
    
    다양한 성과 지표 계산 및 분석
    """
    
    def __init__(self, config: Dict):
        """
        성과 분석기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 성과 분석 설정
        self.analysis_config = config.get('performance_analysis', {})
        self.risk_free_rate = self.analysis_config.get('risk_free_rate', 0.02)  # 2%
        self.confidence_level = self.analysis_config.get('confidence_level', 0.95)  # 95%
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("성과 분석기 초기화 완료")
        self.logger.info(f"무위험 수익률: {self.risk_free_rate:.2%}")
        self.logger.info(f"신뢰 수준: {self.confidence_level:.1%}")
    
    def calculate_comprehensive_metrics(self, returns: List[float], 
                                      benchmark_returns: Optional[List[float]] = None) -> PerformanceMetrics:
        """
        종합 성과 지표 계산
        
        Args:
            returns: 수익률 리스트
            benchmark_returns: 벤치마크 수익률 (선택사항)
            
        Returns:
            성과 지표 객체
        """
        try:
            if not returns:
                return self._get_empty_metrics()
            
            returns_array = np.array(returns)
            
            # 기본 성과 지표
            total_return = self._calculate_total_return(returns_array)
            cagr = self._calculate_cagr(returns_array)
            volatility = self._calculate_volatility(returns_array)
            sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
            max_drawdown = self._calculate_max_drawdown(returns_array)
            win_rate = self._calculate_win_rate(returns_array)
            profit_factor = self._calculate_profit_factor(returns_array)
            
            # 고급 성과 지표
            sortino_ratio = self._calculate_sortino_ratio(returns_array)
            calmar_ratio = self._calculate_calmar_ratio(cagr, max_drawdown)
            var_95 = self._calculate_var(returns_array, self.confidence_level)
            cvar_95 = self._calculate_cvar(returns_array, self.confidence_level)
            
            return PerformanceMetrics(
                total_return=total_return,
                cagr=cagr,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                volatility=volatility,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            self.logger.error(f"종합 성과 지표 계산 실패: {e}")
            return self._get_empty_metrics()
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """총 수익률 계산"""
        try:
            if len(returns) == 0:
                return 0.0
            return float(np.prod(1 + returns) - 1)
        except Exception as e:
            self.logger.error(f"총 수익률 계산 실패: {e}")
            return 0.0
    
    def _calculate_cagr(self, returns: np.ndarray) -> float:
        """연평균 복리 수익률 (CAGR) 계산"""
        try:
            total_return = self._calculate_total_return(returns)
            num_years = len(returns) / 252  # 252 거래일
            if num_years > 0:
                return (1 + total_return) ** (1 / num_years) - 1
            return 0.0
        except Exception as e:
            self.logger.error(f"CAGR 계산 실패: {e}")
            return 0.0
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """변동성 계산 (연율화)"""
        try:
            return np.std(returns) * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"변동성 계산 실패: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """샤프 비율 계산"""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            if np.std(excess_returns) > 0:
                return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return 0.0
        except Exception as e:
            self.logger.error(f"샤프 비율 계산 실패: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            return np.min(drawdowns)
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """승률 계산"""
        try:
            positive_returns = np.sum(returns > 0)
            return positive_returns / len(returns) if len(returns) > 0 else 0.0
        except Exception as e:
            self.logger.error(f"승률 계산 실패: {e}")
            return 0.0
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """수익 팩터 계산"""
        try:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) > 0:
                return abs(np.sum(positive_returns) / np.sum(negative_returns))
            elif len(positive_returns) > 0:
                return float('inf')
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"수익 팩터 계산 실패: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Sortino 비율 계산"""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            return 0.0
        except Exception as e:
            self.logger.error(f"Sortino 비율 계산 실패: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """Calmar 비율 계산"""
        try:
            if abs(max_drawdown) > 0:
                return cagr / abs(max_drawdown)
            return 0.0
        except Exception as e:
            self.logger.error(f"Calmar 비율 계산 실패: {e}")
            return 0.0
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Value at Risk (VaR) 계산"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            self.logger.error(f"VaR 계산 실패: {e}")
            return 0.0
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Conditional Value at Risk (CVaR) 계산"""
        try:
            var = self._calculate_var(returns, confidence_level)
            tail_returns = returns[returns <= var]
            return np.mean(tail_returns) if len(tail_returns) > 0 else var
        except Exception as e:
            self.logger.error(f"CVaR 계산 실패: {e}")
            return 0.0
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """빈 성과 지표 반환"""
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            volatility=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    def analyze_regime_performance(self, regime_returns: Dict[str, List[float]]) -> Dict[str, PerformanceMetrics]:
        """
        체제별 성과 분석
        
        Args:
            regime_returns: 체제별 수익률 딕셔너리
            
        Returns:
            체제별 성과 지표
        """
        try:
            regime_metrics = {}
            
            for regime, returns in regime_returns.items():
                if returns:
                    metrics = self.calculate_comprehensive_metrics(returns)
                    regime_metrics[regime] = metrics
                else:
                    regime_metrics[regime] = self._get_empty_metrics()
            
            return regime_metrics
            
        except Exception as e:
            self.logger.error(f"체제별 성과 분석 실패: {e}")
            return {}
    
    def compare_with_benchmark(self, strategy_returns: List[float], 
                              benchmark_returns: List[float]) -> Dict[str, float]:
        """
        벤치마크 대비 성과 비교
        
        Args:
            strategy_returns: 전략 수익률
            benchmark_returns: 벤치마크 수익률
            
        Returns:
            비교 지표 딕셔너리
        """
        try:
            if not strategy_returns or not benchmark_returns:
                return {}
            
            strategy_array = np.array(strategy_returns)
            benchmark_array = np.array(benchmark_returns)
            
            # 길이 맞추기
            min_length = min(len(strategy_array), len(benchmark_array))
            strategy_array = strategy_array[:min_length]
            benchmark_array = benchmark_array[:min_length]
            
            # 기본 성과 지표
            strategy_metrics = self.calculate_comprehensive_metrics(strategy_array.tolist())
            benchmark_metrics = self.calculate_comprehensive_metrics(benchmark_array.tolist())
            
            # 초과 수익률
            excess_returns = strategy_array - benchmark_array
            excess_metrics = self.calculate_comprehensive_metrics(excess_returns.tolist())
            
            # 정보 비율
            information_ratio = excess_metrics.sharpe_ratio
            
            # 베타 계산
            covariance = np.cov(strategy_array, benchmark_array)[0, 1]
            benchmark_variance = np.var(benchmark_array)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            
            # 알파 계산
            alpha = strategy_metrics.cagr - (self.risk_free_rate + beta * (benchmark_metrics.cagr - self.risk_free_rate))
            
            # 추적 오차
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            
            return {
                'strategy_total_return': strategy_metrics.total_return,
                'benchmark_total_return': benchmark_metrics.total_return,
                'excess_return': strategy_metrics.total_return - benchmark_metrics.total_return,
                'strategy_sharpe': strategy_metrics.sharpe_ratio,
                'benchmark_sharpe': benchmark_metrics.sharpe_ratio,
                'information_ratio': information_ratio,
                'alpha': alpha,
                'beta': beta,
                'tracking_error': tracking_error,
                'strategy_max_drawdown': strategy_metrics.max_drawdown,
                'benchmark_max_drawdown': benchmark_metrics.max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"벤치마크 비교 분석 실패: {e}")
            return {}
    
    def calculate_rolling_metrics(self, returns: List[float], window_size: int = 252) -> pd.DataFrame:
        """
        롤링 성과 지표 계산
        
        Args:
            returns: 수익률 리스트
            window_size: 롤링 윈도우 크기
            
        Returns:
            롤링 성과 지표 DataFrame
        """
        try:
            if len(returns) < window_size:
                return pd.DataFrame()
            
            returns_array = np.array(returns)
            rolling_metrics = []
            
            for i in range(window_size, len(returns_array)):
                window_returns = returns_array[i-window_size:i]
                metrics = self.calculate_comprehensive_metrics(window_returns.tolist())
                
                rolling_metrics.append({
                    'period': i,
                    'total_return': metrics.total_return,
                    'cagr': metrics.cagr,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'volatility': metrics.volatility,
                    'win_rate': metrics.win_rate
                })
            
            return pd.DataFrame(rolling_metrics)
            
        except Exception as e:
            self.logger.error(f"롤링 성과 지표 계산 실패: {e}")
            return pd.DataFrame()
    
    def generate_performance_report(self, returns: List[float], 
                                  regime_returns: Optional[Dict[str, List[float]]] = None,
                                  benchmark_returns: Optional[List[float]] = None) -> Dict[str, Union[str, float, Dict]]:
        """
        성과 보고서 생성
        
        Args:
            returns: 수익률 리스트
            regime_returns: 체제별 수익률 (선택사항)
            benchmark_returns: 벤치마크 수익률 (선택사항)
            
        Returns:
            성과 보고서
        """
        try:
            # 기본 성과 지표
            metrics = self.calculate_comprehensive_metrics(returns)
            
            # 체제별 성과 분석
            regime_analysis = {}
            if regime_returns:
                regime_analysis = self.analyze_regime_performance(regime_returns)
            
            # 벤치마크 비교
            benchmark_comparison = {}
            if benchmark_returns:
                benchmark_comparison = self.compare_with_benchmark(returns, benchmark_returns)
            
            # 롤링 성과 지표
            rolling_metrics = self.calculate_rolling_metrics(returns)
            
            # 보고서 생성
            report = {
                'summary': {
                    'total_return': f"{metrics.total_return:.2%}",
                    'cagr': f"{metrics.cagr:.2%}",
                    'sharpe_ratio': f"{metrics.sharpe_ratio:.3f}",
                    'max_drawdown': f"{metrics.max_drawdown:.2%}",
                    'volatility': f"{metrics.volatility:.2%}",
                    'win_rate': f"{metrics.win_rate:.2%}",
                    'profit_factor': f"{metrics.profit_factor:.2f}",
                    'sortino_ratio': f"{metrics.sortino_ratio:.3f}",
                    'calmar_ratio': f"{metrics.calmar_ratio:.3f}"
                },
                'detailed_metrics': {
                    'total_return': metrics.total_return,
                    'cagr': metrics.cagr,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'volatility': metrics.volatility,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sortino_ratio': metrics.sortino_ratio,
                    'calmar_ratio': metrics.calmar_ratio,
                    'var_95': metrics.var_95,
                    'cvar_95': metrics.cvar_95
                },
                'regime_analysis': regime_analysis,
                'benchmark_comparison': benchmark_comparison,
                'rolling_metrics': rolling_metrics.to_dict('records') if not rolling_metrics.empty else [],
                'data_points': len(returns)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"성과 보고서 생성 실패: {e}")
            return {}
    
    def get_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        리스크 지표 계산
        
        Args:
            returns: 수익률 리스트
            
        Returns:
            리스크 지표 딕셔너리
        """
        try:
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            # 기본 리스크 지표
            volatility = self._calculate_volatility(returns_array)
            max_drawdown = self._calculate_max_drawdown(returns_array)
            var_95 = self._calculate_var(returns_array, 0.95)
            cvar_95 = self._calculate_cvar(returns_array, 0.95)
            
            # 추가 리스크 지표
            downside_deviation = np.std(returns_array[returns_array < 0]) * np.sqrt(252)
            upside_deviation = np.std(returns_array[returns_array > 0]) * np.sqrt(252)
            
            # 꼬리 위험
            skewness = self._calculate_skewness(returns_array)
            kurtosis = self._calculate_kurtosis(returns_array)
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'downside_deviation': downside_deviation,
                'upside_deviation': upside_deviation,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
        except Exception as e:
            self.logger.error(f"리스크 지표 계산 실패: {e}")
            return {}
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """왜도 계산"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                return np.mean(((returns - mean_return) / std_return) ** 3)
            return 0.0
        except Exception as e:
            self.logger.error(f"왜도 계산 실패: {e}")
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """첨도 계산"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                return np.mean(((returns - mean_return) / std_return) ** 4) - 3
            return 0.0
        except Exception as e:
            self.logger.error(f"첨도 계산 실패: {e}")
            return 0.0


# 편의 함수들
def create_performance_analyzer(config: Dict) -> PerformanceAnalyzer:
    """성과 분석기 생성 편의 함수"""
    return PerformanceAnalyzer(config)


def calculate_quick_metrics(returns: List[float]) -> Dict[str, float]:
    """빠른 성과 지표 계산 편의 함수"""
    analyzer = PerformanceAnalyzer({})
    metrics = analyzer.calculate_comprehensive_metrics(returns)
    
    return {
        'total_return': metrics.total_return,
        'cagr': metrics.cagr,
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown': metrics.max_drawdown,
        'volatility': metrics.volatility,
        'win_rate': metrics.win_rate
    }
