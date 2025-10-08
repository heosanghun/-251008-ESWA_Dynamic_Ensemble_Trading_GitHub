"""
고급 성과 최적화 시스템
Advanced Performance Optimization System

논문 목표 100% 달성을 위한 고급 성과 최적화
- 동적 하이퍼파라미터 튜닝
- 적응형 리스크 관리
- 실시간 성과 모니터링
- 자동 최적화 알고리즘
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
import optuna
import joblib
import os

class OptimizationStatus(Enum):
    """최적화 상태"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGED = "converged"
    FAILED = "failed"
    COMPLETED = "completed"

@dataclass
class OptimizationResult:
    """최적화 결과"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    recommendations: List[str]
    status: OptimizationStatus

@dataclass
class PerformanceTargets:
    """성과 목표"""
    target_sharpe_ratio: float = 1.89
    target_max_drawdown: float = 0.162
    target_bear_market_return: float = 0.079
    target_volatility: float = 0.15
    target_win_rate: float = 0.6
    target_profit_factor: float = 1.5

class AdvancedPerformanceOptimizer:
    """
    고급 성과 최적화 시스템
    
    논문 목표 100% 달성을 위한 고급 성과 최적화 기능
    """
    
    def __init__(self, config: Dict):
        """
        고급 성과 최적화자 초기화
        
        Args:
            config: 최적화 설정 딕셔너리
        """
        self.config = config
        self.optimization_config = config.get('hyperparameter_tuning', {})
        
        # 성과 목표 설정
        self.targets = PerformanceTargets()
        
        # 최적화 히스토리
        self.optimization_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # 최적화 상태
        self.status = OptimizationStatus.INITIALIZING
        
        # 베이지안 최적화 설정
        self.study = None
        self.best_params = None
        self.best_score = -np.inf
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("고급 성과 최적화자 초기화 완료")
        self.logger.info(f"목표 샤프 비율: {self.targets.target_sharpe_ratio}")
        self.logger.info(f"목표 최대 낙폭: {self.targets.target_max_drawdown:.1%}")
        self.logger.info(f"목표 베어마켓 수익률: {self.targets.target_bear_market_return:.1%}")
    
    def optimize_system_performance(self, data: Dict[str, pd.DataFrame], 
                                  initial_config: Dict) -> OptimizationResult:
        """
        시스템 성과 최적화
        
        Args:
            data: 시장 데이터
            initial_config: 초기 설정
            
        Returns:
            최적화 결과
        """
        try:
            self.logger.info("시스템 성과 최적화 시작")
            self.status = OptimizationStatus.RUNNING
            
            # 1. 베이지안 최적화 설정
            self._setup_bayesian_optimization()
            
            # 2. 최적화 실행
            optimization_result = self._run_bayesian_optimization(data, initial_config)
            
            # 3. 결과 분석 및 권장사항 생성
            recommendations = self._generate_optimization_recommendations(optimization_result)
            
            # 4. 최적화 결과 정리
            result = OptimizationResult(
                best_params=optimization_result['best_params'],
                best_score=optimization_result['best_score'],
                optimization_history=list(self.optimization_history),
                convergence_info=optimization_result['convergence_info'],
                recommendations=recommendations,
                status=OptimizationStatus.COMPLETED
            )
            
            self.status = OptimizationStatus.COMPLETED
            self.logger.info(f"시스템 성과 최적화 완료 - 최적 점수: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"시스템 성과 최적화 실패: {e}")
            self.status = OptimizationStatus.FAILED
            return OptimizationResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                convergence_info={},
                recommendations=["최적화 실패"],
                status=OptimizationStatus.FAILED
            )
    
    def _setup_bayesian_optimization(self):
        """베이지안 최적화 설정"""
        try:
            # Optuna 스터디 생성
            self.study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            self.logger.info("베이지안 최적화 설정 완료")
            
        except Exception as e:
            self.logger.error(f"베이지안 최적화 설정 실패: {e}")
            raise
    
    def _run_bayesian_optimization(self, data: Dict[str, pd.DataFrame], 
                                 initial_config: Dict) -> Dict[str, Any]:
        """베이지안 최적화 실행"""
        try:
            n_trials = self.optimization_config.get('n_iterations', 50)
            n_random_starts = self.optimization_config.get('n_random_starts', 10)
            
            self.logger.info(f"베이지안 최적화 실행 시작 - {n_trials}회 시도")
            
            # 최적화 함수 정의
            def objective(trial):
                # 하이퍼파라미터 제안
                params = self._suggest_hyperparameters(trial)
                
                # 성과 평가
                score = self._evaluate_performance(data, initial_config, params)
                
                # 히스토리 업데이트
                self.optimization_history.append({
                    'trial': trial.number,
                    'params': params,
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
                return score
            
            # 최적화 실행
            self.study.optimize(objective, n_trials=n_trials)
            
            # 최적화 결과 추출
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            # 수렴 정보 계산
            convergence_info = self._calculate_convergence_info()
            
            self.logger.info(f"베이지안 최적화 완료 - 최적 점수: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'convergence_info': convergence_info,
                'n_trials': n_trials,
                'study': self.study
            }
            
        except Exception as e:
            self.logger.error(f"베이지안 최적화 실행 실패: {e}")
            raise
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """하이퍼파라미터 제안"""
        try:
            # PPO 하이퍼파라미터
            ppo_params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.001, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192]),
                'n_epochs': trial.suggest_int('n_epochs', 5, 20),
                'gamma': trial.suggest_float('gamma', 0.99, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.01, log=True)
            }
            
            # 리스크 관리 하이퍼파라미터
            risk_params = {
                'max_position_size': trial.suggest_float('max_position_size', 0.05, 0.2),
                'max_drawdown_limit': trial.suggest_float('max_drawdown_limit', 0.05, 0.15),
                'volatility_target': trial.suggest_float('volatility_target', 0.08, 0.20),
                'confidence_threshold': trial.suggest_float('confidence_threshold', 0.6, 0.9)
            }
            
            # 앙상블 하이퍼파라미터
            ensemble_params = {
                'temperature': trial.suggest_float('temperature', 3, 15),
                'evaluation_period': trial.suggest_int('evaluation_period', 10, 30),
                'min_weight': trial.suggest_float('min_weight', 0.01, 0.1)
            }
            
            # 네트워크 하이퍼파라미터
            network_params = {
                'hidden_layers': trial.suggest_categorical('hidden_layers', [
                    [64, 32], [128, 64], [256, 128], [256, 128, 64]
                ]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
            }
            
            return {
                'ppo': ppo_params,
                'risk_management': risk_params,
                'ensemble': ensemble_params,
                'network': network_params
            }
            
        except Exception as e:
            self.logger.error(f"하이퍼파라미터 제안 실패: {e}")
            return {}
    
    def _evaluate_performance(self, data: Dict[str, pd.DataFrame], 
                            initial_config: Dict, params: Dict) -> float:
        """성과 평가"""
        try:
            # 설정 업데이트
            updated_config = self._update_config_with_params(initial_config, params)
            
            # 성과 시뮬레이션
            performance_scores = []
            
            for asset, df in data.items():
                if len(df) < 100:
                    continue
                
                # 거래 성과 시뮬레이션
                performance = self._simulate_trading_performance(df, updated_config)
                
                # 성과 점수 계산
                score = self._calculate_performance_score(performance)
                performance_scores.append(score)
            
            # 전체 성과 점수
            if performance_scores:
                overall_score = np.mean(performance_scores)
            else:
                overall_score = 0.0
            
            # 성과 히스토리 업데이트
            self.performance_history.append({
                'params': params,
                'score': overall_score,
                'timestamp': pd.Timestamp.now()
            })
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"성과 평가 실패: {e}")
            return 0.0
    
    def _update_config_with_params(self, initial_config: Dict, params: Dict) -> Dict:
        """설정을 하이퍼파라미터로 업데이트"""
        try:
            updated_config = initial_config.copy()
            
            # PPO 설정 업데이트
            if 'ppo' in params:
                ppo_config = updated_config.get('agents', {}).get('ppo', {})
                ppo_config.update(params['ppo'])
                updated_config['agents']['ppo'] = ppo_config
            
            # 리스크 관리 설정 업데이트
            if 'risk_management' in params:
                risk_config = updated_config.get('risk_management', {})
                risk_config.update(params['risk_management'])
                updated_config['risk_management'] = risk_config
            
            # 앙상블 설정 업데이트
            if 'ensemble' in params:
                ensemble_config = updated_config.get('ensemble', {}).get('dynamic_weights', {})
                ensemble_config.update(params['ensemble'])
                updated_config['ensemble']['dynamic_weights'] = ensemble_config
            
            # 네트워크 설정 업데이트
            if 'network' in params:
                network_config = updated_config.get('agents', {}).get('network', {})
                network_config.update(params['network'])
                updated_config['agents']['network'] = network_config
            
            return updated_config
            
        except Exception as e:
            self.logger.error(f"설정 업데이트 실패: {e}")
            return initial_config
    
    def _simulate_trading_performance(self, df: pd.DataFrame, config: Dict) -> Dict[str, float]:
        """거래 성과 시뮬레이션"""
        try:
            # 가격 데이터 추출
            prices = df['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # 설정에서 파라미터 추출
            max_position_size = config.get('risk_management', {}).get('max_position_size', 0.12)
            volatility_target = config.get('risk_management', {}).get('volatility_target', 0.12)
            
            # 변동성 계산
            volatility = np.std(returns) * np.sqrt(252)
            
            # 포지션 크기 조절 (변동성 기반)
            position_size = min(max_position_size, volatility_target / max(volatility, 0.01))
            
            # 수익률 시뮬레이션 (포지션 크기 적용)
            simulated_returns = returns * position_size
            
            # 성과 지표 계산
            total_return = np.prod(1 + simulated_returns) - 1
            sharpe_ratio = np.mean(simulated_returns) / np.std(simulated_returns) * np.sqrt(252) if np.std(simulated_returns) > 0 else 0
            
            # 최대 낙폭 계산
            cumulative_returns = np.cumprod(1 + simulated_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # 승률 계산
            win_rate = np.mean(simulated_returns > 0)
            
            # 수익 팩터 계산
            positive_returns = simulated_returns[simulated_returns > 0]
            negative_returns = simulated_returns[simulated_returns < 0]
            profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns)) if len(negative_returns) > 0 and np.sum(negative_returns) != 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'position_size': position_size
            }
            
        except Exception as e:
            self.logger.error(f"거래 성과 시뮬레이션 실패: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'position_size': 0.0
            }
    
    def _calculate_performance_score(self, performance: Dict[str, float]) -> float:
        """성과 점수 계산"""
        try:
            # 목표 대비 성과 점수 계산
            sharpe_score = min(performance['sharpe_ratio'] / self.targets.target_sharpe_ratio, 1.0)
            drawdown_score = min(self.targets.target_max_drawdown / abs(performance['max_drawdown']), 1.0) if performance['max_drawdown'] != 0 else 0
            return_score = min(performance['total_return'] / self.targets.target_bear_market_return, 1.0)
            volatility_score = min(self.targets.target_volatility / performance['volatility'], 1.0) if performance['volatility'] > 0 else 0
            win_rate_score = min(performance['win_rate'] / self.targets.target_win_rate, 1.0)
            profit_factor_score = min(performance['profit_factor'] / self.targets.target_profit_factor, 1.0)
            
            # 가중 평균 점수
            weights = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]  # 샤프 비율, 낙폭, 수익률, 변동성, 승률, 수익 팩터
            scores = [sharpe_score, drawdown_score, return_score, volatility_score, win_rate_score, profit_factor_score]
            
            overall_score = np.average(scores, weights=weights)
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"성과 점수 계산 실패: {e}")
            return 0.0
    
    def _calculate_convergence_info(self) -> Dict[str, Any]:
        """수렴 정보 계산"""
        try:
            if not self.optimization_history:
                return {}
            
            # 최적화 히스토리에서 점수 추출
            scores = [entry['score'] for entry in self.optimization_history]
            
            # 수렴 통계
            convergence_info = {
                'n_trials': len(scores),
                'best_score': max(scores),
                'worst_score': min(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'improvement_rate': (max(scores) - min(scores)) / abs(min(scores)) if min(scores) != 0 else 0,
                'convergence_rate': self._calculate_convergence_rate(scores),
                'stability': 1.0 - (np.std(scores[-10:]) / np.mean(scores[-10:])) if len(scores) >= 10 and np.mean(scores[-10:]) != 0 else 0
            }
            
            return convergence_info
            
        except Exception as e:
            self.logger.error(f"수렴 정보 계산 실패: {e}")
            return {}
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """수렴률 계산"""
        try:
            if len(scores) < 10:
                return 0.0
            
            # 최근 10개 시도의 개선률
            recent_scores = scores[-10:]
            improvement = (max(recent_scores) - min(recent_scores)) / abs(min(recent_scores)) if min(recent_scores) != 0 else 0
            
            return improvement
            
        except Exception as e:
            self.logger.error(f"수렴률 계산 실패: {e}")
            return 0.0
    
    def _generate_optimization_recommendations(self, optimization_result: Dict) -> List[str]:
        """최적화 권장사항 생성"""
        try:
            recommendations = []
            
            best_params = optimization_result['best_params']
            best_score = optimization_result['best_score']
            convergence_info = optimization_result['convergence_info']
            
            # 최적화 성과 기반 권장사항
            if best_score > 0.8:
                recommendations.append("최적화가 매우 성공적입니다. 현재 설정을 유지하세요.")
            elif best_score > 0.6:
                recommendations.append("최적화가 성공적입니다. 추가 개선을 위해 더 많은 시도를 고려하세요.")
            elif best_score > 0.4:
                recommendations.append("최적화가 부분적으로 성공했습니다. 하이퍼파라미터 범위를 조정하세요.")
            else:
                recommendations.append("최적화가 실패했습니다. 전략을 재검토하세요.")
            
            # 수렴 정보 기반 권장사항
            if convergence_info.get('convergence_rate', 0) < 0.01:
                recommendations.append("수렴이 완료되었습니다. 추가 최적화는 불필요합니다.")
            elif convergence_info.get('convergence_rate', 0) < 0.05:
                recommendations.append("수렴이 거의 완료되었습니다. 몇 번 더 시도해보세요.")
            else:
                recommendations.append("수렴이 아직 완료되지 않았습니다. 더 많은 시도가 필요합니다.")
            
            # 안정성 기반 권장사항
            stability = convergence_info.get('stability', 0)
            if stability > 0.9:
                recommendations.append("최적화 결과가 매우 안정적입니다.")
            elif stability > 0.7:
                recommendations.append("최적화 결과가 안정적입니다.")
            else:
                recommendations.append("최적화 결과가 불안정합니다. 더 많은 시도가 필요합니다.")
            
            # 특정 하이퍼파라미터 기반 권장사항
            if 'learning_rate' in best_params:
                lr = best_params['learning_rate']
                if lr < 0.0003:
                    recommendations.append("학습률이 낮습니다. 더 빠른 수렴을 위해 증가를 고려하세요.")
                elif lr > 0.0008:
                    recommendations.append("학습률이 높습니다. 안정성을 위해 감소를 고려하세요.")
            
            if 'max_position_size' in best_params:
                pos_size = best_params['max_position_size']
                if pos_size < 0.08:
                    recommendations.append("포지션 크기가 작습니다. 수익률 향상을 위해 증가를 고려하세요.")
                elif pos_size > 0.15:
                    recommendations.append("포지션 크기가 큽니다. 리스크 관리를 위해 감소를 고려하세요.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"최적화 권장사항 생성 실패: {e}")
            return ["최적화 권장사항을 생성할 수 없습니다."]
    
    def save_optimization_results(self, result: OptimizationResult, filepath: str):
        """최적화 결과 저장"""
        try:
            # 결과를 딕셔너리로 변환
            result_dict = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_params': result.best_params,
                'best_score': result.best_score,
                'optimization_history': [
                    {
                        'trial': entry['trial'],
                        'params': entry['params'],
                        'score': entry['score'],
                        'timestamp': entry['timestamp'].isoformat()
                    }
                    for entry in result.optimization_history
                ],
                'convergence_info': result.convergence_info,
                'recommendations': result.recommendations,
                'status': result.status.value
            }
            
            # JSON 파일로 저장
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"최적화 결과 저장 완료: {filepath}")
            
        except Exception as e:
            self.logger.error(f"최적화 결과 저장 실패: {e}")
    
    def load_optimization_results(self, filepath: str) -> Optional[OptimizationResult]:
        """최적화 결과 로드"""
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                result_dict = json.load(f)
            
            # OptimizationResult 객체 생성
            result = OptimizationResult(
                best_params=result_dict['best_params'],
                best_score=result_dict['best_score'],
                optimization_history=result_dict['optimization_history'],
                convergence_info=result_dict['convergence_info'],
                recommendations=result_dict['recommendations'],
                status=OptimizationStatus(result_dict['status'])
            )
            
            self.logger.info(f"최적화 결과 로드 완료: {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"최적화 결과 로드 실패: {e}")
            return None


# 편의 함수들
def create_advanced_performance_optimizer(config: Dict) -> AdvancedPerformanceOptimizer:
    """고급 성과 최적화자 생성 편의 함수"""
    return AdvancedPerformanceOptimizer(config)


def optimize_system_performance(data: Dict[str, pd.DataFrame], 
                              initial_config: Dict, 
                              config: Dict) -> OptimizationResult:
    """시스템 성과 최적화 편의 함수"""
    optimizer = create_advanced_performance_optimizer(config)
    return optimizer.optimize_system_performance(data, initial_config)
