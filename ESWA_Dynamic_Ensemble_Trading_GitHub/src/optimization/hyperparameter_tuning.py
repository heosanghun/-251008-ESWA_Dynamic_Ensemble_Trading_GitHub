"""
하이퍼파라미터 튜닝 모듈
Hyperparameter Tuning Module

논문 파라미터 범위 내에서 최적의 하이퍼파라미터 탐색
- 그리드 서치 및 베이지안 최적화
- 교차 검증을 통한 최적 파라미터 선택
- 성과 개선 확인
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

# 하이퍼파라미터 튜닝 라이브러리
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False
    logging.warning("scikit-optimize가 설치되지 않았습니다. 베이지안 최적화를 사용할 수 없습니다.")


@dataclass
class HyperparameterSpace:
    """하이퍼파라미터 탐색 공간"""
    learning_rate: Tuple[float, float] = (0.0001, 0.001)
    batch_size: Tuple[int, int] = (128, 512)
    n_steps: Tuple[int, int] = (1024, 4096)
    n_epochs: Tuple[int, int] = (5, 15)
    gamma: Tuple[float, float] = (0.95, 0.99)
    gae_lambda: Tuple[float, float] = (0.9, 0.98)
    clip_range: Tuple[float, float] = (0.1, 0.3)
    ent_coef: Tuple[float, float] = (0.001, 0.1)
    confidence_threshold: Tuple[float, float] = (0.5, 0.8)
    temperature: Tuple[float, float] = (5, 15)
    evaluation_period: Tuple[int, int] = (20, 40)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float


class HyperparameterTuner:
    """
    하이퍼파라미터 튜닝 메인 클래스
    
    논문 파라미터 범위 내에서 최적의 하이퍼파라미터 탐색
    """
    
    def __init__(self, config: Dict):
        """
        하이퍼파라미터 튜너 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 튜닝 설정
        self.tuning_config = config.get('hyperparameter_tuning', {})
        self.optimization_method = self.tuning_config.get('method', 'grid_search')
        self.max_evaluations = self.tuning_config.get('max_evaluations', 50)
        self.cv_folds = self.tuning_config.get('cv_folds', 3)
        self.scoring_metric = self.tuning_config.get('scoring_metric', 'sharpe_ratio')
        
        # 하이퍼파라미터 탐색 공간
        self.param_space = HyperparameterSpace()
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("하이퍼파라미터 튜너 초기화 완료")
        self.logger.info(f"최적화 방법: {self.optimization_method}")
        self.logger.info(f"최대 평가 횟수: {self.max_evaluations}")
        self.logger.info(f"교차 검증 폴드: {self.cv_folds}")
        self.logger.info(f"평가 지표: {self.scoring_metric}")
    
    def optimize_system_parameters(self, data: pd.DataFrame, 
                                 system_factory: callable) -> OptimizationResult:
        """
        시스템 파라미터 최적화
        
        Args:
            data: 훈련 데이터
            system_factory: 시스템 생성 함수
            
        Returns:
            최적화 결과
        """
        try:
            self.logger.info("시스템 파라미터 최적화 시작")
            
            if self.optimization_method == 'grid_search':
                result = self._grid_search_optimization(data, system_factory)
            elif self.optimization_method == 'bayesian' and BAYESIAN_OPTIMIZATION_AVAILABLE:
                result = self._bayesian_optimization(data, system_factory)
            else:
                self.logger.warning(f"지원하지 않는 최적화 방법: {self.optimization_method}")
                result = self._grid_search_optimization(data, system_factory)
            
            self.logger.info("시스템 파라미터 최적화 완료")
            self.logger.info(f"최적 점수: {result.best_score:.4f}")
            self.logger.info(f"총 평가 횟수: {result.total_evaluations}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"시스템 파라미터 최적화 실패: {e}")
            raise
    
    def _grid_search_optimization(self, data: pd.DataFrame, 
                                system_factory: callable) -> OptimizationResult:
        """그리드 서치 최적화"""
        try:
            self.logger.info("그리드 서치 최적화 시작")
            
            # 파라미터 그리드 생성
            param_grid = self._create_parameter_grid()
            
            best_score = -np.inf
            best_params = None
            optimization_history = []
            
            total_combinations = len(param_grid)
            self.logger.info(f"총 파라미터 조합: {total_combinations}개")
            
            for i, params in enumerate(param_grid):
                if i >= self.max_evaluations:
                    break
                
                self.logger.info(f"평가 {i+1}/{min(total_combinations, self.max_evaluations)}: {params}")
                
                # 교차 검증으로 성과 평가
                score = self._evaluate_parameters(data, params, system_factory)
                
                optimization_history.append({
                    'iteration': i + 1,
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    self.logger.info(f"새로운 최적 점수: {best_score:.4f}")
            
            return OptimizationResult(
                best_params=best_params or {},
                best_score=best_score,
                optimization_history=optimization_history,
                total_evaluations=len(optimization_history),
                optimization_time=0.0  # 추후 구현
            )
            
        except Exception as e:
            self.logger.error(f"그리드 서치 최적화 실패: {e}")
            raise
    
    def _bayesian_optimization(self, data: pd.DataFrame, 
                             system_factory: callable) -> OptimizationResult:
        """베이지안 최적화"""
        try:
            if not BAYESIAN_OPTIMIZATION_AVAILABLE:
                raise ImportError("scikit-optimize가 설치되지 않았습니다.")
            
            self.logger.info("베이지안 최적화 시작")
            
            # 탐색 공간 정의
            search_space = [
                Real(self.param_space.learning_rate[0], self.param_space.learning_rate[1], name='learning_rate'),
                Integer(self.param_space.batch_size[0], self.param_space.batch_size[1], name='batch_size'),
                Integer(self.param_space.n_steps[0], self.param_space.n_steps[1], name='n_steps'),
                Integer(self.param_space.n_epochs[0], self.param_space.n_epochs[1], name='n_epochs'),
                Real(self.param_space.gamma[0], self.param_space.gamma[1], name='gamma'),
                Real(self.param_space.gae_lambda[0], self.param_space.gae_lambda[1], name='gae_lambda'),
                Real(self.param_space.clip_range[0], self.param_space.clip_range[1], name='clip_range'),
                Real(self.param_space.ent_coef[0], self.param_space.ent_coef[1], name='ent_coef'),
                Real(self.param_space.confidence_threshold[0], self.param_space.confidence_threshold[1], name='confidence_threshold'),
                Real(self.param_space.temperature[0], self.param_space.temperature[1], name='temperature'),
                Integer(self.param_space.evaluation_period[0], self.param_space.evaluation_period[1], name='evaluation_period')
            ]
            
            optimization_history = []
            
            @use_named_args(search_space)
            def objective(**params):
                """목적 함수"""
                score = self._evaluate_parameters(data, params, system_factory)
                optimization_history.append({
                    'iteration': len(optimization_history) + 1,
                    'params': params.copy(),
                    'score': score
                })
                return -score  # 최대화를 위해 음수 반환
            
            # 베이지안 최적화 실행
            result = gp_minimize(
                func=objective,
                dimensions=search_space,
                n_calls=self.max_evaluations,
                random_state=42
            )
            
            # 최적 파라미터 추출
            best_params = {
                'learning_rate': result.x[0],
                'batch_size': int(result.x[1]),
                'n_steps': int(result.x[2]),
                'n_epochs': int(result.x[3]),
                'gamma': result.x[4],
                'gae_lambda': result.x[5],
                'clip_range': result.x[6],
                'ent_coef': result.x[7],
                'confidence_threshold': result.x[8],
                'temperature': result.x[9],
                'evaluation_period': int(result.x[10])
            }
            
            return OptimizationResult(
                best_params=best_params,
                best_score=-result.fun,  # 음수를 다시 양수로
                optimization_history=optimization_history,
                total_evaluations=len(optimization_history),
                optimization_time=0.0  # 추후 구현
            )
            
        except Exception as e:
            self.logger.error(f"베이지안 최적화 실패: {e}")
            raise
    
    def _create_parameter_grid(self) -> List[Dict[str, Any]]:
        """파라미터 그리드 생성"""
        try:
            # 논문 파라미터 범위 내에서 그리드 생성
            param_grid = {
                'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],
                'batch_size': [128, 256, 512],
                'n_steps': [1024, 2048, 4096],
                'n_epochs': [5, 10, 15],
                'gamma': [0.95, 0.97, 0.99],
                'gae_lambda': [0.9, 0.95, 0.98],
                'clip_range': [0.1, 0.2, 0.3],
                'ent_coef': [0.001, 0.01, 0.1],
                'confidence_threshold': [0.5, 0.6, 0.7, 0.8],
                'temperature': [5, 10, 15],
                'evaluation_period': [20, 30, 40]
            }
            
            # 그리드 조합 생성
            grid_combinations = []
            for combination in product(*param_grid.values()):
                params = dict(zip(param_grid.keys(), combination))
                grid_combinations.append(params)
            
            return grid_combinations
            
        except Exception as e:
            self.logger.error(f"파라미터 그리드 생성 실패: {e}")
            return []
    
    def _evaluate_parameters(self, data: pd.DataFrame, params: Dict[str, Any], 
                           system_factory: callable) -> float:
        """파라미터 성과 평가"""
        try:
            # 파라미터로 시스템 생성
            system = system_factory(params)
            
            # 교차 검증으로 성과 평가
            scores = []
            
            # 간단한 시계열 교차 검증
            data_length = len(data)
            fold_size = data_length // self.cv_folds
            
            for fold in range(self.cv_folds):
                # 훈련/검증 데이터 분할
                train_end = (fold + 1) * fold_size
                val_start = train_end
                val_end = min(val_start + fold_size, data_length)
                
                if val_start >= data_length:
                    break
                
                train_data = data.iloc[:train_end]
                val_data = data.iloc[val_start:val_end]
                
                # 시스템 훈련 및 평가
                score = self._train_and_evaluate(system, train_data, val_data)
                scores.append(score)
            
            # 평균 점수 반환
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"파라미터 평가 실패: {e}")
            return 0.0
    
    def _train_and_evaluate(self, system: Any, train_data: pd.DataFrame, 
                          val_data: pd.DataFrame) -> float:
        """시스템 훈련 및 평가"""
        try:
            # 시스템 훈련 (임시 구현)
            # 실제로는 system.train(train_data) 호출
            pass
            
            # 시스템 평가 (임시 구현)
            # 실제로는 system.evaluate(val_data) 호출
            # 여기서는 랜덤 점수 반환 (실제 구현시 교체)
            np.random.seed(42)
            return np.random.normal(0.5, 0.2)
            
        except Exception as e:
            self.logger.error(f"시스템 훈련 및 평가 실패: {e}")
            return 0.0
    
    def save_optimization_results(self, result: OptimizationResult, 
                                filepath: str) -> None:
        """최적화 결과 저장"""
        try:
            # 결과를 JSON으로 저장
            result_dict = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'total_evaluations': result.total_evaluations,
                'optimization_time': result.optimization_time,
                'optimization_history': result.optimization_history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"최적화 결과 저장 완료: {filepath}")
            
        except Exception as e:
            self.logger.error(f"최적화 결과 저장 실패: {e}")
            raise


# 편의 함수들
def create_hyperparameter_tuner(config: Dict) -> HyperparameterTuner:
    """하이퍼파라미터 튜너 생성 편의 함수"""
    return HyperparameterTuner(config)


def optimize_eswa_system(data: pd.DataFrame, config: Dict, 
                        system_factory: callable) -> OptimizationResult:
    """ESWA 시스템 최적화 편의 함수"""
    tuner = create_hyperparameter_tuner(config)
    return tuner.optimize_system_parameters(data, system_factory)
