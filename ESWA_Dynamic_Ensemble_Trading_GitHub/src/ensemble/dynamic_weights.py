"""
동적 가중치 할당 모듈
Dynamic Weight Allocation Module

성과 기반 동적 가중치 할당 시스템
- 최근 30일 성과 기반 가중치 계산
- Softmax 함수를 통한 가중치 정규화 (Temperature T=10)
- 성과 지표별 가중치 조정
- 시간 가중치 적용 (최근 성과에 더 높은 가중치)

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import deque
from abc import ABC, abstractmethod
import time
from datetime import datetime, timedelta


class BaseWeightCalculator(ABC):
    """
    기본 가중치 계산기 추상 클래스
    
    모든 가중치 계산 방법의 기본 인터페이스
    """
    
    def __init__(self, config: Dict):
        """
        기본 가중치 계산기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def calculate_weights(self, performance_data: Dict[str, List[float]], 
                         current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        가중치 계산 (추상 메서드)
        
        Args:
            performance_data: 성과 데이터
            current_time: 현재 시간
            
        Returns:
            계산된 가중치 딕셔너리
        """
        pass


class PerformanceBasedWeightCalculator(BaseWeightCalculator):
    """
    성과 기반 가중치 계산기
    
    최근 성과를 기반으로 동적 가중치 계산
    """
    
    def __init__(self, config: Dict):
        """
        성과 기반 가중치 계산기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 가중치 계산 설정
        self.ensemble_config = config.get('ensemble', {})
        self.dynamic_weights_config = self.ensemble_config.get('dynamic_weights', {})
        
        # 성과 평가 윈도우 (30일)
        self.performance_window = self.dynamic_weights_config.get('performance_window', 30)
        
        # Softmax 온도 파라미터
        self.temperature = self.dynamic_weights_config.get('temperature', 10.0)
        
        # 성과 지표 가중치
        self.metric_weights = self.dynamic_weights_config.get('metric_weights', {
            'cumulative_return': 0.4,
            'sharpe_ratio': 0.3,
            'max_drawdown': 0.2,
            'win_rate': 0.1
        })
        
        # 시간 가중치 설정
        self.time_decay_factor = self.dynamic_weights_config.get('time_decay_factor', 0.95)
        
        # 최소 가중치 (다양성 보장)
        self.min_weight = self.dynamic_weights_config.get('min_weight', 0.05)
        
        self.logger.info("성과 기반 가중치 계산기 초기화 완료")
        self.logger.info(f"성과 윈도우: {self.performance_window}일")
        self.logger.info(f"Softmax 온도: {self.temperature}")
    
    def calculate_weights(self, performance_data: Dict[str, List[float]], 
                         current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        성과 기반 가중치 계산
        
        Args:
            performance_data: 체제별 성과 데이터
            current_time: 현재 시간
            
        Returns:
            체제별 가중치 딕셔너리
        """
        try:
            if not performance_data:
                return self._get_equal_weights(performance_data)
            
            # 각 체제별 성과 점수 계산
            regime_scores = {}
            
            for regime, performance_history in performance_data.items():
                if not performance_history:
                    regime_scores[regime] = 0.0
                    continue
                
                # 최근 성과 데이터 추출
                recent_performance = self._get_recent_performance(
                    performance_history, self.performance_window
                )
                
                if not recent_performance:
                    regime_scores[regime] = 0.0
                    continue
                
                # 성과 점수 계산
                score = self._calculate_performance_score(recent_performance, current_time)
                regime_scores[regime] = score
            
            # Softmax 함수를 통한 가중치 정규화
            weights = self._apply_softmax_normalization(regime_scores)
            
            # 최소 가중치 보장
            weights = self._ensure_minimum_weights(weights)
            
            # 가중치 정규화
            weights = self._normalize_weights(weights)
            
            self.logger.debug(f"계산된 가중치: {weights}")
            
            return weights
            
        except Exception as e:
            self.logger.error(f"가중치 계산 실패: {e}")
            return self._get_equal_weights(performance_data)
    
    def _get_recent_performance(self, performance_history: List[float], 
                              window_size: int) -> List[float]:
        """
        최근 성과 데이터 추출
        
        Args:
            performance_history: 전체 성과 히스토리
            window_size: 윈도우 크기
            
        Returns:
            최근 성과 데이터
        """
        try:
            if len(performance_history) <= window_size:
                return performance_history
            
            return performance_history[-window_size:]
            
        except Exception as e:
            self.logger.error(f"최근 성과 데이터 추출 실패: {e}")
            return []
    
    def _calculate_performance_score(self, performance_data: List[float], 
                                   current_time: Optional[datetime] = None) -> float:
        """
        성과 점수 계산
        
        Args:
            performance_data: 성과 데이터
            current_time: 현재 시간
            
        Returns:
            성과 점수
        """
        try:
            if not performance_data:
                return 0.0
            
            # 기본 성과 지표 계산
            metrics = self._calculate_basic_metrics(performance_data)
            
            # 시간 가중치 적용
            time_weighted_score = self._apply_time_weights(performance_data, current_time)
            
            # 종합 성과 점수 계산
            composite_score = self._calculate_composite_score(metrics, time_weighted_score)
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"성과 점수 계산 실패: {e}")
            return 0.0
    
    def _calculate_basic_metrics(self, performance_data: List[float]) -> Dict[str, float]:
        """
        기본 성과 지표 계산
        
        Args:
            performance_data: 성과 데이터
            
        Returns:
            성과 지표 딕셔너리
        """
        try:
            if not performance_data:
                return {}
            
            performance_array = np.array(performance_data)
            
            # 누적 수익률
            cumulative_return = np.sum(performance_array)
            
            # 샤프 비율 (간단한 버전)
            if len(performance_array) > 1:
                sharpe_ratio = np.mean(performance_array) / (np.std(performance_array) + 1e-8)
            else:
                sharpe_ratio = 0.0
            
            # 최대 낙폭
            cumulative_returns = np.cumsum(performance_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # 승률
            positive_returns = np.sum(performance_array > 0)
            win_rate = positive_returns / len(performance_array) if len(performance_array) > 0 else 0.0
            
            return {
                'cumulative_return': cumulative_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            self.logger.error(f"기본 성과 지표 계산 실패: {e}")
            return {}
    
    def _apply_time_weights(self, performance_data: List[float], 
                          current_time: Optional[datetime] = None) -> float:
        """
        시간 가중치 적용
        
        Args:
            performance_data: 성과 데이터
            current_time: 현재 시간
            
        Returns:
            시간 가중치가 적용된 점수
        """
        try:
            if not performance_data:
                return 0.0
            
            # 시간 가중치 계산 (최근일수록 높은 가중치)
            n = len(performance_data)
            time_weights = np.array([self.time_decay_factor ** (n - 1 - i) for i in range(n)])
            
            # 가중 평균 계산
            weighted_score = np.average(performance_data, weights=time_weights)
            
            return weighted_score
            
        except Exception as e:
            self.logger.error(f"시간 가중치 적용 실패: {e}")
            return np.mean(performance_data) if performance_data else 0.0
    
    def _calculate_composite_score(self, metrics: Dict[str, float], 
                                 time_weighted_score: float) -> float:
        """
        종합 성과 점수 계산
        
        Args:
            metrics: 기본 성과 지표
            time_weighted_score: 시간 가중치 점수
            
        Returns:
            종합 성과 점수
        """
        try:
            composite_score = 0.0
            
            # 기본 성과 지표 가중합
            for metric_name, weight in self.metric_weights.items():
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    
                    # 지표별 정규화
                    if metric_name == 'max_drawdown':
                        # 최대 낙폭은 음수이므로 절댓값으로 변환
                        metric_value = abs(metric_value)
                    
                    composite_score += weight * metric_value
            
            # 시간 가중치 점수 추가 (0.3 가중치)
            composite_score += 0.3 * time_weighted_score
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"종합 성과 점수 계산 실패: {e}")
            return time_weighted_score
    
    def _apply_softmax_normalization(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Softmax 함수를 통한 가중치 정규화
        
        Args:
            scores: 체제별 성과 점수
            
        Returns:
            정규화된 가중치
        """
        try:
            if not scores:
                return {}
            
            # 점수를 배열로 변환
            regime_names = list(scores.keys())
            score_values = np.array(list(scores.values()))
            
            # Softmax 함수 적용
            # exp(scores / temperature) / sum(exp(scores / temperature))
            exp_scores = np.exp(score_values / self.temperature)
            softmax_weights = exp_scores / np.sum(exp_scores)
            
            # 딕셔너리로 변환
            weights = dict(zip(regime_names, softmax_weights))
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Softmax 정규화 실패: {e}")
            return self._get_equal_weights(scores)
    
    def _ensure_minimum_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        최소 가중치 보장
        
        Args:
            weights: 가중치 딕셔너리
            
        Returns:
            최소 가중치가 보장된 가중치
        """
        try:
            if not weights:
                return weights
            
            # 최소 가중치 미만인 체제들 식별
            below_minimum = [regime for regime, weight in weights.items() 
                           if weight < self.min_weight]
            
            if not below_minimum:
                return weights
            
            # 최소 가중치 할당
            total_minimum_allocation = len(below_minimum) * self.min_weight
            
            # 나머지 가중치에서 차감
            remaining_weight = 1.0 - total_minimum_allocation
            
            if remaining_weight <= 0:
                # 균등 분배
                return {regime: 1.0 / len(weights) for regime in weights.keys()}
            
            # 가중치 재분배
            adjusted_weights = {}
            total_above_minimum = sum(weight for regime, weight in weights.items() 
                                    if regime not in below_minimum)
            
            for regime, weight in weights.items():
                if regime in below_minimum:
                    adjusted_weights[regime] = self.min_weight
                else:
                    # 비례적으로 축소
                    adjusted_weights[regime] = weight * (remaining_weight / total_above_minimum)
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"최소 가중치 보장 실패: {e}")
            return weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        가중치 정규화 (합이 1이 되도록)
        
        Args:
            weights: 가중치 딕셔너리
            
        Returns:
            정규화된 가중치
        """
        try:
            if not weights:
                return weights
            
            total_weight = sum(weights.values())
            
            if total_weight == 0:
                return self._get_equal_weights(weights)
            
            # 정규화
            normalized_weights = {
                regime: weight / total_weight 
                for regime, weight in weights.items()
            }
            
            return normalized_weights
            
        except Exception as e:
            self.logger.error(f"가중치 정규화 실패: {e}")
            return self._get_equal_weights(weights)
    
    def _get_equal_weights(self, performance_data: Dict) -> Dict[str, float]:
        """
        균등 가중치 반환
        
        Args:
            performance_data: 성과 데이터
            
        Returns:
            균등 가중치 딕셔너리
        """
        try:
            if not performance_data:
                return {}
            
            num_regimes = len(performance_data)
            equal_weight = 1.0 / num_regimes
            
            return {regime: equal_weight for regime in performance_data.keys()}
            
        except Exception as e:
            self.logger.error(f"균등 가중치 계산 실패: {e}")
            return {}


class AdaptiveWeightCalculator(BaseWeightCalculator):
    """
    적응형 가중치 계산기
    
    시장 상황 변화에 적응하는 가중치 계산
    """
    
    def __init__(self, config: Dict):
        """
        적응형 가중치 계산기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 적응형 설정
        self.adaptation_config = config.get('ensemble', {}).get('adaptive_weights', {})
        self.volatility_threshold = self.adaptation_config.get('volatility_threshold', 0.02)
        self.performance_threshold = self.adaptation_config.get('performance_threshold', 0.1)
        
        # 기본 가중치 계산기
        self.base_calculator = PerformanceBasedWeightCalculator(config)
        
        # 적응 히스토리
        self.adaptation_history = deque(maxlen=100)
        
        self.logger.info("적응형 가중치 계산기 초기화 완료")
    
    def calculate_weights(self, performance_data: Dict[str, List[float]], 
                         current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        적응형 가중치 계산
        
        Args:
            performance_data: 성과 데이터
            current_time: 현재 시간
            
        Returns:
            적응형 가중치 딕셔너리
        """
        try:
            # 기본 가중치 계산
            base_weights = self.base_calculator.calculate_weights(performance_data, current_time)
            
            # 적응 조정
            adapted_weights = self._apply_adaptive_adjustments(
                base_weights, performance_data, current_time
            )
            
            # 적응 히스토리 업데이트
            self.adaptation_history.append({
                'timestamp': current_time or datetime.now(),
                'base_weights': base_weights,
                'adapted_weights': adapted_weights
            })
            
            return adapted_weights
            
        except Exception as e:
            self.logger.error(f"적응형 가중치 계산 실패: {e}")
            return self.base_calculator.calculate_weights(performance_data, current_time)
    
    def _apply_adaptive_adjustments(self, base_weights: Dict[str, float], 
                                  performance_data: Dict[str, List[float]], 
                                  current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        적응 조정 적용
        
        Args:
            base_weights: 기본 가중치
            performance_data: 성과 데이터
            current_time: 현재 시간
            
        Returns:
            적응 조정된 가중치
        """
        try:
            # 시장 변동성 분석
            market_volatility = self._calculate_market_volatility(performance_data)
            
            # 성과 편차 분석
            performance_dispersion = self._calculate_performance_dispersion(performance_data)
            
            # 적응 조정 계수
            adaptation_factor = self._calculate_adaptation_factor(
                market_volatility, performance_dispersion
            )
            
            # 가중치 조정
            adjusted_weights = {}
            for regime, weight in base_weights.items():
                # 변동성이 높을 때는 보수적 조정
                if market_volatility > self.volatility_threshold:
                    adjusted_weight = weight * (1 - adaptation_factor * 0.1)
                else:
                    adjusted_weight = weight * (1 + adaptation_factor * 0.05)
                
                adjusted_weights[regime] = max(0.01, adjusted_weight)  # 최소 가중치 보장
            
            # 정규화
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {
                    regime: weight / total_weight 
                    for regime, weight in adjusted_weights.items()
                }
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"적응 조정 적용 실패: {e}")
            return base_weights
    
    def _calculate_market_volatility(self, performance_data: Dict[str, List[float]]) -> float:
        """시장 변동성 계산"""
        try:
            all_returns = []
            for regime_data in performance_data.values():
                all_returns.extend(regime_data)
            
            if len(all_returns) < 2:
                return 0.0
            
            return np.std(all_returns)
            
        except Exception as e:
            self.logger.error(f"시장 변동성 계산 실패: {e}")
            return 0.0
    
    def _calculate_performance_dispersion(self, performance_data: Dict[str, List[float]]) -> float:
        """성과 편차 계산"""
        try:
            regime_means = []
            for regime_data in performance_data.values():
                if regime_data:
                    regime_means.append(np.mean(regime_data))
            
            if len(regime_means) < 2:
                return 0.0
            
            return np.std(regime_means)
            
        except Exception as e:
            self.logger.error(f"성과 편차 계산 실패: {e}")
            return 0.0
    
    def _calculate_adaptation_factor(self, volatility: float, dispersion: float) -> float:
        """적응 계수 계산"""
        try:
            # 변동성과 편차를 기반으로 적응 계수 계산
            volatility_factor = min(1.0, volatility / self.volatility_threshold)
            dispersion_factor = min(1.0, dispersion / self.performance_threshold)
            
            adaptation_factor = (volatility_factor + dispersion_factor) / 2.0
            
            return adaptation_factor
            
        except Exception as e:
            self.logger.error(f"적응 계수 계산 실패: {e}")
            return 0.0


class DynamicWeightAllocation:
    """
    동적 가중치 할당 메인 클래스
    
    성과 기반 동적 가중치 할당 시스템 관리
    """
    
    def __init__(self, config: Dict, weight_calculator_type: str = 'performance_based'):
        """
        동적 가중치 할당 초기화
        
        Args:
            config: 설정 딕셔너리
            weight_calculator_type: 가중치 계산기 타입
        """
        self.config = config
        self.weight_calculator_type = weight_calculator_type
        
        # 가중치 계산기 초기화
        if weight_calculator_type == 'performance_based':
            self.weight_calculator = PerformanceBasedWeightCalculator(config)
        elif weight_calculator_type == 'adaptive':
            self.weight_calculator = AdaptiveWeightCalculator(config)
        else:
            raise ValueError(f"지원하지 않는 가중치 계산기 타입: {weight_calculator_type}")
        
        # 가중치 히스토리
        self.weight_history = deque(maxlen=1000)
        self.performance_history = {}
        
        # 현재 가중치
        self.current_weights = {}
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"동적 가중치 할당 초기화 완료: {weight_calculator_type}")
    
    def update_weights(self, performance_data: Dict[str, List[float]], 
                      current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        가중치 업데이트
        
        Args:
            performance_data: 체제별 성과 데이터
            current_time: 현재 시간
            
        Returns:
            업데이트된 가중치
        """
        try:
            # 성과 히스토리 업데이트
            self._update_performance_history(performance_data)
            
            # 가중치 계산
            new_weights = self.weight_calculator.calculate_weights(
                performance_data, current_time
            )
            
            # 가중치 히스토리 업데이트
            self.weight_history.append({
                'timestamp': current_time or datetime.now(),
                'weights': new_weights.copy()
            })
            
            # 현재 가중치 업데이트
            self.current_weights = new_weights
            
            self.logger.debug(f"가중치 업데이트 완료: {new_weights}")
            
            return new_weights
            
        except Exception as e:
            self.logger.error(f"가중치 업데이트 실패: {e}")
            return self.current_weights
    
    def _update_performance_history(self, performance_data: Dict[str, List[float]]):
        """성과 히스토리 업데이트"""
        try:
            for regime, data in performance_data.items():
                if regime not in self.performance_history:
                    self.performance_history[regime] = deque(maxlen=1000)
                
                # 새로운 성과 데이터 추가
                for performance in data:
                    self.performance_history[regime].append(performance)
            
        except Exception as e:
            self.logger.error(f"성과 히스토리 업데이트 실패: {e}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """현재 가중치 반환"""
        return self.current_weights.copy()
    
    def get_weight_history(self, days: int = 30) -> List[Dict]:
        """가중치 히스토리 반환"""
        try:
            if not self.weight_history:
                return []
            
            # 최근 N일 데이터 반환
            recent_history = list(self.weight_history)[-days:]
            return recent_history
            
        except Exception as e:
            self.logger.error(f"가중치 히스토리 조회 실패: {e}")
            return []
    
    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """가중치 통계 반환"""
        try:
            if not self.weight_history:
                return {}
            
            # 모든 체제 수집
            all_regimes = set()
            for entry in self.weight_history:
                all_regimes.update(entry['weights'].keys())
            
            # 체제별 통계 계산
            statistics = {}
            for regime in all_regimes:
                weights = [entry['weights'].get(regime, 0.0) for entry in self.weight_history]
                
                if weights:
                    statistics[regime] = {
                        'mean_weight': np.mean(weights),
                        'std_weight': np.std(weights),
                        'min_weight': np.min(weights),
                        'max_weight': np.max(weights),
                        'current_weight': self.current_weights.get(regime, 0.0)
                    }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"가중치 통계 계산 실패: {e}")
            return {}
    
    def reset_weights(self):
        """가중치 초기화"""
        try:
            self.current_weights = {}
            self.weight_history.clear()
            self.performance_history.clear()
            
            self.logger.info("가중치 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"가중치 초기화 실패: {e}")


# 편의 함수들
def create_dynamic_weight_allocation(config: Dict, 
                                   weight_calculator_type: str = 'performance_based') -> DynamicWeightAllocation:
    """동적 가중치 할당 생성 편의 함수"""
    return DynamicWeightAllocation(config, weight_calculator_type)


def create_performance_based_calculator(config: Dict) -> PerformanceBasedWeightCalculator:
    """성과 기반 가중치 계산기 생성 편의 함수"""
    return PerformanceBasedWeightCalculator(config)


def create_adaptive_calculator(config: Dict) -> AdaptiveWeightCalculator:
    """적응형 가중치 계산기 생성 편의 함수"""
    return AdaptiveWeightCalculator(config)
