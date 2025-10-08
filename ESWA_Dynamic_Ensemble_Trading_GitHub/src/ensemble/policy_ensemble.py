"""
정책 앙상블 모듈
Policy Ensemble Module

최종 정책 결정 및 액션 생성
- 체제별 에이전트 풀의 액션 수집
- 동적 가중치를 통한 액션 융합
- 최종 거래 결정 생성
- 불확실성 관리 및 리스크 제어

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict, deque
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from datetime import datetime

from .dynamic_weights import DynamicWeightAllocation, create_dynamic_weight_allocation
from ..agents.agent_pool import AgentPool
from ..regime.confidence_selection import ConfidenceBasedSelection


class BaseActionFusion(ABC):
    """
    기본 액션 융합 추상 클래스
    
    모든 액션 융합 방법의 기본 인터페이스
    """
    
    def __init__(self, config: Dict):
        """
        기본 액션 융합 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fuse_actions(self, regime_actions: Dict[str, List[Tuple[int, float, float]]], 
                    weights: Dict[str, float]) -> Tuple[int, float, Dict[str, float]]:
        """
        액션 융합 (추상 메서드)
        
        Args:
            regime_actions: 체제별 액션 리스트
            weights: 체제별 가중치
            
        Returns:
            융합된 액션, 신뢰도, 액션 분포
        """
        pass


class WeightedActionFusion(BaseActionFusion):
    """
    가중치 기반 액션 융합
    
    동적 가중치를 사용한 액션 융합
    """
    
    def __init__(self, config: Dict):
        """
        가중치 기반 액션 융합 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 융합 설정
        self.ensemble_config = config.get('ensemble', {})
        self.fusion_config = self.ensemble_config.get('action_fusion', {})
        
        # 액션 융합 방법
        self.fusion_method = self.fusion_config.get('method', 'weighted_average')
        
        # 신뢰도 임계값
        self.confidence_threshold = self.fusion_config.get('confidence_threshold', 0.6)
        
        # 액션 차원
        self.action_dim = 5  # Strong Buy, Buy, Hold, Sell, Strong Sell
        
        self.logger.info("가중치 기반 액션 융합 초기화 완료")
        self.logger.info(f"융합 방법: {self.fusion_method}")
    
    def fuse_actions(self, regime_actions: Dict[str, List[Tuple[int, float, float]]], 
                    weights: Dict[str, float]) -> Tuple[int, float, Dict[str, float]]:
        """
        가중치 기반 액션 융합
        
        Args:
            regime_actions: 체제별 액션 리스트
            weights: 체제별 가중치
            
        Returns:
            융합된 액션, 신뢰도, 액션 분포
        """
        try:
            if not regime_actions or not weights:
                return 2, 0.0, {}  # 기본값: Hold
            
            # 액션 분포 계산
            action_distribution = self._calculate_action_distribution(regime_actions, weights)
            
            # 융합 방법에 따른 최종 액션 결정
            if self.fusion_method == 'weighted_average':
                final_action, confidence = self._weighted_average_fusion(action_distribution)
            elif self.fusion_method == 'weighted_voting':
                final_action, confidence = self._weighted_voting_fusion(regime_actions, weights)
            elif self.fusion_method == 'softmax_fusion':
                final_action, confidence = self._softmax_fusion(action_distribution)
            else:
                final_action, confidence = self._weighted_average_fusion(action_distribution)
            
            return final_action, confidence, action_distribution
            
        except Exception as e:
            self.logger.error(f"액션 융합 실패: {e}")
            return 2, 0.0, {}
    
    def _calculate_action_distribution(self, regime_actions: Dict[str, List[Tuple[int, float, float]]], 
                                     weights: Dict[str, float]) -> Dict[str, float]:
        """
        액션 분포 계산
        
        Args:
            regime_actions: 체제별 액션 리스트
            weights: 체제별 가중치
            
        Returns:
            액션별 확률 분포
        """
        try:
            # 액션별 가중치 합계 초기화
            action_weights = {i: 0.0 for i in range(self.action_dim)}
            
            for regime, actions in regime_actions.items():
                regime_weight = weights.get(regime, 0.0)
                
                if regime_weight == 0.0 or not actions:
                    continue
                
                # 체제 내 에이전트별 가중치 (균등 분배)
                agent_weight = regime_weight / len(actions) if actions else 0.0
                
                for action, log_prob, value in actions:
                    if 0 <= action < self.action_dim:
                        # 로그 확률을 확률로 변환하여 가중치 적용
                        action_prob = np.exp(log_prob) if log_prob != 0.0 else 1.0 / self.action_dim
                        action_weights[action] += agent_weight * action_prob
            
            # 정규화
            total_weight = sum(action_weights.values())
            if total_weight > 0:
                action_distribution = {
                    str(action): weight / total_weight 
                    for action, weight in action_weights.items()
                }
            else:
                # 균등 분포
                action_distribution = {
                    str(action): 1.0 / self.action_dim 
                    for action in range(self.action_dim)
                }
            
            return action_distribution
            
        except Exception as e:
            self.logger.error(f"액션 분포 계산 실패: {e}")
            return {str(i): 1.0 / self.action_dim for i in range(self.action_dim)}
    
    def _weighted_average_fusion(self, action_distribution: Dict[str, float]) -> Tuple[int, float]:
        """
        가중 평균 융합
        
        Args:
            action_distribution: 액션 분포
            
        Returns:
            최종 액션, 신뢰도
        """
        try:
            if not action_distribution:
                return 2, 0.0
            
            # 가장 높은 확률의 액션 선택
            best_action = max(action_distribution.items(), key=lambda x: x[1])
            final_action = int(best_action[0])
            confidence = best_action[1]
            
            return final_action, confidence
            
        except Exception as e:
            self.logger.error(f"가중 평균 융합 실패: {e}")
            return 2, 0.0
    
    def _weighted_voting_fusion(self, regime_actions: Dict[str, List[Tuple[int, float, float]]], 
                               weights: Dict[str, float]) -> Tuple[int, float]:
        """
        가중 투표 융합
        
        Args:
            regime_actions: 체제별 액션 리스트
            weights: 체제별 가중치
            
        Returns:
            최종 액션, 신뢰도
        """
        try:
            # 액션별 투표 수 집계
            action_votes = {i: 0.0 for i in range(self.action_dim)}
            
            for regime, actions in regime_actions.items():
                regime_weight = weights.get(regime, 0.0)
                
                if regime_weight == 0.0 or not actions:
                    continue
                
                # 체제 내 에이전트별 투표
                for action, log_prob, value in actions:
                    if 0 <= action < self.action_dim:
                        # 로그 확률을 투표 가중치로 사용
                        vote_weight = np.exp(log_prob) if log_prob != 0.0 else 1.0
                        action_votes[action] += regime_weight * vote_weight
            
            # 가장 많은 투표를 받은 액션 선택
            best_action = max(action_votes.items(), key=lambda x: x[1])
            final_action = best_action[0]
            
            # 신뢰도 계산 (최대 투표 비율)
            total_votes = sum(action_votes.values())
            confidence = best_action[1] / total_votes if total_votes > 0 else 0.0
            
            return final_action, confidence
            
        except Exception as e:
            self.logger.error(f"가중 투표 융합 실패: {e}")
            return 2, 0.0
    
    def _softmax_fusion(self, action_distribution: Dict[str, float]) -> Tuple[int, float]:
        """
        Softmax 융합
        
        Args:
            action_distribution: 액션 분포
            
        Returns:
            최종 액션, 신뢰도
        """
        try:
            if not action_distribution:
                return 2, 0.0
            
            # 확률값을 배열로 변환
            action_probs = np.array([action_distribution.get(str(i), 0.0) for i in range(self.action_dim)])
            
            # Softmax 적용 (온도 파라미터 = 1.0)
            softmax_probs = F.softmax(torch.tensor(action_probs), dim=0).numpy()
            
            # 가장 높은 확률의 액션 선택
            final_action = np.argmax(softmax_probs)
            confidence = softmax_probs[final_action]
            
            return final_action, confidence
            
        except Exception as e:
            self.logger.error(f"Softmax 융합 실패: {e}")
            return 2, 0.0


class PolicyEnsemble:
    """
    정책 앙상블 메인 클래스
    
    체제별 에이전트 풀을 통합하여 최종 거래 결정 생성
    """
    
    def __init__(self, config: Dict, agent_pool: AgentPool = None, 
                 weight_calculator_type: str = 'performance_based'):
        """
        정책 앙상블 초기화
        
        Args:
            config: 설정 딕셔너리
            agent_pool: 에이전트 풀 (None이면 자동 생성)
            weight_calculator_type: 가중치 계산기 타입
        """
        self.config = config
        
        # 에이전트 풀 초기화 (없으면 자동 생성)
        if agent_pool is None:
            from ..agents.agent_pool import AgentPool
            self.agent_pool = AgentPool(config)
        else:
            self.agent_pool = agent_pool
        
        # 동적 가중치 할당 초기화
        self.weight_allocation = create_dynamic_weight_allocation(
            config, weight_calculator_type
        )
        
        # 액션 융합 초기화
        self.action_fusion = WeightedActionFusion(config)
        
        # 불확실성 관리 초기화
        self.uncertainty_manager = ConfidenceBasedSelection(config)
        
        # 의사결정 히스토리
        self.decision_history = deque(maxlen=1000)
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # 현재 상태
        self.current_regime = None
        self.current_weights = {}
        self.last_decision = None
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("정책 앙상블 초기화 완료")
    
    def make_decision(self, state: np.ndarray, regime: str, 
                     regime_probabilities: Optional[Dict[str, float]] = None,
                     current_time: Optional[datetime] = None) -> Dict[str, Union[int, float, Dict]]:
        """
        최종 거래 결정 생성
        
        Args:
            state: 현재 상태 (273차원 멀티모달 특징)
            regime: 현재 시장 체제
            regime_probabilities: 체제 확률 (선택사항)
            current_time: 현재 시간 (선택사항)
            
        Returns:
            거래 결정 딕셔너리
        """
        try:
            # 1. 불확실성 관리 (체제 확률이 제공된 경우)
            if regime_probabilities is not None:
                regime = self.uncertainty_manager.select_regime(
                    regime_probabilities, self.current_regime
                )
            
            # 2. 체제별 에이전트 액션 수집
            regime_actions = self._collect_regime_actions(state, regime)
            
            # 3. 동적 가중치 업데이트
            performance_data = self._get_performance_data()
            weights = self.weight_allocation.update_weights(performance_data, current_time)
            
            # 4. 액션 융합
            final_action, confidence, action_distribution = self.action_fusion.fuse_actions(
                regime_actions, weights
            )
            
            # 5. 신뢰도 기반 의사결정 조정
            if confidence < self.action_fusion.confidence_threshold:
                # 신뢰도가 낮으면 보수적 액션 (Hold)
                final_action = 2
                confidence = 0.5
            
            # 6. 의사결정 히스토리 업데이트
            decision_info = {
                'timestamp': current_time or datetime.now(),
                'regime': regime,
                'final_action': final_action,
                'confidence': confidence,
                'action_distribution': action_distribution,
                'weights': weights.copy(),
                'regime_actions': regime_actions
            }
            
            self.decision_history.append(decision_info)
            
            # 7. 상태 업데이트
            self.current_regime = regime
            self.current_weights = weights
            self.last_decision = decision_info
            
            # 8. 최종 결정 반환
            final_decision = {
                'action': final_action,
                'confidence': confidence,
                'regime': regime,
                'action_distribution': action_distribution,
                'weights': weights,
                'decision_info': decision_info
            }
            
            self.logger.debug(f"거래 결정 생성: 액션={final_action}, 신뢰도={confidence:.3f}, 체제={regime}")
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"거래 결정 생성 실패: {e}")
            return self._get_default_decision()
    
    def _collect_regime_actions(self, state: np.ndarray, regime: str) -> Dict[str, List[Tuple[int, float, float]]]:
        """
        체제별 에이전트 액션 수집
        
        Args:
            state: 현재 상태
            regime: 현재 체제
            
        Returns:
            체제별 액션 리스트
        """
        try:
            regime_actions = {}
            
            # 현재 체제의 에이전트 액션 수집
            current_regime_actions = self.agent_pool.get_agent_actions(regime, state)
            regime_actions[regime] = current_regime_actions
            
            # 다른 체제의 에이전트 액션도 수집 (다양성 확보)
            for other_regime in ['bull_market', 'bear_market', 'sideways_market']:
                if other_regime != regime:
                    other_actions = self.agent_pool.get_agent_actions(other_regime, state)
                    regime_actions[other_regime] = other_actions
            
            return regime_actions
            
        except Exception as e:
            self.logger.error(f"체제별 액션 수집 실패: {e}")
            return {regime: [(2, 0.0, 0.0)]}  # 기본값: Hold
    
    def _get_performance_data(self) -> Dict[str, List[float]]:
        """
        성과 데이터 수집
        
        Returns:
            체제별 성과 데이터
        """
        try:
            performance_data = {}
            
            # 에이전트 풀에서 성과 데이터 수집
            for regime in ['bull_market', 'bear_market', 'sideways_market']:
                regime_performance = self.agent_pool.get_recent_performance(regime, days=30)
                
                if regime_performance and 'average_performance' in regime_performance:
                    # 성과 히스토리에서 데이터 추출
                    performance_history = list(self.agent_pool.performance_history.get(regime, []))
                    performance_data[regime] = performance_history
                else:
                    performance_data[regime] = []
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"성과 데이터 수집 실패: {e}")
            return {}
    
    def _get_default_decision(self) -> Dict[str, Union[int, float, Dict]]:
        """기본 거래 결정 반환"""
        return {
            'action': 2,  # Hold
            'confidence': 0.5,
            'regime': 'sideways_market',
            'action_distribution': {str(i): 1.0/5 for i in range(5)},
            'weights': {'bull_market': 0.33, 'bear_market': 0.33, 'sideways_market': 0.34},
            'decision_info': {}
        }
    
    def update_performance(self, regime: str, performance: float):
        """
        성과 업데이트
        
        Args:
            regime: 체제
            performance: 성과 점수
        """
        try:
            # 에이전트 풀 성과 업데이트
            self.agent_pool.update_performance_history(regime, performance)
            
            # 앙상블 성과 히스토리 업데이트
            self.performance_history[regime].append(performance)
            
        except Exception as e:
            self.logger.error(f"성과 업데이트 실패: {e}")
    
    def get_decision_statistics(self, days: int = 30) -> Dict[str, Union[float, int, Dict]]:
        """
        의사결정 통계 반환
        
        Args:
            days: 최근 일수
            
        Returns:
            의사결정 통계
        """
        try:
            if not self.decision_history:
                return {}
            
            # 최근 의사결정 데이터
            recent_decisions = list(self.decision_history)[-days:]
            
            if not recent_decisions:
                return {}
            
            # 액션 분포 통계
            action_counts = defaultdict(int)
            regime_counts = defaultdict(int)
            confidence_scores = []
            
            for decision in recent_decisions:
                action = decision.get('final_action', 2)
                regime = decision.get('regime', 'unknown')
                confidence = decision.get('confidence', 0.0)
                
                action_counts[action] += 1
                regime_counts[regime] += 1
                confidence_scores.append(confidence)
            
            # 통계 계산
            total_decisions = len(recent_decisions)
            action_distribution = {
                action: count / total_decisions 
                for action, count in action_counts.items()
            }
            
            regime_distribution = {
                regime: count / total_decisions 
                for regime, count in regime_counts.items()
            }
            
            statistics = {
                'total_decisions': total_decisions,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'std_confidence': np.std(confidence_scores) if confidence_scores else 0.0,
                'action_distribution': action_distribution,
                'regime_distribution': regime_distribution,
                'current_weights': self.current_weights.copy()
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"의사결정 통계 계산 실패: {e}")
            return {}
    
    def get_ensemble_summary(self) -> Dict[str, Union[str, int, bool, Dict]]:
        """앙상블 요약 정보 반환"""
        try:
            summary = {
                'weight_calculator_type': self.weight_allocation.weight_calculator_type,
                'current_regime': self.current_regime,
                'current_weights': self.current_weights.copy(),
                'total_decisions': len(self.decision_history),
                'agent_pool_summary': self.agent_pool.get_pool_summary(),
                'weight_statistics': self.weight_allocation.get_weight_statistics(),
                'decision_statistics': self.get_decision_statistics()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"앙상블 요약 정보 생성 실패: {e}")
            return {}
    
    def reset_ensemble(self):
        """앙상블 초기화"""
        try:
            self.decision_history.clear()
            self.performance_history.clear()
            self.weight_allocation.reset_weights()
            
            self.current_regime = None
            self.current_weights = {}
            self.last_decision = None
            
            self.logger.info("앙상블 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"앙상블 초기화 실패: {e}")


# 편의 함수들
def create_policy_ensemble(config: Dict, agent_pool: AgentPool, 
                          weight_calculator_type: str = 'performance_based') -> PolicyEnsemble:
    """정책 앙상블 생성 편의 함수"""
    return PolicyEnsemble(config, agent_pool, weight_calculator_type)


def create_weighted_action_fusion(config: Dict) -> WeightedActionFusion:
    """가중치 기반 액션 융합 생성 편의 함수"""
    return WeightedActionFusion(config)
