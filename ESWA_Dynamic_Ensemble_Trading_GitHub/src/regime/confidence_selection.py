"""
Confidence-based Selection 메커니즘 모듈
Confidence-based Selection Mechanism Module

시장 체제 분류의 불확실성을 관리하고 정책 안정성을 확보
- 확률 임계값 기반 체제 선택
- 불확실성 상황에서 이전 체제 유지
- 체제 전환 시 정책 안정성 보장
- Classification with a Reject Option 구현

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import deque
import json
from pathlib import Path


class ConfidenceBasedSelection:
    """
    Confidence-based Selection 메커니즘
    
    시장 체제 분류의 불확실성을 관리하여 정책 안정성 확보
    """
    
    def __init__(self, config: Dict):
        """
        Confidence-based Selection 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.regime_config = config.get('regime', {})
        
        # 임계값 설정
        self.confidence_threshold = self.regime_config.get('classifier', {}).get('confidence_threshold', 0.6)
        
        # 체제 이름 및 라벨
        self.regime_names = ['bull_market', 'bear_market', 'sideways_market']
        self.regime_labels = {name: i for i, name in enumerate(self.regime_names)}
        
        # 상태 관리
        self.current_regime = 'sideways_market'  # 기본값
        self.previous_regime = 'sideways_market'
        self.regime_history = deque(maxlen=100)  # 최근 100개 체제 기록
        self.confidence_history = deque(maxlen=100)  # 최근 100개 신뢰도 기록
        
        # 통계 정보
        self.regime_transitions = 0
        self.confidence_rejections = 0
        self.total_predictions = 0
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Confidence-based Selection 초기화 완료")
        self.logger.info(f"신뢰도 임계값: {self.confidence_threshold}")
    
    def calculate_confidence(self, regime_probabilities: Dict[str, float]) -> float:
        """
        체제 분류 신뢰도 계산
        
        Args:
            regime_probabilities: 체제별 확률 딕셔너리
            
        Returns:
            최대 확률값 (신뢰도)
        """
        try:
            if not regime_probabilities:
                return 0.0
            
            # 최대 확률값을 신뢰도로 사용
            max_probability = max(regime_probabilities.values())
            
            return float(max_probability)
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 실패: {e}")
            return 0.0
    
    def select_regime(self, regime_probabilities: Dict[str, float]) -> str:
        """
        신뢰도 기반 체제 선택
        
        Args:
            regime_probabilities: 체제별 확률 딕셔너리
            
        Returns:
            선택된 체제 이름
        """
        try:
            self.total_predictions += 1
            
            # 신뢰도 계산
            confidence = self.calculate_confidence(regime_probabilities)
            
            # 신뢰도 기록
            self.confidence_history.append(confidence)
            
            # 신뢰도가 임계값 이상인 경우
            if confidence >= self.confidence_threshold:
                # 최대 확률 체제 선택
                selected_regime = max(regime_probabilities, key=regime_probabilities.get)
                
                # 체제 전환 확인
                if selected_regime != self.current_regime:
                    self.regime_transitions += 1
                    self.previous_regime = self.current_regime
                    self.current_regime = selected_regime
                    
                    self.logger.info(f"체제 전환: {self.previous_regime} → {self.current_regime} "
                                   f"(신뢰도: {confidence:.3f})")
                
                # 체제 기록
                self.regime_history.append(self.current_regime)
                
                return self.current_regime
            
            else:
                # 신뢰도가 임계값 미만인 경우 (Rejection)
                self.confidence_rejections += 1
                
                # 이전 체제 유지
                self.logger.debug(f"신뢰도 부족으로 이전 체제 유지: {self.current_regime} "
                                f"(신뢰도: {confidence:.3f} < {self.confidence_threshold})")
                
                # 체제 기록 (이전 체제 유지)
                self.regime_history.append(self.current_regime)
                
                return self.current_regime
                
        except Exception as e:
            self.logger.error(f"체제 선택 실패: {e}")
            # 오류 시 현재 체제 유지
            return self.current_regime
    
    def get_regime_confidence(self, regime_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        체제별 신뢰도 정보 반환
        
        Args:
            regime_probabilities: 체제별 확률 딕셔너리
            
        Returns:
            체제별 신뢰도 정보
        """
        try:
            confidence = self.calculate_confidence(regime_probabilities)
            
            # 각 체제별 신뢰도 계산
            regime_confidences = {}
            for regime_name, probability in regime_probabilities.items():
                # 개별 체제 신뢰도 (해당 체제가 선택될 확률)
                regime_confidences[regime_name] = {
                    'probability': probability,
                    'confidence': confidence,
                    'is_selected': probability == max(regime_probabilities.values()),
                    'is_above_threshold': confidence >= self.confidence_threshold
                }
            
            return regime_confidences
            
        except Exception as e:
            self.logger.error(f"체제 신뢰도 계산 실패: {e}")
            return {}
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """
        불확실성 지표 계산
        
        Returns:
            불확실성 지표 딕셔너리
        """
        try:
            if not self.confidence_history:
                return {
                    'average_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0,
                    'rejection_rate': 0.0,
                    'transition_rate': 0.0
                }
            
            # 신뢰도 통계
            confidences = list(self.confidence_history)
            average_confidence = np.mean(confidences)
            min_confidence = np.min(confidences)
            max_confidence = np.max(confidences)
            
            # 거부율 (Rejection Rate)
            rejection_rate = self.confidence_rejections / max(self.total_predictions, 1)
            
            # 전환율 (Transition Rate)
            transition_rate = self.regime_transitions / max(self.total_predictions, 1)
            
            return {
                'average_confidence': float(average_confidence),
                'min_confidence': float(min_confidence),
                'max_confidence': float(max_confidence),
                'rejection_rate': float(rejection_rate),
                'transition_rate': float(transition_rate)
            }
            
        except Exception as e:
            self.logger.error(f"불확실성 지표 계산 실패: {e}")
            return {}
    
    def get_regime_stability_metrics(self) -> Dict[str, Union[int, float, List[str]]]:
        """
        체제 안정성 지표 계산
        
        Returns:
            체제 안정성 지표 딕셔너리
        """
        try:
            if not self.regime_history:
                return {
                    'current_regime': self.current_regime,
                    'regime_duration': 0,
                    'recent_regimes': [],
                    'regime_distribution': {}
                }
            
            # 현재 체제 지속 시간
            regime_duration = 0
            for regime in reversed(self.regime_history):
                if regime == self.current_regime:
                    regime_duration += 1
                else:
                    break
            
            # 최근 체제들 (최근 10개)
            recent_regimes = list(self.regime_history)[-10:]
            
            # 체제 분포
            regime_counts = {}
            for regime in self.regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total_count = len(self.regime_history)
            regime_distribution = {
                regime: count / total_count 
                for regime, count in regime_counts.items()
            }
            
            return {
                'current_regime': self.current_regime,
                'regime_duration': regime_duration,
                'recent_regimes': recent_regimes,
                'regime_distribution': regime_distribution
            }
            
        except Exception as e:
            self.logger.error(f"체제 안정성 지표 계산 실패: {e}")
            return {}
    
    def analyze_regime_transitions(self) -> Dict[str, Union[int, List[Tuple[str, str]]]]:
        """
        체제 전환 분석
        
        Returns:
            체제 전환 분석 결과
        """
        try:
            if len(self.regime_history) < 2:
                return {
                    'total_transitions': 0,
                    'transition_pairs': [],
                    'most_common_transition': None
                }
            
            # 전환 쌍 추출
            transition_pairs = []
            for i in range(1, len(self.regime_history)):
                prev_regime = self.regime_history[i-1]
                curr_regime = self.regime_history[i]
                if prev_regime != curr_regime:
                    transition_pairs.append((prev_regime, curr_regime))
            
            # 가장 빈번한 전환
            if transition_pairs:
                transition_counts = {}
                for pair in transition_pairs:
                    transition_counts[pair] = transition_counts.get(pair, 0) + 1
                
                most_common_transition = max(transition_counts, key=transition_counts.get)
            else:
                most_common_transition = None
            
            return {
                'total_transitions': len(transition_pairs),
                'transition_pairs': transition_pairs,
                'most_common_transition': most_common_transition
            }
            
        except Exception as e:
            self.logger.error(f"체제 전환 분석 실패: {e}")
            return {}
    
    def reset_state(self):
        """상태 초기화"""
        try:
            self.current_regime = 'sideways_market'
            self.previous_regime = 'sideways_market'
            self.regime_history.clear()
            self.confidence_history.clear()
            self.regime_transitions = 0
            self.confidence_rejections = 0
            self.total_predictions = 0
            
            self.logger.info("Confidence-based Selection 상태 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"상태 초기화 실패: {e}")
    
    def update_threshold(self, new_threshold: float):
        """
        신뢰도 임계값 업데이트
        
        Args:
            new_threshold: 새로운 임계값 (0.0 ~ 1.0)
        """
        try:
            if not (0.0 <= new_threshold <= 1.0):
                raise ValueError("임계값은 0.0과 1.0 사이여야 합니다.")
            
            old_threshold = self.confidence_threshold
            self.confidence_threshold = new_threshold
            
            self.logger.info(f"신뢰도 임계값 업데이트: {old_threshold} → {new_threshold}")
            
        except Exception as e:
            self.logger.error(f"임계값 업데이트 실패: {e}")
            raise
    
    def get_status_report(self) -> Dict[str, Union[str, int, float, Dict]]:
        """
        상태 리포트 생성
        
        Returns:
            종합 상태 리포트
        """
        try:
            uncertainty_metrics = self.get_uncertainty_metrics()
            stability_metrics = self.get_regime_stability_metrics()
            transition_analysis = self.analyze_regime_transitions()
            
            report = {
                'current_status': {
                    'current_regime': self.current_regime,
                    'previous_regime': self.previous_regime,
                    'confidence_threshold': self.confidence_threshold
                },
                'uncertainty_metrics': uncertainty_metrics,
                'stability_metrics': stability_metrics,
                'transition_analysis': transition_analysis,
                'statistics': {
                    'total_predictions': self.total_predictions,
                    'regime_transitions': self.regime_transitions,
                    'confidence_rejections': self.confidence_rejections
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"상태 리포트 생성 실패: {e}")
            return {}
    
    def save_state(self, path: str):
        """상태 저장"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                'current_regime': self.current_regime,
                'previous_regime': self.previous_regime,
                'regime_history': list(self.regime_history),
                'confidence_history': list(self.confidence_history),
                'regime_transitions': self.regime_transitions,
                'confidence_rejections': self.confidence_rejections,
                'total_predictions': self.total_predictions,
                'confidence_threshold': self.confidence_threshold
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Confidence-based Selection 상태 저장: {path}")
            
        except Exception as e:
            self.logger.error(f"상태 저장 실패: {e}")
            raise
    
    def load_state(self, path: str):
        """상태 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            self.current_regime = state_data['current_regime']
            self.previous_regime = state_data['previous_regime']
            self.regime_history = deque(state_data['regime_history'], maxlen=100)
            self.confidence_history = deque(state_data['confidence_history'], maxlen=100)
            self.regime_transitions = state_data['regime_transitions']
            self.confidence_rejections = state_data['confidence_rejections']
            self.total_predictions = state_data['total_predictions']
            self.confidence_threshold = state_data['confidence_threshold']
            
            self.logger.info(f"Confidence-based Selection 상태 로드: {path}")
            
        except Exception as e:
            self.logger.error(f"상태 로드 실패: {e}")
            raise


class RegimeUncertaintyManager:
    """
    체제 불확실성 관리자
    
    체제 전환 시 불확실성을 관리하고 안전장치 제공
    """
    
    def __init__(self, config: Dict):
        """
        체제 불확실성 관리자 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 불확실성 관리 파라미터
        self.uncertainty_threshold = 0.4  # 불확실성 임계값
        self.transition_cooldown = 5  # 체제 전환 쿨다운 (시간 단위)
        self.last_transition_time = 0
        
        # 안전장치 설정
        self.enable_safety_mechanisms = True
        self.max_transitions_per_hour = 3  # 시간당 최대 전환 횟수
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("체제 불확실성 관리자 초기화 완료")
    
    def assess_uncertainty(self, regime_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        체제 불확실성 평가
        
        Args:
            regime_probabilities: 체제별 확률 딕셔너리
            
        Returns:
            불확실성 평가 결과
        """
        try:
            if not regime_probabilities:
                return {'uncertainty_level': 1.0, 'is_uncertain': True}
            
            probabilities = list(regime_probabilities.values())
            
            # 엔트로피 기반 불확실성 계산
            entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
            max_entropy = np.log(len(probabilities))
            normalized_entropy = entropy / max_entropy
            
            # 최대 확률 기반 불확실성
            max_prob = max(probabilities)
            confidence_uncertainty = 1.0 - max_prob
            
            # 종합 불확실성 점수
            uncertainty_score = (normalized_entropy + confidence_uncertainty) / 2
            
            is_uncertain = uncertainty_score > self.uncertainty_threshold
            
            return {
                'uncertainty_level': float(uncertainty_score),
                'is_uncertain': is_uncertain,
                'entropy': float(normalized_entropy),
                'confidence_uncertainty': float(confidence_uncertainty)
            }
            
        except Exception as e:
            self.logger.error(f"불확실성 평가 실패: {e}")
            return {'uncertainty_level': 1.0, 'is_uncertain': True}
    
    def should_allow_transition(self, current_time: float, 
                              proposed_regime: str, 
                              current_regime: str) -> bool:
        """
        체제 전환 허용 여부 결정
        
        Args:
            current_time: 현재 시간
            proposed_regime: 제안된 체제
            current_regime: 현재 체제
            
        Returns:
            전환 허용 여부
        """
        try:
            if not self.enable_safety_mechanisms:
                return True
            
            # 동일한 체제로의 전환은 허용
            if proposed_regime == current_regime:
                return True
            
            # 쿨다운 기간 확인
            time_since_last_transition = current_time - self.last_transition_time
            if time_since_last_transition < self.transition_cooldown:
                self.logger.debug(f"체제 전환 쿨다운 중: {time_since_last_transition:.1f}초 < {self.transition_cooldown}초")
                return False
            
            # 시간당 전환 횟수 제한 (실제 구현에서는 더 정교한 로직 필요)
            # 여기서는 단순화된 구현
            
            return True
            
        except Exception as e:
            self.logger.error(f"전환 허용 여부 결정 실패: {e}")
            return False
    
    def apply_safety_mechanisms(self, regime_probabilities: Dict[str, float],
                               current_regime: str, 
                               current_time: float) -> Dict[str, float]:
        """
        안전장치 적용
        
        Args:
            regime_probabilities: 체제별 확률
            current_regime: 현재 체제
            current_time: 현재 시간
            
        Returns:
            안전장치가 적용된 체제별 확률
        """
        try:
            if not self.enable_safety_mechanisms:
                return regime_probabilities
            
            # 불확실성 평가
            uncertainty_assessment = self.assess_uncertainty(regime_probabilities)
            
            # 높은 불확실성 상황에서 현재 체제 유지
            if uncertainty_assessment['is_uncertain']:
                adjusted_probabilities = {regime: 0.0 for regime in regime_probabilities.keys()}
                adjusted_probabilities[current_regime] = 1.0
                
                self.logger.debug(f"높은 불확실성으로 현재 체제 유지: {current_regime}")
                return adjusted_probabilities
            
            # 전환 허용 여부 확인
            max_prob_regime = max(regime_probabilities, key=regime_probabilities.get)
            if not self.should_allow_transition(current_time, max_prob_regime, current_regime):
                adjusted_probabilities = {regime: 0.0 for regime in regime_probabilities.keys()}
                adjusted_probabilities[current_regime] = 1.0
                
                self.logger.debug(f"안전장치로 현재 체제 유지: {current_regime}")
                return adjusted_probabilities
            
            return regime_probabilities
            
        except Exception as e:
            self.logger.error(f"안전장치 적용 실패: {e}")
            return regime_probabilities


# 편의 함수들
def create_confidence_selection(config: Dict) -> ConfidenceBasedSelection:
    """Confidence-based Selection 생성 편의 함수"""
    return ConfidenceBasedSelection(config)


def create_uncertainty_manager(config: Dict) -> RegimeUncertaintyManager:
    """체제 불확실성 관리자 생성 편의 함수"""
    return RegimeUncertaintyManager(config)
