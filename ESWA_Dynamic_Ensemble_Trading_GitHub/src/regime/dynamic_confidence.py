"""
동적 신뢰도 임계값 조절 시스템
Dynamic Confidence Threshold Adjustment System

시장 상황과 모델 성능에 따른 동적 신뢰도 임계값 조절
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

logger = logging.getLogger(__name__)

class DynamicConfidenceThreshold:
    """동적 신뢰도 임계값 조절 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        동적 신뢰도 임계값 조절 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 신뢰도 임계값 파라미터
        self.base_confidence_threshold = config.get('base_confidence_threshold', 0.6)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        self.max_confidence_threshold = config.get('max_confidence_threshold', 0.9)
        self.confidence_adjustment_step = config.get('confidence_adjustment_step', 0.05)
        
        # 성능 기반 조절 파라미터
        self.performance_window = config.get('performance_window', 30)
        self.accuracy_threshold = config.get('accuracy_threshold', 0.7)
        self.precision_threshold = config.get('precision_threshold', 0.6)
        self.recall_threshold = config.get('recall_threshold', 0.6)
        
        # 시장 상황 기반 조절 파라미터
        self.volatility_threshold_high = config.get('volatility_threshold_high', 0.25)
        self.volatility_threshold_low = config.get('volatility_threshold_low', 0.10)
        self.regime_stability_threshold = config.get('regime_stability_threshold', 0.8)
        
        # 현재 신뢰도 임계값
        self.current_confidence_threshold = self.base_confidence_threshold
        
        # 성능 히스토리
        self.performance_history = []
        self.confidence_history = []
        self.adjustment_history = []
        
        logger.info(f"동적 신뢰도 임계값 조절 시스템 초기화 완료")
        logger.info(f"기본 신뢰도 임계값: {self.base_confidence_threshold}")
        logger.info(f"신뢰도 범위: [{self.min_confidence_threshold}, {self.max_confidence_threshold}]")
    
    def calculate_performance_based_threshold(self, recent_predictions: List[Dict[str, Any]],
                                            recent_actuals: List[str]) -> float:
        """
        성능 기반 신뢰도 임계값 계산
        
        Args:
            recent_predictions: 최근 예측 결과 (신뢰도 포함)
            recent_actuals: 최근 실제 값
            
        Returns:
            성능 기반 신뢰도 임계값
        """
        try:
            logger.info("성능 기반 신뢰도 임계값 계산 시작")
            
            if len(recent_predictions) < 10:
                return self.current_confidence_threshold
            
            # 예측 결과와 실제 값 정렬
            predictions_with_confidence = []
            for pred, actual in zip(recent_predictions, recent_actuals):
                if 'confidence' in pred and 'prediction' in pred:
                    predictions_with_confidence.append({
                        'prediction': pred['prediction'],
                        'confidence': pred['confidence'],
                        'actual': actual
                    })
            
            if len(predictions_with_confidence) < 10:
                return self.current_confidence_threshold
            
            # 신뢰도별 성능 분석
            confidence_levels = np.arange(0.1, 1.0, 0.1)
            performance_by_confidence = {}
            
            for conf_level in confidence_levels:
                # 해당 신뢰도 이상의 예측만 선택
                high_conf_predictions = [
                    p for p in predictions_with_confidence 
                    if p['confidence'] >= conf_level
                ]
                
                if len(high_conf_predictions) >= 5:  # 최소 5개 샘플 필요
                    # 정확도 계산
                    correct_predictions = sum(1 for p in high_conf_predictions 
                                            if p['prediction'] == p['actual'])
                    accuracy = correct_predictions / len(high_conf_predictions)
                    
                    performance_by_confidence[conf_level] = {
                        'accuracy': accuracy,
                        'sample_size': len(high_conf_predictions)
                    }
            
            # 최적 신뢰도 임계값 찾기 (정확도와 샘플 수의 균형)
            best_threshold = self.current_confidence_threshold
            best_score = -np.inf
            
            for conf_level, perf in performance_by_confidence.items():
                # 점수 = 정확도 * 샘플 수 가중치
                sample_weight = min(perf['sample_size'] / len(predictions_with_confidence), 1.0)
                score = perf['accuracy'] * sample_weight
                
                if score > best_score:
                    best_score = score
                    best_threshold = conf_level
            
            # 임계값 범위 제한
            adjusted_threshold = np.clip(best_threshold, self.min_confidence_threshold, self.max_confidence_threshold)
            
            logger.info(f"성능 기반 신뢰도 임계값 계산 완료: {adjusted_threshold:.3f}")
            
            return adjusted_threshold
            
        except Exception as e:
            logger.error(f"성능 기반 신뢰도 임계값 계산 실패: {e}")
            return self.current_confidence_threshold
    
    def calculate_market_condition_based_threshold(self, current_volatility: float,
                                                 regime_stability: float,
                                                 market_regime: str) -> float:
        """
        시장 상황 기반 신뢰도 임계값 계산
        
        Args:
            current_volatility: 현재 변동성
            regime_stability: 체제 안정성
            market_regime: 현재 시장 체제
            
        Returns:
            시장 상황 기반 신뢰도 임계값
        """
        try:
            logger.info("시장 상황 기반 신뢰도 임계값 계산 시작")
            
            # 기본 임계값
            threshold = self.base_confidence_threshold
            
            # 변동성 기반 조절
            if current_volatility > self.volatility_threshold_high:
                # 높은 변동성 -> 신뢰도 임계값 낮춤 (더 많은 예측 허용)
                threshold -= self.confidence_adjustment_step * 2
            elif current_volatility < self.volatility_threshold_low:
                # 낮은 변동성 -> 신뢰도 임계값 높임 (더 엄격한 예측)
                threshold += self.confidence_adjustment_step
            
            # 체제 안정성 기반 조절
            if regime_stability > self.regime_stability_threshold:
                # 안정적인 체제 -> 신뢰도 임계값 낮춤
                threshold -= self.confidence_adjustment_step
            else:
                # 불안정한 체제 -> 신뢰도 임계값 높임
                threshold += self.confidence_adjustment_step
            
            # 시장 체제별 조절
            regime_adjustments = {
                'bull': -self.confidence_adjustment_step * 0.5,  # 상승장에서는 약간 낮춤
                'bear': self.confidence_adjustment_step * 0.5,   # 하락장에서는 약간 높임
                'sideways': 0  # 횡보장에서는 변화 없음
            }
            
            if market_regime in regime_adjustments:
                threshold += regime_adjustments[market_regime]
            
            # 임계값 범위 제한
            adjusted_threshold = np.clip(threshold, self.min_confidence_threshold, self.max_confidence_threshold)
            
            logger.info(f"시장 상황 기반 신뢰도 임계값 계산 완료: {adjusted_threshold:.3f}")
            
            return adjusted_threshold
            
        except Exception as e:
            logger.error(f"시장 상황 기반 신뢰도 임계값 계산 실패: {e}")
            return self.current_confidence_threshold
    
    def calculate_adaptive_threshold(self, recent_predictions: List[Dict[str, Any]],
                                   recent_actuals: List[str],
                                   current_volatility: float,
                                   regime_stability: float,
                                   market_regime: str) -> float:
        """
        적응적 신뢰도 임계값 계산 (성능과 시장 상황을 종합적으로 고려)
        
        Args:
            recent_predictions: 최근 예측 결과
            recent_actuals: 최근 실제 값
            current_volatility: 현재 변동성
            regime_stability: 체제 안정성
            market_regime: 현재 시장 체제
            
        Returns:
            적응적 신뢰도 임계값
        """
        try:
            logger.info("적응적 신뢰도 임계값 계산 시작")
            
            # 성능 기반 임계값
            performance_threshold = self.calculate_performance_based_threshold(
                recent_predictions, recent_actuals
            )
            
            # 시장 상황 기반 임계값
            market_threshold = self.calculate_market_condition_based_threshold(
                current_volatility, regime_stability, market_regime
            )
            
            # 가중 평균 (성능 70%, 시장 상황 30%)
            adaptive_threshold = (performance_threshold * 0.7 + market_threshold * 0.3)
            
            # 현재 임계값과의 차이를 고려한 점진적 조절
            threshold_diff = adaptive_threshold - self.current_confidence_threshold
            
            # 점진적 조절 (급격한 변화 방지)
            max_change = self.confidence_adjustment_step * 2
            if abs(threshold_diff) > max_change:
                if threshold_diff > 0:
                    new_threshold = self.current_confidence_threshold + max_change
                else:
                    new_threshold = self.current_confidence_threshold - max_change
            else:
                new_threshold = adaptive_threshold
            
            # 임계값 범위 제한
            final_threshold = np.clip(new_threshold, self.min_confidence_threshold, self.max_confidence_threshold)
            
            logger.info(f"적응적 신뢰도 임계값 계산 완료: {final_threshold:.3f}")
            logger.info(f"  성능 기반: {performance_threshold:.3f}")
            logger.info(f"  시장 상황 기반: {market_threshold:.3f}")
            logger.info(f"  조절량: {final_threshold - self.current_confidence_threshold:.3f}")
            
            return final_threshold
            
        except Exception as e:
            logger.error(f"적응적 신뢰도 임계값 계산 실패: {e}")
            return self.current_confidence_threshold
    
    def update_confidence_threshold(self, new_threshold: float, 
                                  adjustment_reason: str = "manual") -> Dict[str, Any]:
        """
        신뢰도 임계값 업데이트
        
        Args:
            new_threshold: 새로운 임계값
            adjustment_reason: 조절 이유
            
        Returns:
            업데이트 결과
        """
        try:
            logger.info(f"신뢰도 임계값 업데이트: {self.current_confidence_threshold:.3f} -> {new_threshold:.3f}")
            
            old_threshold = self.current_confidence_threshold
            
            # 임계값 범위 확인
            if new_threshold < self.min_confidence_threshold:
                new_threshold = self.min_confidence_threshold
                logger.warning(f"신뢰도 임계값이 최소값으로 제한됨: {new_threshold}")
            elif new_threshold > self.max_confidence_threshold:
                new_threshold = self.max_confidence_threshold
                logger.warning(f"신뢰도 임계값이 최대값으로 제한됨: {new_threshold}")
            
            # 임계값 업데이트
            self.current_confidence_threshold = new_threshold
            
            # 히스토리 업데이트
            self.confidence_history.append({
                'timestamp': datetime.now().isoformat(),
                'old_threshold': old_threshold,
                'new_threshold': new_threshold,
                'adjustment_reason': adjustment_reason
            })
            
            self.adjustment_history.append({
                'timestamp': datetime.now().isoformat(),
                'threshold_change': new_threshold - old_threshold,
                'reason': adjustment_reason
            })
            
            result = {
                'success': True,
                'old_threshold': old_threshold,
                'new_threshold': new_threshold,
                'threshold_change': new_threshold - old_threshold,
                'adjustment_reason': adjustment_reason,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"신뢰도 임계값 업데이트 완료")
            
            return result
            
        except Exception as e:
            logger.error(f"신뢰도 임계값 업데이트 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    def evaluate_threshold_performance(self, predictions: List[Dict[str, Any]],
                                     actuals: List[str]) -> Dict[str, Any]:
        """
        임계값 성능 평가
        
        Args:
            predictions: 예측 결과 (신뢰도 포함)
            actuals: 실제 값
            
        Returns:
            임계값 성능 평가 결과
        """
        try:
            logger.info("임계값 성능 평가 시작")
            
            if len(predictions) != len(actuals) or len(predictions) == 0:
                return {'error': 'Invalid input data'}
            
            # 현재 임계값으로 필터링된 예측
            filtered_predictions = []
            filtered_actuals = []
            
            for pred, actual in zip(predictions, actuals):
                if 'confidence' in pred and pred['confidence'] >= self.current_confidence_threshold:
                    filtered_predictions.append(pred['prediction'])
                    filtered_actuals.append(actual)
            
            if len(filtered_predictions) == 0:
                return {
                    'error': 'No predictions above threshold',
                    'threshold': self.current_confidence_threshold,
                    'total_predictions': len(predictions),
                    'filtered_predictions': 0
                }
            
            # 성능 지표 계산
            accuracy = accuracy_score(filtered_actuals, filtered_predictions)
            precision = precision_score(filtered_actuals, filtered_predictions, average='weighted', zero_division=0)
            recall = recall_score(filtered_actuals, filtered_predictions, average='weighted', zero_division=0)
            f1 = f1_score(filtered_actuals, filtered_predictions, average='weighted', zero_division=0)
            
            # 전체 예측 대비 필터링된 예측 비율
            coverage = len(filtered_predictions) / len(predictions)
            
            result = {
                'threshold': self.current_confidence_threshold,
                'total_predictions': len(predictions),
                'filtered_predictions': len(filtered_predictions),
                'coverage': coverage,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'evaluation_date': datetime.now().isoformat()
            }
            
            # 성능 히스토리 업데이트
            self.performance_history.append(result)
            
            logger.info(f"임계값 성능 평가 완료")
            logger.info(f"  정확도: {accuracy:.4f}")
            logger.info(f"  커버리지: {coverage:.4f}")
            logger.info(f"  F1 점수: {f1:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"임계값 성능 평가 실패: {e}")
            return {'error': str(e)}
    
    def get_optimal_threshold(self, predictions: List[Dict[str, Any]],
                            actuals: List[str]) -> Dict[str, Any]:
        """
        최적 신뢰도 임계값 찾기
        
        Args:
            predictions: 예측 결과 (신뢰도 포함)
            actuals: 실제 값
            
        Returns:
            최적 임계값 결과
        """
        try:
            logger.info("최적 신뢰도 임계값 찾기 시작")
            
            if len(predictions) != len(actuals) or len(predictions) == 0:
                return {'error': 'Invalid input data'}
            
            # 신뢰도 임계값 후보들
            threshold_candidates = np.arange(0.1, 1.0, 0.05)
            
            best_threshold = self.current_confidence_threshold
            best_score = -np.inf
            threshold_results = []
            
            for threshold in threshold_candidates:
                # 해당 임계값으로 필터링
                filtered_predictions = []
                filtered_actuals = []
                
                for pred, actual in zip(predictions, actuals):
                    if 'confidence' in pred and pred['confidence'] >= threshold:
                        filtered_predictions.append(pred['prediction'])
                        filtered_actuals.append(actual)
                
                if len(filtered_predictions) < 5:  # 최소 샘플 수
                    continue
                
                # 성능 지표 계산
                accuracy = accuracy_score(filtered_actuals, filtered_predictions)
                coverage = len(filtered_predictions) / len(predictions)
                
                # 종합 점수 (정확도 * 커버리지)
                score = accuracy * coverage
                
                threshold_results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'score': score,
                    'sample_size': len(filtered_predictions)
                })
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            result = {
                'optimal_threshold': best_threshold,
                'best_score': best_score,
                'threshold_results': threshold_results,
                'current_threshold': self.current_confidence_threshold,
                'improvement_potential': best_score - (self.current_confidence_threshold * 0.5)  # 근사치
            }
            
            logger.info(f"최적 신뢰도 임계값 찾기 완료")
            logger.info(f"  최적 임계값: {best_threshold:.3f}")
            logger.info(f"  최고 점수: {best_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"최적 신뢰도 임계값 찾기 실패: {e}")
            return {'error': str(e)}
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """신뢰도 임계값 요약 정보 반환"""
        try:
            summary = {
                'current_threshold': self.current_confidence_threshold,
                'base_threshold': self.base_confidence_threshold,
                'min_threshold': self.min_confidence_threshold,
                'max_threshold': self.max_confidence_threshold,
                'total_adjustments': len(self.adjustment_history),
                'recent_performance': None,
                'threshold_trend': 'stable'
            }
            
            # 최근 성능
            if self.performance_history:
                recent_perf = self.performance_history[-1]
                summary['recent_performance'] = {
                    'accuracy': recent_perf.get('accuracy', 0),
                    'coverage': recent_perf.get('coverage', 0),
                    'f1_score': recent_perf.get('f1_score', 0)
                }
            
            # 임계값 트렌드
            if len(self.confidence_history) >= 3:
                recent_thresholds = [h['new_threshold'] for h in self.confidence_history[-3:]]
                if recent_thresholds[-1] > recent_thresholds[0]:
                    summary['threshold_trend'] = 'increasing'
                elif recent_thresholds[-1] < recent_thresholds[0]:
                    summary['threshold_trend'] = 'decreasing'
            
            return summary
            
        except Exception as e:
            logger.error(f"신뢰도 임계값 요약 생성 실패: {e}")
            return {'error': str(e)}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("동적 신뢰도 임계값 조절 시스템 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {
            'base_confidence_threshold': 0.6,
            'min_confidence_threshold': 0.3,
            'max_confidence_threshold': 0.9,
            'confidence_adjustment_step': 0.05,
            'performance_window': 30,
            'accuracy_threshold': 0.7,
            'volatility_threshold_high': 0.25,
            'volatility_threshold_low': 0.10,
            'regime_stability_threshold': 0.8
        }
        
        # 동적 신뢰도 임계값 조절 시스템 생성
        confidence_system = DynamicConfidenceThreshold(config)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        # 예측 결과 생성 (신뢰도 포함)
        predictions = []
        actuals = []
        
        for i in range(n_samples):
            confidence = np.random.uniform(0.3, 0.95)
            prediction = np.random.choice(['bull', 'bear', 'sideways'])
            actual = np.random.choice(['bull', 'bear', 'sideways'])
            
            predictions.append({
                'prediction': prediction,
                'confidence': confidence
            })
            actuals.append(actual)
        
        print(f"테스트 데이터: {len(predictions)}개 샘플")
        print(f"현재 신뢰도 임계값: {confidence_system.current_confidence_threshold:.3f}")
        
        # 성능 기반 임계값 계산
        performance_threshold = confidence_system.calculate_performance_based_threshold(
            predictions, actuals
        )
        print(f"성능 기반 임계값: {performance_threshold:.3f}")
        
        # 시장 상황 기반 임계값 계산
        market_threshold = confidence_system.calculate_market_condition_based_threshold(
            current_volatility=0.15,
            regime_stability=0.7,
            market_regime='sideways'
        )
        print(f"시장 상황 기반 임계값: {market_threshold:.3f}")
        
        # 적응적 임계값 계산
        adaptive_threshold = confidence_system.calculate_adaptive_threshold(
            predictions, actuals, 0.15, 0.7, 'sideways'
        )
        print(f"적응적 임계값: {adaptive_threshold:.3f}")
        
        # 임계값 업데이트
        update_result = confidence_system.update_confidence_threshold(
            adaptive_threshold, "adaptive_adjustment"
        )
        if update_result['success']:
            print(f"임계값 업데이트 성공: {update_result['threshold_change']:.3f}")
        
        # 성능 평가
        performance_result = confidence_system.evaluate_threshold_performance(
            predictions, actuals
        )
        if 'error' not in performance_result:
            print(f"성능 평가:")
            print(f"  정확도: {performance_result['accuracy']:.4f}")
            print(f"  커버리지: {performance_result['coverage']:.4f}")
            print(f"  F1 점수: {performance_result['f1_score']:.4f}")
        
        # 최적 임계값 찾기
        optimal_result = confidence_system.get_optimal_threshold(predictions, actuals)
        if 'error' not in optimal_result:
            print(f"최적 임계값: {optimal_result['optimal_threshold']:.3f}")
            print(f"최고 점수: {optimal_result['best_score']:.4f}")
        
        # 요약 정보
        summary = confidence_system.get_threshold_summary()
        print(f"\n요약 정보:")
        print(f"  현재 임계값: {summary['current_threshold']:.3f}")
        print(f"  총 조절 횟수: {summary['total_adjustments']}")
        print(f"  임계값 트렌드: {summary['threshold_trend']}")
        
        print(f"\n[성공] 동적 신뢰도 임계값 조절 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 동적 신뢰도 임계값 조절 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
