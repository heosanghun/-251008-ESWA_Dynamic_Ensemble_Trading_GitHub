"""
고급 앙상블 가중치 할당 알고리즘
Advanced Ensemble Weight Allocation Algorithm

동적 가중치 할당을 통한 앙상블 성능 최적화
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class AdvancedWeightAllocation:
    """고급 앙상블 가중치 할당 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        고급 가중치 할당 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 가중치 할당 파라미터
        self.min_weight = config.get('min_weight', 0.01)
        self.max_weight = config.get('max_weight', 0.5)
        self.temperature = config.get('temperature', 10.0)
        self.evaluation_period = config.get('evaluation_period', 30)
        
        # 가중치 할당 방법
        self.weight_allocation_methods = [
            'performance_based',
            'confidence_based',
            'diversity_based',
            'risk_adjusted',
            'dynamic_adaptive'
        ]
        
        # 성능 추적
        self.performance_history = {}
        self.weight_history = {}
        
        logger.info(f"고급 앙상블 가중치 할당 시스템 초기화 완료")
        logger.info(f"최소 가중치: {self.min_weight}")
        logger.info(f"최대 가중치: {self.max_weight}")
        logger.info(f"온도 파라미터: {self.temperature}")
    
    def calculate_performance_based_weights(self, agent_performances: Dict[str, float],
                                          lookback_period: int = 30) -> Dict[str, float]:
        """
        성능 기반 가중치 계산
        
        Args:
            agent_performances: 에이전트별 성능 딕셔너리
            lookback_period: 성능 평가 기간
            
        Returns:
            성능 기반 가중치
        """
        try:
            logger.info("성능 기반 가중치 계산 시작")
            
            if not agent_performances:
                return {}
            
            # 성능 정규화 (0-1 범위)
            performances = np.array(list(agent_performances.values()))
            min_perf = np.min(performances)
            max_perf = np.max(performances)
            
            if max_perf == min_perf:
                # 모든 성능이 동일한 경우 균등 가중치
                num_agents = len(agent_performances)
                return {agent: 1.0 / num_agents for agent in agent_performances.keys()}
            
            # 정규화된 성능
            normalized_performances = (performances - min_perf) / (max_perf - min_perf)
            
            # 소프트맥스 적용 (온도 파라미터 사용)
            exp_performances = np.exp(normalized_performances * self.temperature)
            softmax_weights = exp_performances / np.sum(exp_performances)
            
            # 가중치 범위 제한
            weights = {}
            for i, agent in enumerate(agent_performances.keys()):
                weight = np.clip(softmax_weights[i], self.min_weight, self.max_weight)
                weights[agent] = weight
            
            # 가중치 정규화 (합이 1이 되도록)
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {agent: weight / total_weight for agent, weight in weights.items()}
            
            logger.info(f"성능 기반 가중치 계산 완료")
            for agent, weight in weights.items():
                logger.info(f"  {agent}: {weight:.4f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"성능 기반 가중치 계산 실패: {e}")
            return {}
    
    def calculate_confidence_based_weights(self, agent_confidences: Dict[str, float]) -> Dict[str, float]:
        """
        신뢰도 기반 가중치 계산
        
        Args:
            agent_confidences: 에이전트별 신뢰도 딕셔너리
            
        Returns:
            신뢰도 기반 가중치
        """
        try:
            logger.info("신뢰도 기반 가중치 계산 시작")
            
            if not agent_confidences:
                return {}
            
            # 신뢰도 정규화
            confidences = np.array(list(agent_confidences.values()))
            
            # 신뢰도 기반 가중치 계산 (신뢰도가 높을수록 높은 가중치)
            confidence_weights = confidences ** 2  # 제곱하여 차이 확대
            
            # 가중치 범위 제한
            weights = {}
            for i, agent in enumerate(agent_confidences.keys()):
                weight = np.clip(confidence_weights[i], self.min_weight, self.max_weight)
                weights[agent] = weight
            
            # 가중치 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {agent: weight / total_weight for agent, weight in weights.items()}
            
            logger.info(f"신뢰도 기반 가중치 계산 완료")
            for agent, weight in weights.items():
                logger.info(f"  {agent}: {weight:.4f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"신뢰도 기반 가중치 계산 실패: {e}")
            return {}
    
    def calculate_diversity_based_weights(self, agent_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        다양성 기반 가중치 계산 (예측 다양성을 고려)
        
        Args:
            agent_predictions: 에이전트별 예측 결과
            
        Returns:
            다양성 기반 가중치
        """
        try:
            logger.info("다양성 기반 가중치 계산 시작")
            
            if not agent_predictions:
                return {}
            
            agents = list(agent_predictions.keys())
            num_agents = len(agents)
            
            if num_agents < 2:
                return {agents[0]: 1.0} if agents else {}
            
            # 예측 다양성 계산
            predictions_matrix = np.array([agent_predictions[agent] for agent in agents])
            
            # 각 에이전트의 다른 에이전트들과의 평균 거리 계산
            diversity_scores = []
            for i in range(num_agents):
                distances = []
                for j in range(num_agents):
                    if i != j:
                        # 예측 간 거리 계산 (유클리드 거리)
                        distance = np.linalg.norm(predictions_matrix[i] - predictions_matrix[j])
                        distances.append(distance)
                diversity_scores.append(np.mean(distances))
            
            # 다양성 점수를 가중치로 변환 (다양성이 높을수록 높은 가중치)
            diversity_weights = np.array(diversity_scores)
            
            # 가중치 범위 제한
            weights = {}
            for i, agent in enumerate(agents):
                weight = np.clip(diversity_weights[i], self.min_weight, self.max_weight)
                weights[agent] = weight
            
            # 가중치 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {agent: weight / total_weight for agent, weight in weights.items()}
            
            logger.info(f"다양성 기반 가중치 계산 완료")
            for agent, weight in weights.items():
                logger.info(f"  {agent}: {weight:.4f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"다양성 기반 가중치 계산 실패: {e}")
            return {}
    
    def calculate_risk_adjusted_weights(self, agent_performances: Dict[str, float],
                                      agent_risks: Dict[str, float]) -> Dict[str, float]:
        """
        리스크 조정 가중치 계산 (성능/리스크 비율 기반)
        
        Args:
            agent_performances: 에이전트별 성능
            agent_risks: 에이전트별 리스크
            
        Returns:
            리스크 조정 가중치
        """
        try:
            logger.info("리스크 조정 가중치 계산 시작")
            
            if not agent_performances or not agent_risks:
                return {}
            
            # 성능/리스크 비율 계산
            risk_adjusted_scores = {}
            for agent in agent_performances.keys():
                if agent in agent_risks and agent_risks[agent] > 0:
                    risk_adjusted_scores[agent] = agent_performances[agent] / agent_risks[agent]
                else:
                    risk_adjusted_scores[agent] = 0.0
            
            # 리스크 조정 점수를 가중치로 변환
            scores = np.array(list(risk_adjusted_scores.values()))
            
            if np.sum(scores) == 0:
                # 모든 점수가 0인 경우 균등 가중치
                num_agents = len(risk_adjusted_scores)
                return {agent: 1.0 / num_agents for agent in risk_adjusted_scores.keys()}
            
            # 소프트맥스 적용
            exp_scores = np.exp(scores * self.temperature)
            softmax_weights = exp_scores / np.sum(exp_scores)
            
            # 가중치 범위 제한
            weights = {}
            for i, agent in enumerate(risk_adjusted_scores.keys()):
                weight = np.clip(softmax_weights[i], self.min_weight, self.max_weight)
                weights[agent] = weight
            
            # 가중치 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {agent: weight / total_weight for agent, weight in weights.items()}
            
            logger.info(f"리스크 조정 가중치 계산 완료")
            for agent, weight in weights.items():
                logger.info(f"  {agent}: {weight:.4f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"리스크 조정 가중치 계산 실패: {e}")
            return {}
    
    def calculate_dynamic_adaptive_weights(self, agent_performances: Dict[str, float],
                                         agent_confidences: Dict[str, float],
                                         market_regime: str,
                                         historical_weights: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
        """
        동적 적응 가중치 계산 (여러 요소를 종합적으로 고려)
        
        Args:
            agent_performances: 에이전트별 성능
            agent_confidences: 에이전트별 신뢰도
            market_regime: 현재 시장 체제
            historical_weights: 과거 가중치 히스토리
            
        Returns:
            동적 적응 가중치
        """
        try:
            logger.info("동적 적응 가중치 계산 시작")
            
            if not agent_performances:
                return {}
            
            # 각 방법별 가중치 계산
            performance_weights = self.calculate_performance_based_weights(agent_performances)
            confidence_weights = self.calculate_confidence_based_weights(agent_confidences)
            
            # 시장 체제별 가중치 조정
            regime_weights = self._get_regime_specific_weights(market_regime, agent_performances)
            
            # 과거 가중치 기반 적응성 조정
            adaptation_weights = self._calculate_adaptation_weights(historical_weights, agent_performances)
            
            # 종합 가중치 계산 (가중 평균)
            method_weights = {
                'performance': 0.4,
                'confidence': 0.3,
                'regime': 0.2,
                'adaptation': 0.1
            }
            
            final_weights = {}
            for agent in agent_performances.keys():
                weight = 0.0
                weight += performance_weights.get(agent, 0) * method_weights['performance']
                weight += confidence_weights.get(agent, 0) * method_weights['confidence']
                weight += regime_weights.get(agent, 0) * method_weights['regime']
                weight += adaptation_weights.get(agent, 0) * method_weights['adaptation']
                
                final_weights[agent] = weight
            
            # 가중치 범위 제한 및 정규화
            final_weights = {agent: np.clip(weight, self.min_weight, self.max_weight) 
                           for agent, weight in final_weights.items()}
            
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                final_weights = {agent: weight / total_weight for agent, weight in final_weights.items()}
            
            logger.info(f"동적 적응 가중치 계산 완료")
            for agent, weight in final_weights.items():
                logger.info(f"  {agent}: {weight:.4f}")
            
            return final_weights
            
        except Exception as e:
            logger.error(f"동적 적응 가중치 계산 실패: {e}")
            return {}
    
    def _get_regime_specific_weights(self, market_regime: str, 
                                   agent_performances: Dict[str, float]) -> Dict[str, float]:
        """시장 체제별 특화 가중치 계산"""
        try:
            # 시장 체제별 에이전트 성능 가중치
            regime_weights = {
                'bull': {'momentum_agent': 1.2, 'trend_agent': 1.1, 'mean_reversion_agent': 0.8},
                'bear': {'momentum_agent': 0.8, 'trend_agent': 0.9, 'mean_reversion_agent': 1.2},
                'sideways': {'momentum_agent': 0.9, 'trend_agent': 0.8, 'mean_reversion_agent': 1.3}
            }
            
            # 기본 가중치 (모든 에이전트 동일)
            base_weights = {agent: 1.0 for agent in agent_performances.keys()}
            
            # 시장 체제별 가중치 적용
            if market_regime in regime_weights:
                regime_specific = regime_weights[market_regime]
                for agent in agent_performances.keys():
                    if agent in regime_specific:
                        base_weights[agent] = regime_specific[agent]
            
            # 정규화
            total_weight = sum(base_weights.values())
            if total_weight > 0:
                base_weights = {agent: weight / total_weight for agent, weight in base_weights.items()}
            
            return base_weights
            
        except Exception as e:
            logger.error(f"시장 체제별 가중치 계산 실패: {e}")
            return {agent: 1.0 / len(agent_performances) for agent in agent_performances.keys()}
    
    def _calculate_adaptation_weights(self, historical_weights: Optional[Dict[str, List[float]]],
                                    agent_performances: Dict[str, float]) -> Dict[str, float]:
        """적응성 가중치 계산 (과거 가중치 기반)"""
        try:
            if not historical_weights:
                # 과거 데이터가 없는 경우 균등 가중치
                num_agents = len(agent_performances)
                return {agent: 1.0 / num_agents for agent in agent_performances.keys()}
            
            # 과거 가중치의 평균과 최근 트렌드 고려
            adaptation_weights = {}
            for agent in agent_performances.keys():
                if agent in historical_weights and len(historical_weights[agent]) > 0:
                    weights_history = historical_weights[agent]
                    
                    # 최근 가중치 평균
                    recent_weights = weights_history[-min(10, len(weights_history)):]
                    avg_weight = np.mean(recent_weights)
                    
                    # 가중치 트렌드 (최근 5개 vs 이전 5개)
                    if len(weights_history) >= 10:
                        recent_trend = np.mean(weights_history[-5:]) - np.mean(weights_history[-10:-5])
                        trend_factor = 1.0 + recent_trend * 0.5  # 트렌드 반영
                    else:
                        trend_factor = 1.0
                    
                    adaptation_weights[agent] = avg_weight * trend_factor
                else:
                    adaptation_weights[agent] = 1.0 / len(agent_performances)
            
            # 정규화
            total_weight = sum(adaptation_weights.values())
            if total_weight > 0:
                adaptation_weights = {agent: weight / total_weight for agent, weight in adaptation_weights.items()}
            
            return adaptation_weights
            
        except Exception as e:
            logger.error(f"적응성 가중치 계산 실패: {e}")
            return {agent: 1.0 / len(agent_performances) for agent in agent_performances.keys()}
    
    def optimize_weights_using_optimization(self, agent_performances: Dict[str, float],
                                          agent_confidences: Dict[str, float],
                                          objective_function: str = 'sharpe_ratio') -> Dict[str, float]:
        """
        최적화 알고리즘을 사용한 가중치 최적화
        
        Args:
            agent_performances: 에이전트별 성능
            agent_confidences: 에이전트별 신뢰도
            objective_function: 최적화 목적 함수
            
        Returns:
            최적화된 가중치
        """
        try:
            logger.info(f"최적화 알고리즘을 사용한 가중치 최적화 시작 (목적 함수: {objective_function})")
            
            if not agent_performances:
                return {}
            
            agents = list(agent_performances.keys())
            num_agents = len(agents)
            
            # 목적 함수 정의
            def objective(weights):
                # 가중치 제약 조건 확인
                if np.any(weights < self.min_weight) or np.any(weights > self.max_weight):
                    return -np.inf
                
                # 가중치 합이 1이 되도록 정규화
                weights = weights / np.sum(weights)
                
                # 목적 함수 계산 (예시: 샤프 비율)
                if objective_function == 'sharpe_ratio':
                    # 가중 평균 성능 / 가중 평균 리스크
                    weighted_performance = np.sum([weights[i] * agent_performances[agents[i]] for i in range(num_agents)])
                    weighted_risk = np.sum([weights[i] * (1 - agent_confidences.get(agents[i], 0.5)) for i in range(num_agents)])
                    
                    if weighted_risk > 0:
                        return weighted_performance / weighted_risk
                    else:
                        return weighted_performance
                else:
                    # 기본: 가중 평균 성능
                    return np.sum([weights[i] * agent_performances[agents[i]] for i in range(num_agents)])
            
            # 초기 가중치 (균등 분배)
            initial_weights = np.ones(num_agents) / num_agents
            
            # 제약 조건: 가중치 합 = 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # 경계 조건: 최소/최대 가중치
            bounds = [(self.min_weight, self.max_weight) for _ in range(num_agents)]
            
            # 최적화 수행
            result = minimize(
                lambda w: -objective(w),  # 최대화를 위해 음수 사용
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_weights = optimal_weights / np.sum(optimal_weights)  # 정규화
                
                weights = {agents[i]: optimal_weights[i] for i in range(num_agents)}
                
                logger.info(f"최적화 완료")
                logger.info(f"최적 목적 함수 값: {-result.fun:.4f}")
                for agent, weight in weights.items():
                    logger.info(f"  {agent}: {weight:.4f}")
                
                return weights
            else:
                logger.warning(f"최적화 실패: {result.message}")
                # 실패 시 성능 기반 가중치 반환
                return self.calculate_performance_based_weights(agent_performances)
                
        except Exception as e:
            logger.error(f"최적화 알고리즘을 사용한 가중치 최적화 실패: {e}")
            return self.calculate_performance_based_weights(agent_performances)
    
    def get_comprehensive_weights(self, agent_performances: Dict[str, float],
                                agent_confidences: Dict[str, float],
                                market_regime: str = 'sideways',
                                method: str = 'dynamic_adaptive') -> Dict[str, Any]:
        """
        종합 가중치 계산 (모든 방법 통합)
        
        Args:
            agent_performances: 에이전트별 성능
            agent_confidences: 에이전트별 신뢰도
            market_regime: 현재 시장 체제
            method: 사용할 방법
            
        Returns:
            종합 가중치 결과
        """
        try:
            logger.info(f"종합 가중치 계산 시작 (방법: {method})")
            
            if not agent_performances:
                return {'error': 'No agent performances provided'}
            
            # 선택된 방법으로 가중치 계산
            if method == 'performance_based':
                weights = self.calculate_performance_based_weights(agent_performances)
            elif method == 'confidence_based':
                weights = self.calculate_confidence_based_weights(agent_confidences)
            elif method == 'risk_adjusted':
                # 리스크는 신뢰도의 역수로 근사
                agent_risks = {agent: 1 - conf for agent, conf in agent_confidences.items()}
                weights = self.calculate_risk_adjusted_weights(agent_performances, agent_risks)
            elif method == 'dynamic_adaptive':
                weights = self.calculate_dynamic_adaptive_weights(
                    agent_performances, agent_confidences, market_regime
                )
            elif method == 'optimization':
                weights = self.optimize_weights_using_optimization(
                    agent_performances, agent_confidences
                )
            else:
                # 기본: 성능 기반
                weights = self.calculate_performance_based_weights(agent_performances)
            
            # 가중치 히스토리 업데이트
            for agent, weight in weights.items():
                if agent not in self.weight_history:
                    self.weight_history[agent] = []
                self.weight_history[agent].append(weight)
            
            result = {
                'weights': weights,
                'method': method,
                'market_regime': market_regime,
                'calculation_date': datetime.now().isoformat(),
                'num_agents': len(agent_performances),
                'weight_sum': sum(weights.values()),
                'weight_statistics': self._calculate_weight_statistics(weights)
            }
            
            logger.info(f"종합 가중치 계산 완료")
            logger.info(f"가중치 합: {sum(weights.values()):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"종합 가중치 계산 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_weight_statistics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 통계 계산"""
        try:
            weight_values = list(weights.values())
            
            return {
                'mean': np.mean(weight_values),
                'std': np.std(weight_values),
                'min': np.min(weight_values),
                'max': np.max(weight_values),
                'entropy': -np.sum([w * np.log(w) if w > 0 else 0 for w in weight_values])
            }
            
        except Exception as e:
            logger.error(f"가중치 통계 계산 실패: {e}")
            return {}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("고급 앙상블 가중치 할당 알고리즘 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {
            'min_weight': 0.01,
            'max_weight': 0.5,
            'temperature': 10.0,
            'evaluation_period': 30
        }
        
        # 테스트 데이터 생성
        agent_performances = {
            'agent_1': 0.15,  # 15% 수익률
            'agent_2': 0.12,  # 12% 수익률
            'agent_3': 0.18,  # 18% 수익률
            'agent_4': 0.10   # 10% 수익률
        }
        
        agent_confidences = {
            'agent_1': 0.8,   # 80% 신뢰도
            'agent_2': 0.9,   # 90% 신뢰도
            'agent_3': 0.7,   # 70% 신뢰도
            'agent_4': 0.85   # 85% 신뢰도
        }
        
        # 고급 가중치 할당 시스템 생성
        weight_allocator = AdvancedWeightAllocation(config)
        
        # 각 방법별 가중치 계산
        methods = ['performance_based', 'confidence_based', 'risk_adjusted', 'dynamic_adaptive', 'optimization']
        
        print(f"에이전트 성능:")
        for agent, perf in agent_performances.items():
            print(f"  {agent}: {perf:.2%}")
        
        print(f"\n에이전트 신뢰도:")
        for agent, conf in agent_confidences.items():
            print(f"  {agent}: {conf:.2%}")
        
        print(f"\n각 방법별 가중치:")
        for method in methods:
            result = weight_allocator.get_comprehensive_weights(
                agent_performances, agent_confidences, 'sideways', method
            )
            
            if 'error' not in result:
                weights = result['weights']
                print(f"\n{method}:")
                for agent, weight in weights.items():
                    print(f"  {agent}: {weight:.4f} ({weight*100:.2f}%)")
                
                stats = result['weight_statistics']
                print(f"  통계: 평균={stats['mean']:.4f}, 표준편차={stats['std']:.4f}, 엔트로피={stats['entropy']:.4f}")
        
        print(f"\n[성공] 고급 앙상블 가중치 할당 알고리즘 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 고급 앙상블 가중치 할당 알고리즘 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
