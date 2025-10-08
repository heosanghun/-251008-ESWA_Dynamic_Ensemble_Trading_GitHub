"""
에이전트 풀 관리 모듈
Agent Pool Management Module

체제별 전문 에이전트 풀 관리 및 훈련
- 각 체제별 5개 PPO 에이전트 (다양성 확보)
- 체제별 맞춤형 보상함수 적용
- 에이전트 풀 훈련 및 성과 관리
- 동적 가중치 할당을 위한 성과 추적

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict, deque
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .ppo_agent import PPOAgent, create_ppo_agent
from .reward_functions import RegimeSpecificRewards, MarketRegime, create_regime_specific_rewards


class AgentPool:
    """
    에이전트 풀 관리자
    
    체제별 전문 에이전트 풀을 관리하고 훈련
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        에이전트 풀 초기화
        
        Args:
            config: 설정 딕셔너리
            device: 연산 장치
        """
        self.config = config
        self.device = device
        
        # 에이전트 풀 설정
        self.agents_config = config.get('agents', {})
        self.pool_config = self.agents_config.get('pool', {})
        self.agents_per_regime = self.pool_config.get('agents_per_regime', 5)
        self.diversity_seeds = self.pool_config.get('diversity_seeds', [42, 123, 456, 789, 999])
        
        # 체제별 에이전트 풀
        self.agent_pools = {}
        self.reward_functions = {}
        
        # 성과 추적
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.recent_performance = defaultdict(lambda: deque(maxlen=30))  # 30일 성과
        
        # 훈련 상태
        self.is_trained = False
        self.training_progress = {}
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 에이전트 풀 초기화
        self._initialize_agent_pools()
        
        self.logger.info("에이전트 풀 관리자 초기화 완료")
        self.logger.info(f"체제별 에이전트 수: {self.agents_per_regime}")
    
    def _initialize_agent_pools(self):
        """체제별 에이전트 풀 초기화"""
        try:
            for regime in MarketRegime:
                regime_name = regime.value
                
                # 체제별 보상함수 초기화
                self.reward_functions[regime_name] = create_regime_specific_rewards(self.config)
                self.reward_functions[regime_name].set_regime(regime)
                
                # 체제별 에이전트 풀 초기화
                agent_pool = []
                for i in range(self.agents_per_regime):
                    agent_id = f"{regime_name}_agent_{i}"
                    seed = self.diversity_seeds[i] if i < len(self.diversity_seeds) else 42 + i
                    
                    agent = create_ppo_agent(
                        config=self.config,
                        agent_id=agent_id,
                        device=self.device,
                        seed=seed
                    )
                    agent_pool.append(agent)
                
                self.agent_pools[regime_name] = agent_pool
                self.training_progress[regime_name] = {
                    'trained_agents': 0,
                    'total_agents': self.agents_per_regime,
                    'training_completed': False
                }
                
                self.logger.info(f"{regime_name} 에이전트 풀 초기화: {len(agent_pool)}개 에이전트")
            
        except Exception as e:
            self.logger.error(f"에이전트 풀 초기화 실패: {e}")
            raise
    
    def train_regime_agents(self, regime: Union[str, MarketRegime], 
                           training_data: List[Dict], episodes: int = 1000) -> Dict[str, float]:
        """
        특정 체제의 에이전트 풀 훈련
        
        Args:
            regime: 훈련할 체제
            training_data: 훈련 데이터
            episodes: 훈련 에피소드 수
            
        Returns:
            훈련 성과 지표
        """
        try:
            if isinstance(regime, MarketRegime):
                regime_name = regime.value
            else:
                regime_name = regime
            
            if regime_name not in self.agent_pools:
                raise ValueError(f"지원하지 않는 체제: {regime_name}")
            
            self.logger.info(f"{regime_name} 에이전트 풀 훈련 시작: {episodes} 에피소드")
            
            # 보상함수 설정
            reward_function = self.reward_functions[regime_name]
            
            # 에이전트별 훈련 성과
            agent_performances = {}
            
            # 각 에이전트 훈련
            for i, agent in enumerate(self.agent_pools[regime_name]):
                self.logger.info(f"{regime_name} 에이전트 {i+1}/{self.agents_per_regime} 훈련 시작")
                
                agent_performance = self._train_single_agent(
                    agent=agent,
                    reward_function=reward_function,
                    training_data=training_data,
                    episodes=episodes
                )
                
                agent_performances[f"agent_{i}"] = agent_performance
                
                # 훈련 진행 상황 업데이트
                self.training_progress[regime_name]['trained_agents'] += 1
                
                self.logger.info(f"{regime_name} 에이전트 {i+1} 훈련 완료")
            
            # 훈련 완료 표시
            self.training_progress[regime_name]['training_completed'] = True
            
            # 전체 성과 계산
            regime_performance = self._calculate_regime_performance(agent_performances)
            
            self.logger.info(f"{regime_name} 에이전트 풀 훈련 완료")
            self.logger.info(f"평균 성과: {regime_performance.get('average_reward', 0):.4f}")
            
            return regime_performance
            
        except Exception as e:
            self.logger.error(f"{regime_name} 에이전트 풀 훈련 실패: {e}")
            raise
    
    def _train_single_agent(self, agent: PPOAgent, reward_function: RegimeSpecificRewards,
                           training_data: List[Dict], episodes: int) -> Dict[str, float]:
        """
        단일 에이전트 훈련
        
        Args:
            agent: 훈련할 에이전트
            reward_function: 보상함수
            training_data: 훈련 데이터
            episodes: 훈련 에피소드 수
            
        Returns:
            에이전트 성과 지표
        """
        try:
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(episodes):
                # 에피소드 데이터 준비
                episode_data = self._prepare_episode_data(training_data, episode)
                
                # 에피소드 실행
                episode_reward, episode_length = self._run_episode(
                    agent=agent,
                    reward_function=reward_function,
                    episode_data=episode_data
                )
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 주기적 로깅
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    self.logger.debug(f"{agent.agent_id} 에피소드 {episode+1}: 평균 보상 {avg_reward:.4f}")
            
            # 성과 지표 계산
            performance = {
                'average_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'best_reward': np.max(episode_rewards),
                'worst_reward': np.min(episode_rewards),
                'average_length': np.mean(episode_lengths),
                'total_episodes': episodes
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"단일 에이전트 훈련 실패: {e}")
            return {}
    
    def _prepare_episode_data(self, training_data: List[Dict], episode: int) -> List[Dict]:
        """
        에피소드 데이터 준비
        
        Args:
            training_data: 전체 훈련 데이터
            episode: 에피소드 번호
            
        Returns:
            에피소드 데이터
        """
        try:
            # 에피소드별로 다른 데이터 샘플링 (데이터 다양성 확보)
            episode_size = min(100, len(training_data))  # 에피소드당 최대 100개 스텝
            start_idx = (episode * episode_size) % (len(training_data) - episode_size)
            end_idx = start_idx + episode_size
            
            return training_data[start_idx:end_idx]
            
        except Exception as e:
            self.logger.error(f"에피소드 데이터 준비 실패: {e}")
            return training_data[:100] if len(training_data) > 100 else training_data
    
    def _run_episode(self, agent: PPOAgent, reward_function: RegimeSpecificRewards,
                    episode_data: List[Dict]) -> Tuple[float, int]:
        """
        에피소드 실행
        
        Args:
            agent: 실행할 에이전트
            reward_function: 보상함수
            episode_data: 에피소드 데이터
            
        Returns:
            에피소드 보상, 에피소드 길이
        """
        try:
            episode_reward = 0.0
            episode_steps = []
            
            for step_data in episode_data:
                # 상태 추출
                state = step_data.get('state', np.zeros(273))
                
                # 액션 선택
                action, log_prob, value = agent.get_action(state)
                
                # 다음 상태 및 보상 계산
                next_state = step_data.get('next_state', state)
                reward = reward_function.calculate_reward(
                    state=step_data,
                    action=action,
                    next_state=step_data,
                    done=step_data.get('done', False)
                )
                
                # 경험 저장
                agent.store_experience(state, action, reward, value, log_prob, False)
                
                # 에피소드 스텝 데이터
                episode_steps.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'value': value,
                    'log_prob': log_prob,
                    'done': step_data.get('done', False)
                })
                
                episode_reward += reward
            
            # 에피소드 훈련
            if len(episode_steps) > 0:
                agent.train_episode(episode_steps)
            
            return episode_reward, len(episode_data)
            
        except Exception as e:
            self.logger.error(f"에피소드 실행 실패: {e}")
            return 0.0, 0
    
    def _calculate_regime_performance(self, agent_performances: Dict[str, Dict]) -> Dict[str, float]:
        """
        체제별 성과 계산
        
        Args:
            agent_performances: 에이전트별 성과
            
        Returns:
            체제별 성과 지표
        """
        try:
            if not agent_performances:
                return {}
            
            # 모든 에이전트의 성과 지표 수집
            all_rewards = []
            all_lengths = []
            
            for agent_perf in agent_performances.values():
                if 'average_reward' in agent_perf:
                    all_rewards.append(agent_perf['average_reward'])
                if 'average_length' in agent_perf:
                    all_lengths.append(agent_perf['average_length'])
            
            # 체제별 성과 계산
            regime_performance = {
                'average_reward': np.mean(all_rewards) if all_rewards else 0.0,
                'std_reward': np.std(all_rewards) if all_rewards else 0.0,
                'best_reward': np.max(all_rewards) if all_rewards else 0.0,
                'worst_reward': np.min(all_rewards) if all_rewards else 0.0,
                'average_length': np.mean(all_lengths) if all_lengths else 0.0,
                'agent_count': len(agent_performances)
            }
            
            return regime_performance
            
        except Exception as e:
            self.logger.error(f"체제별 성과 계산 실패: {e}")
            return {}
    
    def get_agent_actions(self, regime: Union[str, MarketRegime], 
                         state: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        특정 체제의 모든 에이전트에서 액션 수집
        
        Args:
            regime: 체제
            state: 현재 상태
            
        Returns:
            에이전트별 액션 리스트 (action, log_prob, value)
        """
        try:
            if isinstance(regime, MarketRegime):
                regime_name = regime.value
            else:
                regime_name = regime
            
            if regime_name not in self.agent_pools:
                raise ValueError(f"지원하지 않는 체제: {regime_name}")
            
            agent_actions = []
            
            for agent in self.agent_pools[regime_name]:
                if agent.is_trained:
                    action, log_prob, value = agent.get_action(state)
                    agent_actions.append((action, log_prob, value))
                else:
                    # 훈련되지 않은 에이전트는 홀드 액션
                    agent_actions.append((2, 0.0, 0.0))
            
            return agent_actions
            
        except Exception as e:
            self.logger.error(f"에이전트 액션 수집 실패: {e}")
            return [(2, 0.0, 0.0) for _ in range(self.agents_per_regime)]
    
    def update_performance_history(self, regime: Union[str, MarketRegime], 
                                 performance: float):
        """
        성과 히스토리 업데이트
        
        Args:
            regime: 체제
            performance: 성과 점수
        """
        try:
            if isinstance(regime, MarketRegime):
                regime_name = regime.value
            else:
                regime_name = regime
            
            # 성과 히스토리 업데이트
            self.performance_history[regime_name].append(performance)
            self.recent_performance[regime_name].append(performance)
            
        except Exception as e:
            self.logger.error(f"성과 히스토리 업데이트 실패: {e}")
    
    def get_recent_performance(self, regime: Union[str, MarketRegime], 
                              days: int = 30) -> Dict[str, float]:
        """
        최근 성과 지표 반환
        
        Args:
            regime: 체제
            days: 최근 일수
            
        Returns:
            최근 성과 지표
        """
        try:
            if isinstance(regime, MarketRegime):
                regime_name = regime.value
            else:
                regime_name = regime
            
            if regime_name not in self.recent_performance:
                return {}
            
            recent_perfs = list(self.recent_performance[regime_name])
            
            if not recent_perfs:
                return {}
            
            # 최근 N일 성과 (실제로는 스텝 단위)
            recent_data = recent_perfs[-days:] if len(recent_perfs) >= days else recent_perfs
            
            return {
                'average_performance': np.mean(recent_data),
                'std_performance': np.std(recent_data),
                'best_performance': np.max(recent_data),
                'worst_performance': np.min(recent_data),
                'data_points': len(recent_data)
            }
            
        except Exception as e:
            self.logger.error(f"최근 성과 조회 실패: {e}")
            return {}
    
    def get_all_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """모든 체제의 성과 지표 반환"""
        try:
            all_performance = {}
            
            for regime in MarketRegime:
                regime_name = regime.value
                performance = self.get_recent_performance(regime_name)
                all_performance[regime_name] = performance
            
            return all_performance
            
        except Exception as e:
            self.logger.error(f"전체 체제 성과 조회 실패: {e}")
            return {}
    
    def get_training_status(self) -> Dict[str, Dict[str, Union[int, bool]]]:
        """훈련 상태 반환"""
        return self.training_progress.copy()
    
    def is_fully_trained(self) -> bool:
        """모든 에이전트가 훈련되었는지 확인"""
        try:
            for regime_name, progress in self.training_progress.items():
                if not progress['training_completed']:
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"훈련 상태 확인 실패: {e}")
            return False
    
    def save_models(self, base_path: str):
        """
        모든 에이전트 모델 저장
        
        Args:
            base_path: 저장 기본 경로
        """
        try:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            for regime_name, agent_pool in self.agent_pools.items():
                regime_path = base_path / regime_name
                regime_path.mkdir(parents=True, exist_ok=True)
                
                for i, agent in enumerate(agent_pool):
                    agent_path = regime_path / f"agent_{i}"
                    agent.save_model(str(agent_path))
            
            # 메타데이터 저장
            metadata = {
                'agents_per_regime': self.agents_per_regime,
                'diversity_seeds': self.diversity_seeds,
                'training_progress': self.training_progress,
                'performance_history': {
                    regime: list(perf_history) 
                    for regime, perf_history in self.performance_history.items()
                }
            }
            
            with open(base_path / 'agent_pool_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"에이전트 풀 모델 저장 완료: {base_path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_models(self, base_path: str):
        """
        모든 에이전트 모델 로드
        
        Args:
            base_path: 로드 기본 경로
        """
        try:
            base_path = Path(base_path)
            
            # 메타데이터 로드
            metadata_path = base_path / 'agent_pool_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.training_progress = metadata.get('training_progress', {})
                
                # 성과 히스토리 복원
                perf_history = metadata.get('performance_history', {})
                for regime, history in perf_history.items():
                    self.performance_history[regime] = deque(history, maxlen=1000)
            
            # 각 체제별 에이전트 모델 로드
            for regime_name, agent_pool in self.agent_pools.items():
                regime_path = base_path / regime_name
                
                if regime_path.exists():
                    for i, agent in enumerate(agent_pool):
                        agent_path = regime_path / f"agent_{i}"
                        if agent_path.with_suffix('_network.pth').exists():
                            agent.load_model(str(agent_path))
            
            self.is_trained = True
            self.logger.info(f"에이전트 풀 모델 로드 완료: {base_path}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise
    
    def get_pool_summary(self) -> Dict[str, Union[str, int, bool, Dict]]:
        """에이전트 풀 요약 정보 반환"""
        try:
            summary = {
                'total_regimes': len(self.agent_pools),
                'agents_per_regime': self.agents_per_regime,
                'total_agents': len(self.agent_pools) * self.agents_per_regime,
                'is_fully_trained': self.is_fully_trained(),
                'training_progress': self.training_progress,
                'regime_performance': self.get_all_regime_performance()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"풀 요약 정보 생성 실패: {e}")
            return {}


# 편의 함수들
def create_agent_pool(config: Dict, device: str = 'cpu') -> AgentPool:
    """에이전트 풀 생성 편의 함수"""
    return AgentPool(config, device)


def train_all_regime_agents(agent_pool: AgentPool, training_data: Dict[str, List[Dict]], 
                           episodes: int = 1000) -> Dict[str, Dict[str, float]]:
    """모든 체제의 에이전트 풀 훈련 편의 함수"""
    results = {}
    
    for regime_name, data in training_data.items():
        try:
            result = agent_pool.train_regime_agents(regime_name, data, episodes)
            results[regime_name] = result
        except Exception as e:
            logging.error(f"{regime_name} 훈련 실패: {e}")
            results[regime_name] = {}
    
    return results
