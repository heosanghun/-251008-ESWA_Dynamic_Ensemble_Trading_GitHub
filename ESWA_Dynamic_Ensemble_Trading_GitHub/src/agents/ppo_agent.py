"""
PPO 에이전트 모듈
PPO Agent Module

Proximal Policy Optimization 기반 강화학습 에이전트
- Actor-Critic 아키텍처
- 273차원 상태 공간 (멀티모달 특징)
- 5개 액션 공간 (Strong Buy, Buy, Hold, Sell, Strong Sell)
- 체제별 전문화를 위한 맞춤형 보상함수 지원

작성자: Sanghun HEO
버전: 1.0.0
날짜: 2025-09-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import deque
import random
from pathlib import Path
import json


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 네트워크
    
    PPO 알고리즘을 위한 Actor와 Critic 네트워크를 통합
    """
    
    def __init__(self, state_dim: int = 273, action_dim: int = 5, 
                 hidden_dims: List[int] = [128, 64], activation: str = 'relu',
                 dropout: float = 0.1):
        """
        Actor-Critic 네트워크 초기화
        
        Args:
            state_dim: 상태 차원 (273차원 멀티모달 특징)
            action_dim: 액션 차원 (5개 거래 액션)
            hidden_dims: 은닉층 차원 리스트
            activation: 활성화 함수
            dropout: 드롭아웃 비율
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # 활성화 함수 설정
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            self.activation = nn.ReLU
        
        # 공유 백본 네트워크
        self.backbone = self._build_backbone()
        
        # Actor 네트워크 (정책)
        self.actor = self._build_actor()
        
        # Critic 네트워크 (가치 함수)
        self.critic = self._build_critic()
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _build_backbone(self) -> nn.Module:
        """공유 백본 네트워크 구축"""
        layers = []
        input_dim = self.state_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_actor(self) -> nn.Module:
        """Actor 네트워크 구축"""
        return nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            self.activation(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dims[-1] // 2, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self) -> nn.Module:
        """Critic 네트워크 구축"""
        return nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            self.activation(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            state: 상태 텐서 (batch_size, state_dim)
            
        Returns:
            action_probs: 액션 확률 분포 (batch_size, action_dim)
            value: 상태 가치 (batch_size, 1)
        """
        # 공유 백본을 통한 특징 추출
        features = self.backbone(state)
        
        # Actor: 액션 확률 분포
        action_probs = self.actor(features)
        
        # Critic: 상태 가치
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, torch.Tensor]:
        """
        액션 선택
        
        Args:
            state: 상태 텐서
            deterministic: 결정적 액션 선택 여부
            
        Returns:
            action: 선택된 액션
            log_prob: 로그 확률
            value: 상태 가치
        """
        with torch.no_grad():
            action_probs, value = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                # 확률적 액션 선택
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            log_prob = torch.log(action_probs[0, action] + 1e-8)
            
            return action.item(), log_prob.item(), value.squeeze()


class PPOBuffer:
    """
    PPO 경험 버퍼
    
    PPO 알고리즘을 위한 경험 데이터 저장 및 관리
    """
    
    def __init__(self, buffer_size: int = 2048, state_dim: int = 273, 
                 action_dim: int = 5, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        PPO 버퍼 초기화
        
        Args:
            buffer_size: 버퍼 크기
            state_dim: 상태 차원
            action_dim: 액션 차원
            gamma: 할인 인자
            gae_lambda: GAE 람다
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 버퍼 초기화
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # GAE 계산을 위한 변수
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
    
    def store(self, state: np.ndarray, action: int, reward: float, 
              value: float, log_prob: float, done: bool):
        """
        경험 저장
        
        Args:
            state: 상태
            action: 액션
            reward: 보상
            value: 상태 가치
            log_prob: 로그 확률
            done: 종료 여부
        """
        assert self.ptr < self.max_size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0):
        """
        에피소드 종료 시 GAE 계산
        
        Args:
            last_value: 마지막 상태의 가치
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # GAE 계산
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # Returns 계산
        self.returns[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """할인 누적합 계산"""
        result = np.zeros_like(x)
        result[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            result[t] = x[t] + discount * result[t + 1]
        return result
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        버퍼 데이터 반환
        
        Returns:
            버퍼 데이터 딕셔너리
        """
        assert self.ptr == self.max_size
        
        self.ptr, self.path_start_idx = 0, 0
        
        # 정규화
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns,
            'dones': self.dones
        }


class PPOAgent:
    """
    PPO 에이전트 메인 클래스
    
    Proximal Policy Optimization 알고리즘 구현
    """
    
    def __init__(self, config: Dict, agent_id: str = "ppo_agent", 
                 device: str = 'cpu', seed: int = 42):
        """
        PPO 에이전트 초기화
        
        Args:
            config: 설정 딕셔너리
            agent_id: 에이전트 ID
            device: 연산 장치
            seed: 랜덤 시드
        """
        self.config = config
        self.agent_id = agent_id
        self.device = torch.device(device)
        
        # 시드 설정
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # PPO 파라미터
        self.ppo_config = config.get('agents', {}).get('ppo', {})
        self.learning_rate = self.ppo_config.get('learning_rate', 3e-4)
        self.batch_size = self.ppo_config.get('batch_size', 256)
        self.n_steps = self.ppo_config.get('n_steps', 2048)
        self.buffer_size = self.n_steps  # buffer_size 추가
        self.n_epochs = self.ppo_config.get('n_epochs', 10)
        self.gamma = self.ppo_config.get('gamma', 0.99)
        self.gae_lambda = self.ppo_config.get('gae_lambda', 0.95)
        self.clip_range = self.ppo_config.get('clip_range', 0.2)
        self.ent_coef = self.ppo_config.get('ent_coef', 0.01)
        
        # 네트워크 설정
        self.network_config = config.get('agents', {}).get('network', {})
        self.hidden_dims = self.network_config.get('hidden_layers', [128, 64])
        self.activation = self.network_config.get('activation', 'relu')
        self.dropout = self.network_config.get('dropout', 0.1)
        
        # 상태 및 액션 차원
        self.state_dim = 273  # 멀티모달 특징 차원
        self.action_dim = 5   # 거래 액션 차원
        
        # 네트워크 초기화
        self.network = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout
        ).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # 경험 버퍼
        self.buffer = PPOBuffer(
            buffer_size=self.n_steps,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # 훈련 상태
        self.is_trained = False
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # 성과 지표
        self.performance_metrics = {
            'total_reward': 0.0,
            'episode_count': 0,
            'average_reward': 0.0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf')
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        self.logger.info(f"PPO 에이전트 초기화 완료: {agent_id}")
        self.logger.info(f"네트워크 구조: {self.state_dim} → {self.hidden_dims} → {self.action_dim}")
        self.logger.info(f"PPO 파라미터: lr={self.learning_rate}, batch_size={self.batch_size}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        액션 선택
        
        Args:
            state: 상태 배열
            deterministic: 결정적 액션 선택 여부
            
        Returns:
            action: 선택된 액션
            log_prob: 로그 확률
            value: 상태 가치
        """
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
            
            return action, log_prob, value
            
        except Exception as e:
            self.logger.error(f"액션 선택 실패: {e}")
            return 2, 0.0, 0.0  # 기본값: Hold
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """
        경험 저장
        
        Args:
            state: 상태
            action: 액션
            reward: 보상
            value: 상태 가치
            log_prob: 로그 확률
            done: 종료 여부
        """
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self) -> Dict[str, float]:
        """
        PPO 업데이트
        
        Returns:
            훈련 지표 딕셔너리
        """
        try:
            if self.buffer.ptr < self.buffer_size:
                return {}
            
            # 버퍼에서 데이터 가져오기
            data = self.buffer.get()
            
            # 텐서로 변환
            states = torch.FloatTensor(data['states']).to(self.device)
            actions = torch.LongTensor(data['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
            advantages = torch.FloatTensor(data['advantages']).to(self.device)
            returns = torch.FloatTensor(data['returns']).to(self.device)
            
            # 훈련 지표
            policy_losses = []
            value_losses = []
            entropy_losses = []
            kl_divergences = []
            
            # 여러 에포크에 걸쳐 훈련
            for epoch in range(self.n_epochs):
                # 배치 단위로 훈련
                batch_indices = np.random.permutation(self.buffer_size)
                
                for start_idx in range(0, self.buffer_size, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, self.buffer_size)
                    batch_idx = batch_indices[start_idx:end_idx]
                    
                    # 배치 데이터
                    batch_states = states[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_advantages = advantages[batch_idx]
                    batch_returns = returns[batch_idx]
                    
                    # 순전파
                    action_probs, values = self.network(batch_states)
                    
                    # 현재 정책의 로그 확률
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    
                    # 엔트로피
                    entropy = dist.entropy().mean()
                    
                    # 정책 비율
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # PPO 손실
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 가치 함수 손실
                    value_loss = F.mse_loss(values.squeeze(), batch_returns)
                    
                    # KL 발산
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    
                    # 총 손실
                    total_loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
                    
                    # 역전파
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
                    
                    # 지표 저장
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy.item())
                    kl_divergences.append(kl_div.item())
            
            self.training_step += 1
            self.is_trained = True
            
            # 평균 지표 계산
            metrics = {
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy_loss': np.mean(entropy_losses),
                'kl_divergence': np.mean(kl_divergences),
                'training_step': self.training_step
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"PPO 업데이트 실패: {e}")
            return {}
    
    def train_episode(self, episode_data: List[Dict]) -> Dict[str, float]:
        """
        에피소드 훈련
        
        Args:
            episode_data: 에피소드 데이터 리스트
            
        Returns:
            에피소드 성과 지표
        """
        try:
            episode_reward = 0.0
            episode_length = len(episode_data)
            
            # 경험 저장
            for step_data in episode_data:
                state = step_data['state']
                action = step_data['action']
                reward = step_data['reward']
                value = step_data['value']
                log_prob = step_data['log_prob']
                done = step_data.get('done', False)
                
                self.store_experience(state, action, reward, value, log_prob, done)
                episode_reward += reward
            
            # 에피소드 종료
            self.buffer.finish_path()
            
            # 성과 지표 업데이트
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.performance_metrics['total_reward'] += episode_reward
            self.performance_metrics['episode_count'] += 1
            self.performance_metrics['average_reward'] = np.mean(self.episode_rewards)
            self.performance_metrics['best_reward'] = max(self.performance_metrics['best_reward'], episode_reward)
            self.performance_metrics['worst_reward'] = min(self.performance_metrics['worst_reward'], episode_reward)
            
            # PPO 업데이트
            training_metrics = self.update()
            
            # 에피소드 지표
            episode_metrics = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'average_reward': self.performance_metrics['average_reward'],
                'best_reward': self.performance_metrics['best_reward'],
                'worst_reward': self.performance_metrics['worst_reward']
            }
            
            # 훈련 지표와 결합
            episode_metrics.update(training_metrics)
            
            return episode_metrics
            
        except Exception as e:
            self.logger.error(f"에피소드 훈련 실패: {e}")
            return {}
    
    def save_model(self, path: str):
        """모델 저장"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # 네트워크 상태 저장
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'performance_metrics': self.performance_metrics,
                'training_step': self.training_step,
                'agent_id': self.agent_id
            }, f"{path}_network.pth")
            
            # 메타데이터 저장
            metadata = {
                'agent_id': self.agent_id,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dims': self.hidden_dims,
                'activation': self.activation,
                'dropout': self.dropout,
                'ppo_params': {
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'n_steps': self.n_steps,
                    'n_epochs': self.n_epochs,
                    'gamma': self.gamma,
                    'gae_lambda': self.gae_lambda,
                    'clip_range': self.clip_range,
                    'ent_coef': self.ent_coef
                },
                'performance_metrics': self.performance_metrics,
                'training_step': self.training_step
            }
            
            with open(f"{path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"PPO 에이전트 모델 저장: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, path: str):
        """모델 로드"""
        try:
            # 네트워크 상태 로드
            checkpoint = torch.load(f"{path}_network.pth", map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.performance_metrics = checkpoint['performance_metrics']
            self.training_step = checkpoint['training_step']
            
            self.is_trained = True
            
            self.logger.info(f"PPO 에이전트 모델 로드: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Union[float, int, str]]:
        """성과 요약 반환"""
        return {
            'agent_id': self.agent_id,
            'is_trained': self.is_trained,
            'training_step': self.training_step,
            'episode_count': self.performance_metrics['episode_count'],
            'average_reward': self.performance_metrics['average_reward'],
            'best_reward': self.performance_metrics['best_reward'],
            'worst_reward': self.performance_metrics['worst_reward'],
            'total_reward': self.performance_metrics['total_reward']
        }


# 편의 함수들
def create_ppo_agent(config: Dict, agent_id: str = "ppo_agent", 
                    device: str = 'cpu', seed: int = 42) -> PPOAgent:
    """PPO 에이전트 생성 편의 함수"""
    return PPOAgent(config, agent_id, device, seed)


def create_multiple_ppo_agents(config: Dict, num_agents: int = 5, 
                              device: str = 'cpu', base_seed: int = 42) -> List[PPOAgent]:
    """여러 PPO 에이전트 생성 편의 함수"""
    agents = []
    for i in range(num_agents):
        agent_id = f"ppo_agent_{i}"
        seed = base_seed + i
        agent = create_ppo_agent(config, agent_id, device, seed)
        agents.append(agent)
    return agents

