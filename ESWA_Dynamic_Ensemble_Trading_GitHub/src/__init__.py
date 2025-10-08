"""
ESWA 동적 앙상블 강화학습 거래 시스템
Robust Dynamic Ensemble Reinforcement Learning Trading System

이 패키지는 시장 체제 변화에 동적으로 적응하는 강화학습 거래 시스템을 구현합니다.
핵심 구성 요소:
- 멀티모달 특징 추출 (시각적, 기술적, 감정적)
- 시장 체제 분류 및 불확실성 관리
- 체제별 전문 PPO 에이전트 풀
- 동적 앙상블 의사결정 시스템

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "ESWA Dynamic Ensemble Reinforcement Learning Trading System"

# 주요 모듈들
from .multimodal import *
from .regime import *
from .agents import *
from .ensemble import *
from .trading import *
from .utils import *

__all__ = [
    # 멀티모달 특징 추출
    "VisualFeatureExtractor",
    "TechnicalFeatureExtractor", 
    "SentimentFeatureExtractor",
    "MultimodalFeatureFusion",
    
    # 시장 체제 분류
    "MarketRegimeClassifier",
    "ConfidenceBasedSelection",
    
    # PPO 에이전트
    "PPOAgent",
    "AgentPool",
    "RegimeSpecificRewards",
    
    # 앙상블 시스템
    "DynamicWeightAllocation",
    "PolicyEnsemble",
    
    # 거래 환경
    "TradingEnvironment",
    "PortfolioManager",
    "BacktestingEngine",
    
    # 유틸리티
    "DataLoader",
    "PerformanceMetrics",
    "VisualizationTools"
]
