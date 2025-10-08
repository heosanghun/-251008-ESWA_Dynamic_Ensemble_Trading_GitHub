"""
멀티모달 특징 추출 모듈
Multimodal Feature Extraction Module

이 모듈은 거래 시스템을 위한 세 가지 유형의 특징을 추출합니다:
1. 시각적 특징: ResNet-18 기반 캔들스틱 차트 분석
2. 기술적 특징: 15개 핵심 기술지표 계산
3. 감정 특징: LLM 기반 뉴스 감정분석

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

from .visual_features import VisualFeatureExtractor
from .technical_features import TechnicalFeatureExtractor
from .sentiment_features import SentimentFeatureExtractor
from .feature_fusion import MultimodalFeatureFusion

__all__ = [
    "VisualFeatureExtractor",
    "TechnicalFeatureExtractor", 
    "SentimentFeatureExtractor",
    "MultimodalFeatureFusion"
]
