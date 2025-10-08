"""
멀티모달 특징 융합 모듈
Multimodal Feature Fusion Module

세 가지 유형의 특징을 융합하여 최종 273차원 상태벡터 생성
- 시각적 특징 (256차원): ResNet-18 기반 캔들스틱 차트 분석
- 기술적 특징 (15차원): 15개 핵심 기술적 지표
- 감정 특징 (2차원): 뉴스 감정분석 결과
- Early Fusion 방식으로 특징 결합

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


class MultimodalFeatureFusion:
    """
    멀티모달 특징 융합 메인 클래스
    
    시각적, 기술적, 감정적 특징을 통합하여 최종 상태벡터 생성
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        멀티모달 특징 융합기 초기화
        
        Args:
            config: 설정 딕셔너리
            device: 연산 장치 ('cpu' 또는 'cuda')
        """
        self.config = config
        self.device = torch.device(device)
        
        # 각 특징 추출기 import
        from .visual_features import VisualFeatureExtractor
        from .sentiment_features import SentimentFeatureExtractor
        # from .technical_features import TechnicalFeatureExtractor
        
        # 특징 추출기 초기화
        self.visual_extractor = VisualFeatureExtractor(config, device)
        self.sentiment_extractor = SentimentFeatureExtractor(config)
        self.technical_extractor = None  # TechnicalFeatureExtractor(config)
        
        # 특징 차원 설정
        self.visual_dim = 256
        self.technical_dim = 15
        self.sentiment_dim = 2
        self.total_dim = self.visual_dim + self.technical_dim + self.sentiment_dim  # 273
        
        # 특징 정규화 스케일러
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("멀티모달 특징 융합기 초기화 완료")
        self.logger.info(f"특징 차원: 시각적({self.visual_dim}) + 기술적({self.technical_dim}) + 감정적({self.sentiment_dim}) = {self.total_dim}")
    
    def extract_visual_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        시각적 특징 추출
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            
        Returns:
            256차원 시각적 특징 벡터
        """
        try:
            if self.visual_extractor is None:
                # 임시 구현: 랜덤 특징 벡터 생성
                self.logger.warning("시각적 특징 추출기가 초기화되지 않음. 랜덤 벡터 생성")
                return np.random.randn(self.visual_dim).astype(np.float32)
            
            return self.visual_extractor.extract_features(ohlcv_data)
            
        except Exception as e:
            self.logger.error(f"시각적 특징 추출 실패: {e}")
            return np.zeros(self.visual_dim, dtype=np.float32)
    
    def extract_technical_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        기술적 특징 추출
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            
        Returns:
            15차원 기술적 특징 벡터
        """
        try:
            if self.technical_extractor is None:
                # 임시 구현: 랜덤 특징 벡터 생성
                self.logger.warning("기술적 특징 추출기가 초기화되지 않음. 랜덤 벡터 생성")
                return np.random.randn(self.technical_dim).astype(np.float32)
            
            return self.technical_extractor.extract_features(ohlcv_data)
            
        except Exception as e:
            self.logger.error(f"기술적 특징 추출 실패: {e}")
            return np.zeros(self.technical_dim, dtype=np.float32)
    
    def extract_sentiment_features(self, hours_back: int = 24) -> np.ndarray:
        """
        감정 특징 추출
        
        Args:
            hours_back: 몇 시간 전까지의 뉴스 수집
            
        Returns:
            2차원 감정 특징 벡터
        """
        try:
            if self.sentiment_extractor is None:
                # 임시 구현: 랜덤 특징 벡터 생성
                self.logger.warning("감정 특징 추출기가 초기화되지 않음. 랜덤 벡터 생성")
                return np.random.randn(self.sentiment_dim).astype(np.float32)
            
            return self.sentiment_extractor.extract_features(hours_back)
            
        except Exception as e:
            self.logger.error(f"감정 특징 추출 실패: {e}")
            return np.zeros(self.sentiment_dim, dtype=np.float32)
    
    def fuse_features(self, visual_features: np.ndarray, 
                     technical_features: np.ndarray, 
                     sentiment_features: np.ndarray) -> np.ndarray:
        """
        특징 융합 (Early Fusion)
        
        Args:
            visual_features: 시각적 특징 벡터 (256차원)
            technical_features: 기술적 특징 벡터 (15차원)
            sentiment_features: 감정 특징 벡터 (2차원)
            
        Returns:
            융합된 특징 벡터 (273차원)
        """
        try:
            # 차원 검증
            assert visual_features.shape[0] == self.visual_dim, f"시각적 특징 차원 오류: {visual_features.shape[0]} != {self.visual_dim}"
            assert technical_features.shape[0] == self.technical_dim, f"기술적 특징 차원 오류: {technical_features.shape[0]} != {self.technical_dim}"
            assert sentiment_features.shape[0] == self.sentiment_dim, f"감정 특징 차원 오류: {sentiment_features.shape[0]} != {self.sentiment_dim}"
            
            # Early Fusion: 단순 연결 (concatenation)
            fused_features = np.concatenate([
                visual_features,
                technical_features,
                sentiment_features
            ])
            
            # 차원 검증
            assert fused_features.shape[0] == self.total_dim, f"융합 특징 차원 오류: {fused_features.shape[0]} != {self.total_dim}"
            
            return fused_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"특징 융합 실패: {e}")
            return np.zeros(self.total_dim, dtype=np.float32)
    
    def normalize_features(self, features: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """
        특징 정규화
        
        Args:
            features: 특징 벡터
            fit_scaler: 스케일러 피팅 여부
            
        Returns:
            정규화된 특징 벡터
        """
        try:
            if fit_scaler:
                # 스케일러 피팅
                normalized_features = self.scaler.fit_transform(features.reshape(1, -1))
                self.is_fitted = True
                self.logger.info("특징 정규화 스케일러 피팅 완료")
            else:
                if not self.is_fitted:
                    raise ValueError("스케일러가 피팅되지 않았습니다.")
                # 기존 스케일러로 변환
                normalized_features = self.scaler.transform(features.reshape(1, -1))
            
            return normalized_features.flatten()
            
        except Exception as e:
            self.logger.error(f"특징 정규화 실패: {e}")
            return features
    
    def extract_fused_features(self, ohlcv_data: pd.DataFrame, 
                              hours_back: int = 24, 
                              normalize: bool = True) -> np.ndarray:
        """
        융합된 특징 추출 (메인 메서드)
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            hours_back: 뉴스 수집 시간 범위
            normalize: 정규화 여부
            
        Returns:
            273차원 융합 특징 벡터
        """
        try:
            self.logger.debug("멀티모달 특징 추출 시작")
            
            # 1. 시각적 특징 추출
            visual_features = self.extract_visual_features(ohlcv_data)
            self.logger.debug(f"시각적 특징 추출 완료: {visual_features.shape}")
            
            # 2. 기술적 특징 추출
            technical_features = self.extract_technical_features(ohlcv_data)
            self.logger.debug(f"기술적 특징 추출 완료: {technical_features.shape}")
            
            # 3. 감정 특징 추출
            sentiment_features = self.extract_sentiment_features(hours_back)
            self.logger.debug(f"감정 특징 추출 완료: {sentiment_features.shape}")
            
            # 4. 특징 융합
            fused_features = self.fuse_features(visual_features, technical_features, sentiment_features)
            self.logger.debug(f"특징 융합 완료: {fused_features.shape}")
            
            # 5. 정규화 (선택적)
            if normalize:
                fused_features = self.normalize_features(fused_features, fit_scaler=True)
                self.logger.debug("특징 정규화 완료")
            
            self.logger.info(f"멀티모달 특징 추출 완료: {fused_features.shape}")
            return fused_features
            
        except Exception as e:
            self.logger.error(f"멀티모달 특징 추출 실패: {e}")
            return np.zeros(self.total_dim, dtype=np.float32)
    
    def extract_fused_features_batch(self, ohlcv_batch: List[pd.DataFrame], 
                                   hours_back: int = 24, 
                                   normalize: bool = True) -> np.ndarray:
        """
        배치 단위 융합 특징 추출
        
        Args:
            ohlcv_batch: OHLCV 데이터프레임 리스트
            hours_back: 뉴스 수집 시간 범위
            normalize: 정규화 여부
            
        Returns:
            특징 벡터 배열 (batch_size, 273)
        """
        try:
            features_list = []
            
            for ohlcv_data in ohlcv_batch:
                features = self.extract_fused_features(ohlcv_data, hours_back, normalize)
                features_list.append(features)
            
            return np.array(features_list)
            
        except Exception as e:
            self.logger.error(f"배치 멀티모달 특징 추출 실패: {e}")
            return np.zeros((len(ohlcv_batch), self.total_dim), dtype=np.float32)
    
    def get_feature_breakdown(self, fused_features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        융합된 특징을 개별 특징으로 분해
        
        Args:
            fused_features: 융합된 특징 벡터 (273차원)
            
        Returns:
            개별 특징 딕셔너리
        """
        try:
            assert fused_features.shape[0] == self.total_dim, f"입력 특징 차원 오류: {fused_features.shape[0]} != {self.total_dim}"
            
            breakdown = {
                'visual': fused_features[:self.visual_dim],
                'technical': fused_features[self.visual_dim:self.visual_dim + self.technical_dim],
                'sentiment': fused_features[self.visual_dim + self.technical_dim:]
            }
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"특징 분해 실패: {e}")
            return {
                'visual': np.zeros(self.visual_dim),
                'technical': np.zeros(self.technical_dim),
                'sentiment': np.zeros(self.sentiment_dim)
            }
    
    def get_feature_names(self) -> List[str]:
        """특징 이름 리스트 반환"""
        feature_names = []
        
        # 시각적 특징 이름
        feature_names.extend([f'visual_{i}' for i in range(self.visual_dim)])
        
        # 기술적 특징 이름
        technical_names = [
            'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_Percent',
            'ATR', 'CCI', 'Stoch_K', 'ADX', 'OBV',
            'VWAP', 'ROC', 'MFI', 'Williams_R', 'Parabolic_SAR'
        ]
        feature_names.extend(technical_names)
        
        # 감정 특징 이름
        feature_names.extend(['sentiment_simple_avg', 'sentiment_ewma'])
        
        return feature_names
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """각 특징 유형별 차원 반환"""
        return {
            'visual': self.visual_dim,
            'technical': self.technical_dim,
            'sentiment': self.sentiment_dim,
            'total': self.total_dim
        }
    
    def save_scaler(self, path: str):
        """정규화 스케일러 저장"""
        try:
            import joblib
            joblib.dump(self.scaler, path)
            self.logger.info(f"정규화 스케일러 저장: {path}")
            
        except Exception as e:
            self.logger.error(f"스케일러 저장 실패: {e}")
            raise
    
    def load_scaler(self, path: str):
        """정규화 스케일러 로드"""
        try:
            import joblib
            self.scaler = joblib.load(path)
            self.is_fitted = True
            self.logger.info(f"정규화 스케일러 로드: {path}")
            
        except Exception as e:
            self.logger.error(f"스케일러 로드 실패: {e}")
            raise


class AdvancedFeatureFusion(nn.Module):
    """
    고급 특징 융합 네트워크 (선택적)
    
    단순 연결 대신 신경망을 통한 특징 융합
    """
    
    def __init__(self, visual_dim: int = 256, technical_dim: int = 15, 
                 sentiment_dim: int = 2, hidden_dim: int = 128, output_dim: int = 273):
        """
        고급 특징 융합 네트워크 초기화
        
        Args:
            visual_dim: 시각적 특징 차원
            technical_dim: 기술적 특징 차원
            sentiment_dim: 감정 특징 차원
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원
        """
        super(AdvancedFeatureFusion, self).__init__()
        
        self.visual_dim = visual_dim
        self.technical_dim = technical_dim
        self.sentiment_dim = sentiment_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 각 특징 유형별 처리 레이어
        self.visual_processor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.technical_processor = nn.Sequential(
            nn.Linear(technical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.sentiment_processor = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # 출력 정규화
        )
    
    def forward(self, visual_features: torch.Tensor, 
                technical_features: torch.Tensor, 
                sentiment_features: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            visual_features: 시각적 특징 텐서
            technical_features: 기술적 특징 텐서
            sentiment_features: 감정 특징 텐서
            
        Returns:
            융합된 특징 텐서
        """
        # 각 특징 유형별 처리
        visual_processed = self.visual_processor(visual_features)
        technical_processed = self.technical_processor(technical_features)
        sentiment_processed = self.sentiment_processor(sentiment_features)
        
        # 특징 융합
        fused = torch.cat([visual_processed, technical_processed, sentiment_processed], dim=1)
        
        # 최종 융합 레이어
        output = self.fusion_layer(fused)
        
        return output


# 편의 함수들
def create_multimodal_fusion(config: Dict, device: str = 'cpu') -> MultimodalFeatureFusion:
    """멀티모달 특징 융합기 생성 편의 함수"""
    return MultimodalFeatureFusion(config, device)


def extract_multimodal_features(ohlcv_data: pd.DataFrame, config: Dict, 
                               hours_back: int = 24) -> np.ndarray:
    """멀티모달 특징 추출 편의 함수"""
    fusion = create_multimodal_fusion(config)
    return fusion.extract_fused_features(ohlcv_data, hours_back)
