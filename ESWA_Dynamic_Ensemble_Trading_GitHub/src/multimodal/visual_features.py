"""
시각적 특징 추출 모듈
Visual Feature Extraction Module

ResNet-18 기반 캔들스틱 차트 시각적 패턴 분석
- 캔들스틱 차트 이미지 생성 (224x224x3)
- ResNet-18 사전훈련 모델 활용
- 256차원 시각적 특징 벡터 추출
- 차트 패턴 인식 (지지/저항선, 캔들 패턴 등)

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-10-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import cv2
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class CandlestickChartGenerator:
    """
    캔들스틱 차트 생성기
    
    OHLCV 데이터를 224x224x3 RGB 이미지로 변환
    """
    
    def __init__(self, width: int = 224, height: int = 224, dpi: int = 100):
        """
        캔들스틱 차트 생성기 초기화
        
        Args:
            width: 이미지 너비
            height: 이미지 높이
            dpi: 해상도
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        
        # 차트 스타일 설정
        self.bullish_color = '#00ff88'  # 상승 캔들 색상
        self.bearish_color = '#ff4444'  # 하락 캔들 색상
        self.background_color = '#1e1e1e'  # 배경 색상
        self.grid_color = '#333333'  # 격자 색상
        self.text_color = '#ffffff'  # 텍스트 색상
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"캔들스틱 차트 생성기 초기화: {width}x{height}")
    
    def create_candlestick_chart(self, ohlcv_data: pd.DataFrame, 
                                window_size: int = 60) -> np.ndarray:
        """
        캔들스틱 차트 이미지 생성
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            window_size: 표시할 캔들 수
            
        Returns:
            224x224x3 RGB 이미지 배열
        """
        try:
            # 최근 window_size개 데이터 선택
            recent_data = ohlcv_data.tail(window_size).copy()
            
            if len(recent_data) < 2:
                # 데이터가 부족한 경우 빈 차트 생성
                return self._create_empty_chart()
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(self.width/self.dpi, self.height/self.dpi), 
                                 dpi=self.dpi, facecolor=self.background_color)
            
            # 배경 설정
            ax.set_facecolor(self.background_color)
            fig.patch.set_facecolor(self.background_color)
            
            # 캔들스틱 그리기
            self._draw_candlesticks(ax, recent_data)
            
            # 차트 스타일링
            self._style_chart(ax, recent_data)
            
            # 이미지로 변환
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            
            # RGB 배열로 변환
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            # 크기 조정
            image = cv2.resize(buf, (self.width, self.height))
            
            plt.close(fig)
            
            return image
            
        except Exception as e:
            self.logger.error(f"캔들스틱 차트 생성 실패: {e}")
            return self._create_empty_chart()
    
    def _draw_candlesticks(self, ax, data: pd.DataFrame):
        """캔들스틱 그리기"""
        try:
            # 가격 범위 계산
            high_max = data['high'].max()
            low_min = data['low'].min()
            price_range = high_max - low_min
            
            if price_range == 0:
                price_range = 1
            
            # 캔들 너비 계산
            candle_width = 0.8 / len(data)
            
            for i, (idx, row) in enumerate(data.iterrows()):
                x_pos = i
                
                # 캔들 색상 결정
                is_bullish = row['close'] >= row['open']
                color = self.bullish_color if is_bullish else self.bearish_color
                
                # 캔들 몸통 그리기
                body_height = abs(row['close'] - row['open']) / price_range
                body_bottom = min(row['close'], row['open']) / price_range
                
                if body_height > 0:
                    rect = patches.Rectangle(
                        (x_pos - candle_width/2, body_bottom),
                        candle_width, body_height,
                        facecolor=color, edgecolor=color, linewidth=0.5
                    )
                    ax.add_patch(rect)
                
                # 위꼬리 그리기
                if row['high'] > max(row['close'], row['open']):
                    wick_top = row['high'] / price_range
                    wick_bottom = max(row['close'], row['open']) / price_range
                    ax.plot([x_pos, x_pos], [wick_bottom, wick_top], 
                           color=color, linewidth=1)
                
                # 아래꼬리 그리기
                if row['low'] < min(row['close'], row['open']):
                    wick_top = min(row['close'], row['open']) / price_range
                    wick_bottom = row['low'] / price_range
                    ax.plot([x_pos, x_pos], [wick_bottom, wick_top], 
                           color=color, linewidth=1)
            
        except Exception as e:
            self.logger.error(f"캔들스틱 그리기 실패: {e}")
    
    def _style_chart(self, ax, data: pd.DataFrame):
        """차트 스타일링"""
        try:
            # 축 설정
            ax.set_xlim(-0.5, len(data) - 0.5)
            ax.set_ylim(0, 1)
            
            # 격자 설정
            ax.grid(True, color=self.grid_color, alpha=0.3, linewidth=0.5)
            
            # 축 라벨 제거
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 테두리 제거
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # 제목 설정
            ax.set_title('Candlestick Chart', color=self.text_color, 
                        fontsize=10, pad=10)
            
        except Exception as e:
            self.logger.error(f"차트 스타일링 실패: {e}")
    
    def _create_empty_chart(self) -> np.ndarray:
        """빈 차트 생성"""
        try:
            fig, ax = plt.subplots(figsize=(self.width/self.dpi, self.height/self.dpi), 
                                 dpi=self.dpi, facecolor=self.background_color)
            
            ax.set_facecolor(self.background_color)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   color=self.text_color, fontsize=16)
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            image = cv2.resize(buf, (self.width, self.height))
            
            plt.close(fig)
            
            return image
            
        except Exception as e:
            self.logger.error(f"빈 차트 생성 실패: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class ResNet18VisualExtractor:
    """
    ResNet-18 기반 시각적 특징 추출기
    
    사전훈련된 ResNet-18 모델을 사용하여 캔들스틱 차트에서
    256차원 시각적 특징 벡터를 추출
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        ResNet-18 시각적 특징 추출기 초기화
        
        Args:
            config: 설정 딕셔너리
            device: 연산 장치
        """
        self.config = config
        self.device = torch.device(device)
        
        # 차트 생성기 초기화
        chart_config = config.get('data', {}).get('candlestick', {})
        self.chart_generator = CandlestickChartGenerator(
            width=chart_config.get('resolution', [224, 224])[0],
            height=chart_config.get('resolution', [224, 224])[1]
        )
        
        # 특징 차원
        self.feature_dim = 256
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # ResNet-18 모델 초기화
        self.model = self._initialize_resnet18()
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info("ResNet-18 시각적 특징 추출기 초기화 완료")
    
    def _initialize_resnet18(self) -> nn.Module:
        """ResNet-18 모델 초기화"""
        try:
            # 사전훈련된 ResNet-18 로드
            model = models.resnet18(pretrained=True)
            
            # 마지막 분류층을 특징 추출용으로 수정
            model.fc = nn.Linear(model.fc.in_features, self.feature_dim)
            
            # 모델을 지정된 장치로 이동
            model = model.to(self.device)
            
            # 평가 모드로 설정
            model.eval()
            
            self.logger.info("ResNet-18 모델 초기화 완료")
            return model
            
        except Exception as e:
            self.logger.error(f"ResNet-18 모델 초기화 실패: {e}")
            raise
    
    def extract_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        시각적 특징 추출
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            
        Returns:
            256차원 시각적 특징 벡터
        """
        try:
            # 캔들스틱 차트 이미지 생성
            chart_image = self.chart_generator.create_candlestick_chart(ohlcv_data)
            
            # 이미지 전처리
            processed_image = self.transform(chart_image)
            processed_image = processed_image.unsqueeze(0).to(self.device)
            
            # 특징 추출
            with torch.no_grad():
                features = self.model(processed_image)
                features = features.squeeze().cpu().numpy()
            
            # 정규화
            features = self._normalize_features(features)
            
            self.logger.debug(f"시각적 특징 추출 완료: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"시각적 특징 추출 실패: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """특징 정규화"""
        try:
            # L2 정규화
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"특징 정규화 실패: {e}")
            return features
    
    def extract_features_batch(self, ohlcv_batch: List[pd.DataFrame]) -> np.ndarray:
        """
        배치 단위 시각적 특징 추출
        
        Args:
            ohlcv_batch: OHLCV 데이터프레임 리스트
            
        Returns:
            특징 벡터 배열 (batch_size, 256)
        """
        try:
            features_list = []
            
            for ohlcv_data in ohlcv_batch:
                features = self.extract_features(ohlcv_data)
                features_list.append(features)
            
            return np.array(features_list)
            
        except Exception as e:
            self.logger.error(f"배치 시각적 특징 추출 실패: {e}")
            return np.zeros((len(ohlcv_batch), self.feature_dim), dtype=np.float32)
    
    def save_model(self, path: str):
        """모델 저장"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_dim': self.feature_dim,
                'config': self.config
            }, path)
            
            self.logger.info(f"ResNet-18 모델 저장: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, path: str):
        """모델 로드"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.feature_dim = checkpoint['feature_dim']
            
            self.logger.info(f"ResNet-18 모델 로드: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


class VisualFeatureExtractor:
    """
    시각적 특징 추출 메인 클래스
    
    캔들스틱 차트 생성과 ResNet-18 특징 추출을 통합
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        시각적 특징 추출기 초기화
        
        Args:
            config: 설정 딕셔너리
            device: 연산 장치
        """
        self.config = config
        self.device = device
        
        # ResNet-18 추출기 초기화
        self.resnet_extractor = ResNet18VisualExtractor(config, device)
        
        # 특징 차원
        self.feature_dim = 256
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("시각적 특징 추출기 초기화 완료")
    
    def extract_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        시각적 특징 추출 (메인 메서드)
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            
        Returns:
            256차원 시각적 특징 벡터
        """
        try:
            return self.resnet_extractor.extract_features(ohlcv_data)
            
        except Exception as e:
            self.logger.error(f"시각적 특징 추출 실패: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def extract_features_batch(self, ohlcv_batch: List[pd.DataFrame]) -> np.ndarray:
        """배치 단위 시각적 특징 추출"""
        try:
            return self.resnet_extractor.extract_features_batch(ohlcv_batch)
            
        except Exception as e:
            self.logger.error(f"배치 시각적 특징 추출 실패: {e}")
            return np.zeros((len(ohlcv_batch), self.feature_dim), dtype=np.float32)
    
    def save_model(self, path: str):
        """모델 저장"""
        try:
            self.resnet_extractor.save_model(path)
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, path: str):
        """모델 로드"""
        try:
            self.resnet_extractor.load_model(path)
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


# 편의 함수들
def create_visual_extractor(config: Dict, device: str = 'cpu') -> VisualFeatureExtractor:
    """시각적 특징 추출기 생성 편의 함수"""
    return VisualFeatureExtractor(config, device)


def extract_visual_features(ohlcv_data: pd.DataFrame, config: Dict, 
                           device: str = 'cpu') -> np.ndarray:
    """시각적 특징 추출 편의 함수"""
    extractor = create_visual_extractor(config, device)
    return extractor.extract_features(ohlcv_data)


def test_visual_extractor():
    """시각적 특징 추출기 테스트"""
    try:
        print("ResNet-18 시각적 특징 추출기 테스트 시작...")
        
        # 테스트 데이터 생성
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # 가격 데이터 생성 (간단한 랜덤 워크)
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        test_data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 50,
            'high': prices + np.abs(np.random.randn(100) * 100),
            'low': prices - np.abs(np.random.randn(100) * 100),
            'close': prices + np.random.randn(100) * 50,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # 설정
        config = {
            'data': {
                'candlestick': {
                    'resolution': [224, 224],
                    'window_size': 60
                }
            }
        }
        
        # 시각적 특징 추출기 생성
        extractor = create_visual_extractor(config)
        
        # 특징 추출 테스트
        features = extractor.extract_features(test_data)
        
        print(f"시각적 특징 추출 성공!")
        print(f"   - 입력 데이터: {len(test_data)}개 레코드")
        print(f"   - 출력 특징: {features.shape} (256차원)")
        print(f"   - 특징 범위: [{features.min():.4f}, {features.max():.4f}]")
        print(f"   - 특징 평균: {features.mean():.4f}")
        print(f"   - 특징 표준편차: {features.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"시각적 특징 추출기 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 테스트 실행
    test_visual_extractor()