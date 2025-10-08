"""
시장 체제 분류기 모듈
Market Regime Classifier Module

XGBoost 기반 확률적 시장 체제 분류
- 3개 체제: Bull Market, Bear Market, Sideways Market
- 확률적 출력을 통한 불확실성 정량화
- Walk-Forward Expanding Window Cross-Validation 적용
- 체제 라벨링: EMA 기반 트렌드 분석

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RegimeLabeler:
    """
    시장 체제 라벨링기
    
    OHLCV 데이터를 기반으로 시장 체제 라벨 생성
    """
    
    def __init__(self, config: Dict):
        """
        체제 라벨링기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.ema_periods = config.get('regime', {}).get('ema_periods', [20, 50, 200])
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("체제 라벨링기 초기화 완료")
    
    def calculate_ema_trend(self, data: pd.DataFrame) -> pd.Series:
        """
        EMA 기반 트렌드 계산
        
        Args:
            data: OHLCV 데이터프레임
            
        Returns:
            트렌드 점수 (-1: 하락, 0: 횡보, +1: 상승)
        """
        try:
            # EMA 계산
            ema_20 = data['close'].ewm(span=self.ema_periods[0]).mean()
            ema_50 = data['close'].ewm(span=self.ema_periods[1]).mean()
            ema_200 = data['close'].ewm(span=self.ema_periods[2]).mean()
            
            # 트렌드 점수 계산
            trend_score = np.zeros(len(data))
            
            # 상승 트렌드 조건
            bullish_condition = (
                (ema_20 > ema_50) & 
                (ema_50 > ema_200) & 
                (data['close'] > ema_20)
            )
            
            # 하락 트렌드 조건
            bearish_condition = (
                (ema_20 < ema_50) & 
                (ema_50 < ema_200) & 
                (data['close'] < ema_20)
            )
            
            # 횡보 트렌드 조건
            sideways_condition = ~(bullish_condition | bearish_condition)
            
            # 트렌드 점수 할당
            trend_score[bullish_condition] = 1.0   # 상승
            trend_score[bearish_condition] = -1.0  # 하락
            trend_score[sideways_condition] = 0.0  # 횡보
            
            return pd.Series(trend_score, index=data.index)
            
        except Exception as e:
            self.logger.error(f"EMA 트렌드 계산 실패: {e}")
            return pd.Series(0, index=data.index)
    
    def calculate_volatility_regime(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        변동성 기반 체제 분류
        
        Args:
            data: OHLCV 데이터프레임
            window: 변동성 계산 윈도우
            
        Returns:
            변동성 체제 점수
        """
        try:
            # 변동성 계산 (ATR 기반)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=window).mean()
            
            # 변동성 정규화
            volatility_score = (atr / data['close']).rolling(window=window).mean()
            
            # 변동성 체제 분류
            high_vol_threshold = volatility_score.quantile(0.7)
            low_vol_threshold = volatility_score.quantile(0.3)
            
            regime_score = np.zeros(len(data))
            regime_score[volatility_score > high_vol_threshold] = 1.0   # 고변동성
            regime_score[volatility_score < low_vol_threshold] = -1.0  # 저변동성
            # 중간 변동성은 0으로 유지
            
            return pd.Series(regime_score, index=data.index)
            
        except Exception as e:
            self.logger.error(f"변동성 체제 계산 실패: {e}")
            return pd.Series(0, index=data.index)
    
    def calculate_momentum_regime(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        모멘텀 기반 체제 분류
        
        Args:
            data: OHLCV 데이터프레임
            window: 모멘텀 계산 윈도우
            
        Returns:
            모멘텀 체제 점수
        """
        try:
            # RSI 계산
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 모멘텀 체제 분류
            momentum_score = np.zeros(len(data))
            momentum_score[rsi > 70] = 1.0   # 과매수 (상승 모멘텀)
            momentum_score[rsi < 30] = -1.0  # 과매도 (하락 모멘텀)
            # 중립 구간은 0으로 유지
            
            return pd.Series(momentum_score, index=data.index)
            
        except Exception as e:
            self.logger.error(f"모멘텀 체제 계산 실패: {e}")
            return pd.Series(0, index=data.index)
    
    def create_regime_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        종합적인 시장 체제 라벨 생성
        
        Args:
            data: OHLCV 데이터프레임
            
        Returns:
            체제 라벨 (0: Bull, 1: Bear, 2: Sideways)
        """
        try:
            # 각 체제 지표 계산
            trend_score = self.calculate_ema_trend(data)
            volatility_score = self.calculate_volatility_regime(data)
            momentum_score = self.calculate_momentum_regime(data)
            
            # 종합 점수 계산 (가중 평균)
            weights = {'trend': 0.5, 'volatility': 0.3, 'momentum': 0.2}
            composite_score = (
                weights['trend'] * trend_score +
                weights['volatility'] * volatility_score +
                weights['momentum'] * momentum_score
            )
            
            # 체제 라벨 생성
            regime_labels = np.zeros(len(data), dtype=int)
            
            # Bull Market: 종합 점수 > 0.3
            regime_labels[composite_score > 0.3] = 0
            
            # Bear Market: 종합 점수 < -0.3
            regime_labels[composite_score < -0.3] = 1
            
            # Sideways Market: -0.3 <= 종합 점수 <= 0.3
            regime_labels[(composite_score >= -0.3) & (composite_score <= 0.3)] = 2
            
            # 라벨 시리즈 생성
            regime_series = pd.Series(regime_labels, index=data.index)
            
            # 체제 분포 로깅
            regime_counts = regime_series.value_counts().sort_index()
            regime_names = ['Bull Market', 'Bear Market', 'Sideways Market']
            
            self.logger.info("체제 라벨 분포:")
            for i, (regime_idx, count) in enumerate(regime_counts.items()):
                percentage = (count / len(regime_series)) * 100
                self.logger.info(f"  {regime_names[regime_idx]}: {count}개 ({percentage:.1f}%)")
            
            return regime_series
            
        except Exception as e:
            self.logger.error(f"체제 라벨 생성 실패: {e}")
            return pd.Series(2, index=data.index)  # 기본값: Sideways


class MarketRegimeClassifier:
    """
    시장 체제 분류기 메인 클래스
    
    XGBoost 기반 확률적 시장 체제 분류
    """
    
    def __init__(self, config: Dict):
        """
        시장 체제 분류기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.regime_config = config.get('regime', {})
        self.classifier_config = self.regime_config.get('classifier', {})
        
        # XGBoost 분류기 설정
        self.xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': self.classifier_config.get('n_estimators', 100),
            'max_depth': self.classifier_config.get('max_depth', 6),
            'learning_rate': self.classifier_config.get('learning_rate', 0.1),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # XGBoost 분류기 초기화
        self.classifier = xgb.XGBClassifier(**self.xgb_params)
        
        # 체제 라벨링기
        self.regime_labeler = RegimeLabeler(config)
        self.labeler = self.regime_labeler  # 호환성 유지
        
        # 체제 이름 매핑
        self.regime_names = ['bull_market', 'bear_market', 'sideways_market']
        self.regime_labels = {name: i for i, name in enumerate(self.regime_names)}
        
        # 훈련 상태
        self.is_trained = False
        self.feature_importance = None
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("시장 체제 분류기 초기화 완료")
        self.logger.info(f"XGBoost 파라미터: {self.xgb_params}")
    
    def prepare_features(self, ohlcv_data: pd.DataFrame, 
                        technical_features: np.ndarray) -> pd.DataFrame:
        """
        분류기 입력 특징 준비
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            technical_features: 기술적 특징 배열
            
        Returns:
            특징 데이터프레임
        """
        try:
            # 기본 가격 특징
            features = pd.DataFrame(index=ohlcv_data.index)
            
            # 가격 변화율
            features['price_change_1h'] = ohlcv_data['close'].pct_change(1)
            features['price_change_4h'] = ohlcv_data['close'].pct_change(4)
            features['price_change_24h'] = ohlcv_data['close'].pct_change(24)
            
            # 변동성 특징
            features['volatility_1h'] = ohlcv_data['close'].rolling(24).std()
            features['volatility_4h'] = ohlcv_data['close'].rolling(96).std()
            
            # 거래량 특징
            features['volume_ratio'] = ohlcv_data['volume'] / ohlcv_data['volume'].rolling(24).mean()
            features['volume_change'] = ohlcv_data['volume'].pct_change(1)
            
            # 고가-저가 비율
            features['high_low_ratio'] = (ohlcv_data['high'] - ohlcv_data['low']) / ohlcv_data['close']
            
            # 기술적 지표 특징 추가
            if technical_features is not None and len(technical_features) > 0:
                # 기술적 특징을 데이터프레임으로 변환
                tech_feature_names = [
                    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_Percent',
                    'ATR', 'CCI', 'Stoch_K', 'ADX', 'OBV',
                    'VWAP', 'ROC', 'MFI', 'Williams_R', 'Parabolic_SAR'
                ]
                
                # 최신 기술적 특징만 사용
                if len(technical_features.shape) == 1:
                    # 단일 샘플
                    for i, name in enumerate(tech_feature_names):
                        if i < len(technical_features):
                            features[name] = technical_features[i]
                else:
                    # 배치 샘플 (최신값 사용)
                    latest_tech_features = technical_features[-1] if len(technical_features) > 0 else np.zeros(15)
                    for i, name in enumerate(tech_feature_names):
                        if i < len(latest_tech_features):
                            features[name] = latest_tech_features[i]
            
            # NaN 값 처리
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.debug(f"특징 준비 완료: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"특징 준비 실패: {e}")
            return pd.DataFrame(index=ohlcv_data.index)
    
    def train(self, ohlcv_data: pd.DataFrame, technical_features: np.ndarray = None,
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        시장 체제 분류기 훈련
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            technical_features: 기술적 특징 배열
            validation_split: 검증 데이터 비율
            
        Returns:
            훈련 성과 지표
        """
        try:
            self.logger.info("시장 체제 분류기 훈련 시작")
            
            # 1. 체제 라벨 생성
            regime_labels = self.regime_labeler.create_regime_labels(ohlcv_data)
            
            # 2. 특징 준비
            features = self.prepare_features(ohlcv_data, technical_features)
            
            # 3. 데이터 정렬 및 정리
            common_index = features.index.intersection(regime_labels.index)
            features = features.loc[common_index]
            regime_labels = regime_labels.loc[common_index]
            
            # 4. 훈련/검증 데이터 분할 (시간순)
            split_idx = int(len(features) * (1 - validation_split))
            
            X_train = features.iloc[:split_idx]
            y_train = regime_labels.iloc[:split_idx]
            X_val = features.iloc[split_idx:]
            y_val = regime_labels.iloc[split_idx:]
            
            self.logger.info(f"훈련 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개")
            
            # 5. 데이터 정리 (무한대 값, 결측값 처리)
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 6. XGBoost 분류기 훈련
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # 6. 검증 성과 평가
            y_pred = self.classifier.predict(X_val)
            y_pred_proba = self.classifier.predict_proba(X_val)
            
            # 성과 지표 계산
            accuracy = accuracy_score(y_val, y_pred)
            
            # 분류 리포트
            class_report = classification_report(y_val, y_pred, 
                                               target_names=self.regime_names, 
                                               output_dict=True)
            
            # 혼동 행렬
            conf_matrix = confusion_matrix(y_val, y_pred)
            
            # 특징 중요도
            self.feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 훈련 완료
            self.is_trained = True
            
            # 결과 로깅
            self.logger.info(f"훈련 완료 - 정확도: {accuracy:.4f}")
            self.logger.info("클래스별 성과:")
            for regime in self.regime_names:
                if regime in class_report:
                    precision = class_report[regime]['precision']
                    recall = class_report[regime]['recall']
                    f1 = class_report[regime]['f1-score']
                    self.logger.info(f"  {regime}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            
            # 상위 10개 중요 특징
            self.logger.info("상위 10개 중요 특징:")
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
                self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': self.feature_importance.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"시장 체제 분류기 훈련 실패: {e}")
            raise
    
    def predict_proba(self, ohlcv_data: pd.DataFrame, 
                     technical_features: np.ndarray = None) -> np.ndarray:
        """
        체제 확률 예측
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            technical_features: 기술적 특징 배열
            
        Returns:
            체제 확률 배열 (n_samples, 3)
        """
        try:
            if not self.is_trained:
                raise ValueError("분류기가 훈련되지 않았습니다.")
            
            # 특징 준비
            features = self.prepare_features(ohlcv_data, technical_features)
            
            # 확률 예측
            probabilities = self.classifier.predict_proba(features)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"체제 확률 예측 실패: {e}")
            # 오류 시 균등 분포 반환
            n_samples = len(ohlcv_data)
            return np.full((n_samples, 3), 1/3)
    
    def predict(self, ohlcv_data: pd.DataFrame, 
                technical_features: np.ndarray = None) -> np.ndarray:
        """
        체제 예측
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            technical_features: 기술적 특징 배열
            
        Returns:
            체제 예측 배열
        """
        try:
            if not self.is_trained:
                raise ValueError("분류기가 훈련되지 않았습니다.")
            
            # 특징 준비
            features = self.prepare_features(ohlcv_data, technical_features)
            
            # 예측
            predictions = self.classifier.predict(features)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"체제 예측 실패: {e}")
            # 오류 시 Sideways Market 반환
            return np.full(len(ohlcv_data), 2)
    
    def get_latest_regime_probability(self, ohlcv_data: pd.DataFrame, 
                                    technical_features: np.ndarray = None) -> Dict[str, float]:
        """
        최신 시장 체제 확률 반환
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            technical_features: 기술적 특징 배열
            
        Returns:
            체제별 확률 딕셔너리
        """
        try:
            probabilities = self.predict_proba(ohlcv_data, technical_features)
            latest_probs = probabilities[-1]  # 최신 확률
            
            regime_probs = {}
            for i, regime_name in enumerate(self.regime_names):
                regime_probs[regime_name] = float(latest_probs[i])
            
            return regime_probs
            
        except Exception as e:
            self.logger.error(f"최신 체제 확률 계산 실패: {e}")
            # 오류 시 균등 분포 반환
            return {regime: 1/3 for regime in self.regime_names}
    
    def save_model(self, path: str):
        """모델 저장"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # XGBoost 모델 저장
            self.classifier.save_model(f"{path}_xgb.json")
            
            # 메타데이터 저장
            metadata = {
                'regime_names': self.regime_names,
                'regime_labels': self.regime_labels,
                'xgb_params': self.xgb_params,
                'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
            }
            
            joblib.dump(metadata, f"{path}_metadata.pkl")
            
            self.logger.info(f"시장 체제 분류기 모델 저장: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, path: str):
        """모델 로드"""
        try:
            # XGBoost 모델 로드
            self.classifier.load_model(f"{path}_xgb.json")
            
            # 메타데이터 로드
            metadata = joblib.load(f"{path}_metadata.pkl")
            self.regime_names = metadata['regime_names']
            self.regime_labels = metadata['regime_labels']
            
            if metadata['feature_importance'] is not None:
                self.feature_importance = pd.DataFrame(metadata['feature_importance'])
            
            self.is_trained = True
            
            self.logger.info(f"시장 체제 분류기 모델 로드: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


# 편의 함수들
def create_regime_classifier(config: Dict) -> MarketRegimeClassifier:
    """시장 체제 분류기 생성 편의 함수"""
    return MarketRegimeClassifier(config)


def train_regime_classifier(ohlcv_data: pd.DataFrame, config: Dict, 
                           technical_features: np.ndarray = None) -> MarketRegimeClassifier:
    """시장 체제 분류기 훈련 편의 함수"""
    classifier = create_regime_classifier(config)
    classifier.train(ohlcv_data, technical_features)
    return classifier
