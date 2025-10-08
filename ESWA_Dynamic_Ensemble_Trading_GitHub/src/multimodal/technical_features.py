"""
기술적 특징 추출 모듈
Technical Feature Extraction Module

15개 핵심 기술적 지표 계산 및 정규화
- 이동평균 (SMA, EMA)
- 모멘텀 지표 (RSI, MACD, ROC)
- 변동성 지표 (ATR, Bollinger Bands)
- 거래량 지표 (OBV, VWAP, MFI)
- 기타 지표 (CCI, STOCH, ADX, Williams %R, Parabolic SAR)

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, PSARIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, MFIIndicator
# CCI는 수동으로 구현


class TechnicalIndicatorCalculator:
    """
    기술적 지표 계산기
    
    다양한 기술적 지표들을 계산하고 정규화
    """
    
    def __init__(self, config: Dict):
        """
        기술적 지표 계산기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.indicators_config = config.get('technical', {})
        self.lookback_periods = self.indicators_config.get('lookback_periods', [20, 50, 200])
        
        # 정규화 스케일러
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("기술적 지표 계산기 초기화 완료")
    
    def calculate_sma(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        단순 이동평균 (SMA) 계산
        
        Args:
            data: OHLCV 데이터
            periods: 이동평균 기간 리스트
            
        Returns:
            SMA 지표들
        """
        sma_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            sma = SMAIndicator(close=data['close'], window=period)
            sma_data[f'SMA_{period}'] = sma.sma_indicator()
        
        return sma_data
    
    def calculate_ema(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        지수 이동평균 (EMA) 계산
        
        Args:
            data: OHLCV 데이터
            periods: 이동평균 기간 리스트
            
        Returns:
            EMA 지표들
        """
        ema_data = pd.DataFrame(index=data.index)
        
        for period in periods:
            ema = EMAIndicator(close=data['close'], window=period)
            ema_data[f'EMA_{period}'] = ema.ema_indicator()
        
        return ema_data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        상대강도지수 (RSI) 계산
        
        Args:
            data: OHLCV 데이터
            period: RSI 기간
            
        Returns:
            RSI 지표
        """
        rsi = RSIIndicator(close=data['close'], window=period)
        return rsi.rsi()
    
    def calculate_macd(self, data: pd.DataFrame, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD 지표 계산
        
        Args:
            data: OHLCV 데이터
            fast: 빠른 EMA 기간
            slow: 느린 EMA 기간
            signal: 신호선 기간
            
        Returns:
            MACD 지표들 (MACD, Signal, Histogram)
        """
        macd = MACD(close=data['close'], window_fast=fast, 
                   window_slow=slow, window_sign=signal)
        
        macd_data = pd.DataFrame(index=data.index)
        macd_data['MACD'] = macd.macd()
        macd_data['MACD_Signal'] = macd.macd_signal()
        macd_data['MACD_Histogram'] = macd.macd_diff()
        
        return macd_data
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                 period: int = 20, std: float = 2) -> pd.DataFrame:
        """
        볼린저 밴드 계산
        
        Args:
            data: OHLCV 데이터
            period: 이동평균 기간
            std: 표준편차 배수
            
        Returns:
            볼린저 밴드 지표들
        """
        bb = BollingerBands(close=data['close'], window=period, window_dev=std)
        
        bb_data = pd.DataFrame(index=data.index)
        bb_data['BB_Upper'] = bb.bollinger_hband()
        bb_data['BB_Middle'] = bb.bollinger_mavg()
        bb_data['BB_Lower'] = bb.bollinger_lband()
        bb_data['BB_Width'] = bb.bollinger_wband()
        bb_data['BB_Percent'] = bb.bollinger_pband()
        
        return bb_data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        평균진정범위 (ATR) 계산
        
        Args:
            data: OHLCV 데이터
            period: ATR 기간
            
        Returns:
            ATR 지표
        """
        atr = AverageTrueRange(high=data['high'], low=data['low'], 
                              close=data['close'], window=period)
        return atr.average_true_range()
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        상품채널지수 (CCI) 계산
        
        Args:
            data: OHLCV 데이터
            period: CCI 기간
            
        Returns:
            CCI 지표
        """
        # CCI 수동 계산
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def calculate_stochastic(self, data: pd.DataFrame, 
                           k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        스토캐스틱 오실레이터 계산
        
        Args:
            data: OHLCV 데이터
            k_period: %K 기간
            d_period: %D 기간
            
        Returns:
            스토캐스틱 지표들
        """
        stoch = StochasticOscillator(high=data['high'], low=data['low'], 
                                   close=data['close'], window=k_period, 
                                   smooth_window=d_period)
        
        stoch_data = pd.DataFrame(index=data.index)
        stoch_data['Stoch_K'] = stoch.stoch()
        stoch_data['Stoch_D'] = stoch.stoch_signal()
        
        return stoch_data
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ADX (Average Directional Index) 계산
        
        Args:
            data: OHLCV 데이터
            period: ADX 기간
            
        Returns:
            ADX 지표들
        """
        adx = ADXIndicator(high=data['high'], low=data['low'], 
                          close=data['close'], window=period)
        
        adx_data = pd.DataFrame(index=data.index)
        adx_data['ADX'] = adx.adx()
        adx_data['ADX_Positive'] = adx.adx_pos()
        adx_data['ADX_Negative'] = adx.adx_neg()
        
        return adx_data
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        거래량 균형 (OBV) 계산
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            OBV 지표
        """
        obv = OnBalanceVolumeIndicator(close=data['close'], volume=data['volume'])
        return obv.on_balance_volume()
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """
        거래량 가중 평균 가격 (VWAP) 계산
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            VWAP 지표
        """
        # VWAP = Σ(Price × Volume) / Σ(Volume)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    def calculate_roc(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        변화율 (ROC) 계산
        
        Args:
            data: OHLCV 데이터
            period: ROC 기간
            
        Returns:
            ROC 지표
        """
        roc = ROCIndicator(close=data['close'], window=period)
        return roc.roc()
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        자금 흐름 지수 (MFI) 계산
        
        Args:
            data: OHLCV 데이터
            period: MFI 기간
            
        Returns:
            MFI 지표
        """
        mfi = MFIIndicator(high=data['high'], low=data['low'], 
                          close=data['close'], volume=data['volume'], window=period)
        return mfi.money_flow_index()
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        윌리엄스 %R 계산
        
        Args:
            data: OHLCV 데이터
            period: Williams %R 기간
            
        Returns:
            Williams %R 지표
        """
        williams_r = WilliamsRIndicator(high=data['high'], low=data['low'], 
                                      close=data['close'], lbp=period)
        return williams_r.williams_r()
    
    def calculate_parabolic_sar(self, data: pd.DataFrame, 
                               step: float = 0.02, max_step: float = 0.2) -> pd.Series:
        """
        포물선 SAR 계산
        
        Args:
            data: OHLCV 데이터
            step: SAR 스텝
            max_step: 최대 스텝
            
        Returns:
            Parabolic SAR 지표
        """
        psar = PSARIndicator(high=data['high'], low=data['low'], 
                                   close=data['close'], step=step, max_step=max_step)
        return psar.psar()
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 계산
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            모든 기술적 지표가 포함된 데이터프레임
        """
        try:
            self.logger.info("기술적 지표 계산 시작")
            
            # 결과 데이터프레임 초기화
            indicators = pd.DataFrame(index=data.index)
            
            # 1. 이동평균 지표
            sma_data = self.calculate_sma(data, self.lookback_periods)
            ema_data = self.calculate_ema(data, self.lookback_periods)
            indicators = pd.concat([indicators, sma_data, ema_data], axis=1)
            
            # 2. 모멘텀 지표
            indicators['RSI'] = self.calculate_rsi(data)
            macd_data = self.calculate_macd(data)
            indicators = pd.concat([indicators, macd_data], axis=1)
            indicators['ROC'] = self.calculate_roc(data)
            
            # 3. 변동성 지표
            bb_data = self.calculate_bollinger_bands(data)
            indicators = pd.concat([indicators, bb_data], axis=1)
            indicators['ATR'] = self.calculate_atr(data)
            
            # 4. 기타 지표
            indicators['CCI'] = self.calculate_cci(data)
            stoch_data = self.calculate_stochastic(data)
            indicators = pd.concat([indicators, stoch_data], axis=1)
            adx_data = self.calculate_adx(data)
            indicators = pd.concat([indicators, adx_data], axis=1)
            
            # 5. 거래량 지표
            indicators['OBV'] = self.calculate_obv(data)
            indicators['VWAP'] = self.calculate_vwap(data)
            indicators['MFI'] = self.calculate_mfi(data)
            
            # 6. 추가 지표
            indicators['Williams_R'] = self.calculate_williams_r(data)
            indicators['Parabolic_SAR'] = self.calculate_parabolic_sar(data)
            
            # NaN 값 처리
            indicators = indicators.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"기술적 지표 계산 완료: {indicators.shape}")
            return indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {e}")
            # 오류 시 빈 데이터프레임 반환
            return pd.DataFrame(index=data.index)
    
    def normalize_indicators(self, indicators: pd.DataFrame, 
                           fit_scaler: bool = True) -> pd.DataFrame:
        """
        기술적 지표 정규화
        
        Args:
            indicators: 기술적 지표 데이터프레임
            fit_scaler: 스케일러 피팅 여부
            
        Returns:
            정규화된 지표 데이터프레임
        """
        try:
            if fit_scaler:
                # 스케일러 피팅
                normalized_data = self.scaler.fit_transform(indicators.values)
                self.is_fitted = True
                self.logger.info("기술적 지표 정규화 스케일러 피팅 완료")
            else:
                if not self.is_fitted:
                    raise ValueError("스케일러가 피팅되지 않았습니다.")
                # 기존 스케일러로 변환
                normalized_data = self.scaler.transform(indicators.values)
            
            # 정규화된 데이터프레임 생성
            normalized_df = pd.DataFrame(
                normalized_data, 
                index=indicators.index, 
                columns=indicators.columns
            )
            
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 정규화 실패: {e}")
            return indicators


class TechnicalFeatureExtractor:
    """
    기술적 특징 추출 메인 클래스
    
    기술적 지표 계산과 정규화를 통합
    """
    
    def __init__(self, config: Dict):
        """
        기술적 특징 추출기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.calculator = TechnicalIndicatorCalculator(config)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("기술적 특징 추출기 초기화 완료")
    
    def extract_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        OHLCV 데이터로부터 기술적 특징 추출
        
        Args:
            ohlcv_data: OHLCV 데이터프레임
            
        Returns:
            15차원 특징 벡터 (주요 지표들의 최신값)
        """
        try:
            # 모든 기술적 지표 계산
            indicators = self.calculator.calculate_all_indicators(ohlcv_data)
            
            # 정규화
            normalized_indicators = self.calculator.normalize_indicators(indicators)
            
            # 최신값 추출 (15개 주요 지표)
            feature_vector = self._extract_key_features(normalized_indicators)
            
            self.logger.debug(f"기술적 특징 추출 완료: {feature_vector.shape}")
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"기술적 특징 추출 실패: {e}")
            # 오류 시 제로 벡터 반환
            return np.zeros(15, dtype=np.float32)
    
    def _extract_key_features(self, indicators: pd.DataFrame) -> np.ndarray:
        """
        주요 기술적 지표들의 최신값 추출
        
        Args:
            indicators: 정규화된 지표 데이터프레임
            
        Returns:
            15차원 특징 벡터
        """
        try:
            # 최신값 (마지막 행) 추출
            latest_values = indicators.iloc[-1]
            
            # 15개 주요 지표 선택
            key_features = [
                'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_Percent',
                'ATR', 'CCI', 'Stoch_K', 'ADX', 'OBV',
                'VWAP', 'ROC', 'MFI', 'Williams_R', 'Parabolic_SAR'
            ]
            
            feature_vector = []
            for feature in key_features:
                if feature in latest_values:
                    feature_vector.append(latest_values[feature])
                else:
                    feature_vector.append(0.0)  # 누락된 지표는 0으로 대체
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"주요 특징 추출 실패: {e}")
            return np.zeros(15, dtype=np.float32)
    
    def extract_features_batch(self, ohlcv_batch: List[pd.DataFrame]) -> np.ndarray:
        """
        배치 단위 기술적 특징 추출
        
        Args:
            ohlcv_batch: OHLCV 데이터프레임 리스트
            
        Returns:
            특징 벡터 배열 (batch_size, 15)
        """
        try:
            features_list = []
            
            for ohlcv_data in ohlcv_batch:
                features = self.extract_features(ohlcv_data)
                features_list.append(features)
            
            return np.array(features_list)
            
        except Exception as e:
            self.logger.error(f"배치 기술적 특징 추출 실패: {e}")
            return np.zeros((len(ohlcv_batch), 15), dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """특징 이름 리스트 반환"""
        return [
            'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_Percent',
            'ATR', 'CCI', 'Stoch_K', 'ADX', 'OBV',
            'VWAP', 'ROC', 'MFI', 'Williams_R', 'Parabolic_SAR'
        ]
    
    def get_feature_dimension(self) -> int:
        """특징 차원 반환"""
        return 15


# 편의 함수들
def create_technical_extractor(config: Dict) -> TechnicalFeatureExtractor:
    """기술적 특징 추출기 생성 편의 함수"""
    return TechnicalFeatureExtractor(config)


def extract_technical_features(ohlcv_data: pd.DataFrame, config: Dict) -> np.ndarray:
    """기술적 특징 추출 편의 함수"""
    extractor = create_technical_extractor(config)
    return extractor.extract_features(ohlcv_data)
