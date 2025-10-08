"""
다중 조건 기반 거래 신호 필터링 시스템
Multi-Condition Based Trading Signal Filtering System

여러 조건을 종합적으로 고려한 거래 신호 필터링 및 검증
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """거래 신호 타입"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(Enum):
    """신호 강도"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class SignalFilter:
    """거래 신호 필터링 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        거래 신호 필터링 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 필터링 조건 파라미터
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
        self.min_signal_strength = config.get('min_signal_strength', 0.5)
        self.max_risk_level = config.get('max_risk_level', 0.7)
        self.min_volume_ratio = config.get('min_volume_ratio', 0.8)
        
        # 기술적 지표 필터링
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.macd_signal_threshold = config.get('macd_signal_threshold', 0.001)
        self.bollinger_band_threshold = config.get('bollinger_band_threshold', 0.02)
        
        # 시장 상황 필터링
        self.volatility_threshold_high = config.get('volatility_threshold_high', 0.25)
        self.volatility_threshold_low = config.get('volatility_threshold_low', 0.05)
        self.trend_strength_threshold = config.get('trend_strength_threshold', 0.6)
        
        # 리스크 관리 필터링
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.05)
        self.max_position_size = config.get('max_position_size', 0.12)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        
        # 신호 히스토리
        self.signal_history = []
        self.filtered_signals = []
        
        logger.info(f"거래 신호 필터링 시스템 초기화 완료")
        logger.info(f"최소 신뢰도 임계값: {self.min_confidence_threshold}")
        logger.info(f"최소 신호 강도: {self.min_signal_strength}")
        logger.info(f"최대 리스크 수준: {self.max_risk_level}")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        기술적 지표 계산
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            기술적 지표 딕셔너리
        """
        try:
            logger.info("기술적 지표 계산 시작")
            
            if len(data) < 20:
                return {'error': 'Insufficient data for technical indicators'}
            
            # RSI 계산
            rsi = self._calculate_rsi(data['Close'], period=14)
            
            # MACD 계산
            macd_line, macd_signal, macd_histogram = self._calculate_macd(data['Close'])
            
            # 볼린저 밴드 계산
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
            
            # 이동평균 계산
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            
            # 거래량 지표
            volume_sma = data['Volume'].rolling(window=20).mean()
            volume_ratio = data['Volume'] / volume_sma
            
            # 변동성 계산
            volatility = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # 최근 값들
            latest_idx = -1
            
            indicators = {
                'rsi': rsi.iloc[latest_idx] if not pd.isna(rsi.iloc[latest_idx]) else 50,
                'macd_line': macd_line.iloc[latest_idx] if not pd.isna(macd_line.iloc[latest_idx]) else 0,
                'macd_signal': macd_signal.iloc[latest_idx] if not pd.isna(macd_signal.iloc[latest_idx]) else 0,
                'macd_histogram': macd_histogram.iloc[latest_idx] if not pd.isna(macd_histogram.iloc[latest_idx]) else 0,
                'bb_upper': bb_upper.iloc[latest_idx] if not pd.isna(bb_upper.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'bb_middle': bb_middle.iloc[latest_idx] if not pd.isna(bb_middle.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'bb_lower': bb_lower.iloc[latest_idx] if not pd.isna(bb_lower.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'sma_20': sma_20.iloc[latest_idx] if not pd.isna(sma_20.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'sma_50': sma_50.iloc[latest_idx] if not pd.isna(sma_50.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'ema_12': ema_12.iloc[latest_idx] if not pd.isna(ema_12.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'ema_26': ema_26.iloc[latest_idx] if not pd.isna(ema_26.iloc[latest_idx]) else data['Close'].iloc[latest_idx],
                'volume_ratio': volume_ratio.iloc[latest_idx] if not pd.isna(volume_ratio.iloc[latest_idx]) else 1.0,
                'volatility': volatility.iloc[latest_idx] if not pd.isna(volatility.iloc[latest_idx]) else 0.1,
                'current_price': data['Close'].iloc[latest_idx]
            }
            
            logger.info(f"기술적 지표 계산 완료")
            
            return indicators
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"RSI 계산 실패: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
        except Exception as e:
            logger.error(f"MACD 계산 실패: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 실패: {e}")
            return prices, prices, prices
    
    def apply_technical_filters(self, signal: Dict[str, Any], 
                              indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        기술적 지표 기반 필터링 적용
        
        Args:
            signal: 원본 거래 신호
            indicators: 기술적 지표
            
        Returns:
            필터링된 신호
        """
        try:
            logger.info("기술적 지표 기반 필터링 적용")
            
            if 'error' in indicators:
                return {'error': indicators['error']}
            
            filtered_signal = signal.copy()
            filter_results = []
            
            # RSI 필터링
            rsi = indicators['rsi']
            if signal['action'] == SignalType.BUY.value:
                if rsi > self.rsi_overbought:
                    filter_results.append({
                        'filter': 'rsi_overbought',
                        'passed': False,
                        'reason': f'RSI overbought: {rsi:.2f} > {self.rsi_overbought}'
                    })
                else:
                    filter_results.append({
                        'filter': 'rsi_overbought',
                        'passed': True,
                        'reason': f'RSI acceptable: {rsi:.2f} <= {self.rsi_overbought}'
                    })
            elif signal['action'] == SignalType.SELL.value:
                if rsi < self.rsi_oversold:
                    filter_results.append({
                        'filter': 'rsi_oversold',
                        'passed': False,
                        'reason': f'RSI oversold: {rsi:.2f} < {self.rsi_oversold}'
                    })
                else:
                    filter_results.append({
                        'filter': 'rsi_oversold',
                        'passed': True,
                        'reason': f'RSI acceptable: {rsi:.2f} >= {self.rsi_oversold}'
                    })
            
            # MACD 필터링
            macd_histogram = indicators['macd_histogram']
            if signal['action'] == SignalType.BUY.value:
                if macd_histogram < -self.macd_signal_threshold:
                    filter_results.append({
                        'filter': 'macd_bearish',
                        'passed': False,
                        'reason': f'MACD bearish: {macd_histogram:.4f} < {-self.macd_signal_threshold}'
                    })
                else:
                    filter_results.append({
                        'filter': 'macd_bearish',
                        'passed': True,
                        'reason': f'MACD acceptable: {macd_histogram:.4f} >= {-self.macd_signal_threshold}'
                    })
            elif signal['action'] == SignalType.SELL.value:
                if macd_histogram > self.macd_signal_threshold:
                    filter_results.append({
                        'filter': 'macd_bullish',
                        'passed': False,
                        'reason': f'MACD bullish: {macd_histogram:.4f} > {self.macd_signal_threshold}'
                    })
                else:
                    filter_results.append({
                        'filter': 'macd_bullish',
                        'passed': True,
                        'reason': f'MACD acceptable: {macd_histogram:.4f} <= {self.macd_signal_threshold}'
                    })
            
            # 볼린저 밴드 필터링
            current_price = indicators['current_price']
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            
            if signal['action'] == SignalType.BUY.value:
                if current_price > bb_upper * (1 + self.bollinger_band_threshold):
                    filter_results.append({
                        'filter': 'bb_upper_band',
                        'passed': False,
                        'reason': f'Price above upper band: {current_price:.2f} > {bb_upper * (1 + self.bollinger_band_threshold):.2f}'
                    })
                else:
                    filter_results.append({
                        'filter': 'bb_upper_band',
                        'passed': True,
                        'reason': f'Price within upper band range'
                    })
            elif signal['action'] == SignalType.SELL.value:
                if current_price < bb_lower * (1 - self.bollinger_band_threshold):
                    filter_results.append({
                        'filter': 'bb_lower_band',
                        'passed': False,
                        'reason': f'Price below lower band: {current_price:.2f} < {bb_lower * (1 - self.bollinger_band_threshold):.2f}'
                    })
                else:
                    filter_results.append({
                        'filter': 'bb_lower_band',
                        'passed': True,
                        'reason': f'Price within lower band range'
                    })
            
            # 거래량 필터링
            volume_ratio = indicators['volume_ratio']
            if volume_ratio < self.min_volume_ratio:
                filter_results.append({
                    'filter': 'volume_insufficient',
                    'passed': False,
                    'reason': f'Insufficient volume: {volume_ratio:.2f} < {self.min_volume_ratio}'
                })
            else:
                filter_results.append({
                    'filter': 'volume_insufficient',
                    'passed': True,
                    'reason': f'Volume sufficient: {volume_ratio:.2f} >= {self.min_volume_ratio}'
                })
            
            # 필터링 결과 종합
            passed_filters = sum(1 for result in filter_results if result['passed'])
            total_filters = len(filter_results)
            filter_score = passed_filters / total_filters if total_filters > 0 else 0
            
            filtered_signal['technical_filters'] = {
                'results': filter_results,
                'passed_count': passed_filters,
                'total_count': total_filters,
                'filter_score': filter_score,
                'passed': filter_score >= 0.6  # 60% 이상 통과해야 함
            }
            
            logger.info(f"기술적 지표 기반 필터링 완료: {passed_filters}/{total_filters} 통과")
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"기술적 지표 기반 필터링 실패: {e}")
            return {'error': str(e)}
    
    def apply_market_condition_filters(self, signal: Dict[str, Any],
                                     market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        시장 상황 기반 필터링 적용
        
        Args:
            signal: 원본 거래 신호
            market_conditions: 시장 상황 정보
            
        Returns:
            필터링된 신호
        """
        try:
            logger.info("시장 상황 기반 필터링 적용")
            
            filtered_signal = signal.copy()
            filter_results = []
            
            # 변동성 필터링
            volatility = market_conditions.get('volatility', 0.1)
            if volatility > self.volatility_threshold_high:
                filter_results.append({
                    'filter': 'high_volatility',
                    'passed': False,
                    'reason': f'High volatility: {volatility:.3f} > {self.volatility_threshold_high}'
                })
            else:
                filter_results.append({
                    'filter': 'high_volatility',
                    'passed': True,
                    'reason': f'Volatility acceptable: {volatility:.3f} <= {self.volatility_threshold_high}'
                })
            
            # 트렌드 강도 필터링
            trend_strength = market_conditions.get('trend_strength', 0.5)
            if trend_strength < self.trend_strength_threshold:
                filter_results.append({
                    'filter': 'weak_trend',
                    'passed': False,
                    'reason': f'Weak trend: {trend_strength:.3f} < {self.trend_strength_threshold}'
                })
            else:
                filter_results.append({
                    'filter': 'weak_trend',
                    'passed': True,
                    'reason': f'Trend strength acceptable: {trend_strength:.3f} >= {self.trend_strength_threshold}'
                })
            
            # 시장 체제 필터링
            market_regime = market_conditions.get('market_regime', 'sideways')
            if market_regime == 'bear' and signal['action'] == SignalType.BUY.value:
                filter_results.append({
                    'filter': 'bear_market_buy',
                    'passed': False,
                    'reason': 'Buy signal in bear market'
                })
            elif market_regime == 'bull' and signal['action'] == SignalType.SELL.value:
                filter_results.append({
                    'filter': 'bull_market_sell',
                    'passed': False,
                    'reason': 'Sell signal in bull market'
                })
            else:
                filter_results.append({
                    'filter': 'market_regime',
                    'passed': True,
                    'reason': f'Signal appropriate for {market_regime} market'
                })
            
            # 필터링 결과 종합
            passed_filters = sum(1 for result in filter_results if result['passed'])
            total_filters = len(filter_results)
            filter_score = passed_filters / total_filters if total_filters > 0 else 0
            
            filtered_signal['market_condition_filters'] = {
                'results': filter_results,
                'passed_count': passed_filters,
                'total_count': total_filters,
                'filter_score': filter_score,
                'passed': filter_score >= 0.5  # 50% 이상 통과해야 함
            }
            
            logger.info(f"시장 상황 기반 필터링 완료: {passed_filters}/{total_filters} 통과")
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"시장 상황 기반 필터링 실패: {e}")
            return {'error': str(e)}
    
    def apply_risk_management_filters(self, signal: Dict[str, Any],
                                    portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        리스크 관리 기반 필터링 적용
        
        Args:
            signal: 원본 거래 신호
            portfolio_state: 포트폴리오 상태
            
        Returns:
            필터링된 신호
        """
        try:
            logger.info("리스크 관리 기반 필터링 적용")
            
            filtered_signal = signal.copy()
            filter_results = []
            
            # 최대 낙폭 필터링
            current_drawdown = portfolio_state.get('current_drawdown', 0)
            if current_drawdown < -self.max_drawdown_threshold:
                filter_results.append({
                    'filter': 'max_drawdown',
                    'passed': False,
                    'reason': f'Max drawdown exceeded: {current_drawdown:.3f} < {-self.max_drawdown_threshold}'
                })
            else:
                filter_results.append({
                    'filter': 'max_drawdown',
                    'passed': True,
                    'reason': f'Drawdown acceptable: {current_drawdown:.3f} >= {-self.max_drawdown_threshold}'
                })
            
            # 포지션 크기 필터링
            current_position_size = portfolio_state.get('current_position_size', 0)
            if signal['action'] == SignalType.BUY.value:
                if current_position_size >= self.max_position_size:
                    filter_results.append({
                        'filter': 'max_position_size',
                        'passed': False,
                        'reason': f'Max position size reached: {current_position_size:.3f} >= {self.max_position_size}'
                    })
                else:
                    filter_results.append({
                        'filter': 'max_position_size',
                        'passed': True,
                        'reason': f'Position size acceptable: {current_position_size:.3f} < {self.max_position_size}'
                    })
            
            # 상관관계 필터링
            correlation = portfolio_state.get('correlation_with_market', 0)
            if abs(correlation) > self.correlation_threshold:
                filter_results.append({
                    'filter': 'high_correlation',
                    'passed': False,
                    'reason': f'High correlation with market: {abs(correlation):.3f} > {self.correlation_threshold}'
                })
            else:
                filter_results.append({
                    'filter': 'high_correlation',
                    'passed': True,
                    'reason': f'Correlation acceptable: {abs(correlation):.3f} <= {self.correlation_threshold}'
                })
            
            # 필터링 결과 종합
            passed_filters = sum(1 for result in filter_results if result['passed'])
            total_filters = len(filter_results)
            filter_score = passed_filters / total_filters if total_filters > 0 else 0
            
            filtered_signal['risk_management_filters'] = {
                'results': filter_results,
                'passed_count': passed_filters,
                'total_count': total_filters,
                'filter_score': filter_score,
                'passed': filter_score >= 0.67  # 67% 이상 통과해야 함
            }
            
            logger.info(f"리스크 관리 기반 필터링 완료: {passed_filters}/{total_filters} 통과")
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"리스크 관리 기반 필터링 실패: {e}")
            return {'error': str(e)}
    
    def apply_comprehensive_filtering(self, signal: Dict[str, Any],
                                    technical_indicators: Dict[str, Any],
                                    market_conditions: Dict[str, Any],
                                    portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        종합적 필터링 적용 (모든 필터 통합)
        
        Args:
            signal: 원본 거래 신호
            technical_indicators: 기술적 지표
            market_conditions: 시장 상황
            portfolio_state: 포트폴리오 상태
            
        Returns:
            종합적으로 필터링된 신호
        """
        try:
            logger.info("종합적 필터링 적용 시작")
            
            # 기본 신호 검증
            if not self._validate_signal(signal):
                return {'error': 'Invalid signal format'}
            
            # 각 필터 적용
            filtered_signal = signal.copy()
            
            # 기술적 지표 필터링
            filtered_signal = self.apply_technical_filters(filtered_signal, technical_indicators)
            if 'error' in filtered_signal:
                return filtered_signal
            
            # 시장 상황 필터링
            filtered_signal = self.apply_market_condition_filters(filtered_signal, market_conditions)
            if 'error' in filtered_signal:
                return filtered_signal
            
            # 리스크 관리 필터링
            filtered_signal = self.apply_risk_management_filters(filtered_signal, portfolio_state)
            if 'error' in filtered_signal:
                return filtered_signal
            
            # 종합 필터링 결과 계산
            filter_scores = []
            filter_passed = []
            
            for filter_type in ['technical_filters', 'market_condition_filters', 'risk_management_filters']:
                if filter_type in filtered_signal:
                    filter_data = filtered_signal[filter_type]
                    filter_scores.append(filter_data['filter_score'])
                    filter_passed.append(filter_data['passed'])
            
            # 종합 점수 계산
            overall_filter_score = np.mean(filter_scores) if filter_scores else 0
            overall_passed = all(filter_passed) if filter_passed else False
            
            # 신호 강도 조정
            original_strength = signal.get('strength', 0.5)
            adjusted_strength = original_strength * overall_filter_score
            
            # 최종 신호 결정
            if overall_passed and adjusted_strength >= self.min_signal_strength:
                final_action = signal['action']
                final_strength = adjusted_strength
                signal_status = 'approved'
            else:
                final_action = SignalType.HOLD.value
                final_strength = 0.0
                signal_status = 'rejected'
            
            # 최종 결과 구성
            filtered_signal['comprehensive_filtering'] = {
                'overall_filter_score': overall_filter_score,
                'overall_passed': overall_passed,
                'original_strength': original_strength,
                'adjusted_strength': adjusted_strength,
                'final_action': final_action,
                'final_strength': final_strength,
                'signal_status': signal_status,
                'filtering_timestamp': datetime.now().isoformat()
            }
            
            # 신호 히스토리 업데이트
            self.signal_history.append(filtered_signal)
            if signal_status == 'approved':
                self.filtered_signals.append(filtered_signal)
            
            logger.info(f"종합적 필터링 완료")
            logger.info(f"  종합 점수: {overall_filter_score:.3f}")
            logger.info(f"  최종 액션: {final_action}")
            logger.info(f"  신호 상태: {signal_status}")
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"종합적 필터링 실패: {e}")
            return {'error': str(e)}
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """신호 유효성 검증"""
        try:
            required_fields = ['action', 'confidence', 'strength']
            return all(field in signal for field in required_fields)
        except Exception as e:
            logger.error(f"신호 유효성 검증 실패: {e}")
            return False
    
    def get_filtering_statistics(self) -> Dict[str, Any]:
        """필터링 통계 정보 반환"""
        try:
            if not self.signal_history:
                return {'message': 'No signal history available'}
            
            total_signals = len(self.signal_history)
            approved_signals = len(self.filtered_signals)
            approval_rate = approved_signals / total_signals if total_signals > 0 else 0
            
            # 필터별 통과율 계산
            filter_pass_rates = {}
            for filter_type in ['technical_filters', 'market_condition_filters', 'risk_management_filters']:
                pass_count = 0
                for signal in self.signal_history:
                    if filter_type in signal and signal[filter_type]['passed']:
                        pass_count += 1
                filter_pass_rates[filter_type] = pass_count / total_signals if total_signals > 0 else 0
            
            statistics = {
                'total_signals': total_signals,
                'approved_signals': approved_signals,
                'approval_rate': approval_rate,
                'filter_pass_rates': filter_pass_rates,
                'recent_signals': len(self.signal_history[-10:]) if len(self.signal_history) >= 10 else len(self.signal_history)
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"필터링 통계 계산 실패: {e}")
            return {'error': str(e)}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("다중 조건 기반 거래 신호 필터링 시스템 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {
            'min_confidence_threshold': 0.6,
            'min_signal_strength': 0.5,
            'max_risk_level': 0.7,
            'min_volume_ratio': 0.8,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_signal_threshold': 0.001,
            'bollinger_band_threshold': 0.02,
            'volatility_threshold_high': 0.25,
            'volatility_threshold_low': 0.05,
            'trend_strength_threshold': 0.6,
            'max_drawdown_threshold': 0.05,
            'max_position_size': 0.12,
            'correlation_threshold': 0.8
        }
        
        # 거래 신호 필터링 시스템 생성
        signal_filter = SignalFilter(config)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        # OHLCV 데이터 생성
        test_data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, n_samples),
            'High': np.random.uniform(100, 120, n_samples),
            'Low': np.random.uniform(80, 100, n_samples),
            'Close': np.random.uniform(90, 110, n_samples),
            'Volume': np.random.uniform(1000, 10000, n_samples)
        })
        
        # 기술적 지표 계산
        indicators = signal_filter.calculate_technical_indicators(test_data)
        print(f"기술적 지표 계산 완료")
        if 'error' not in indicators:
            print(f"  RSI: {indicators['rsi']:.2f}")
            print(f"  MACD: {indicators['macd_histogram']:.4f}")
            print(f"  거래량 비율: {indicators['volume_ratio']:.2f}")
            print(f"  변동성: {indicators['volatility']:.3f}")
        
        # 테스트 신호 생성
        test_signals = [
            {
                'action': SignalType.BUY.value,
                'confidence': 0.8,
                'strength': 0.7,
                'asset': 'BTC',
                'timestamp': datetime.now().isoformat()
            },
            {
                'action': SignalType.SELL.value,
                'confidence': 0.6,
                'strength': 0.5,
                'asset': 'ETH',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # 시장 상황 및 포트폴리오 상태
        market_conditions = {
            'volatility': 0.15,
            'trend_strength': 0.7,
            'market_regime': 'sideways'
        }
        
        portfolio_state = {
            'current_drawdown': -0.02,
            'current_position_size': 0.08,
            'correlation_with_market': 0.6
        }
        
        # 각 신호에 대해 종합적 필터링 적용
        for i, signal in enumerate(test_signals):
            print(f"\n신호 {i+1} 필터링:")
            print(f"  원본 액션: {signal['action']}")
            print(f"  신뢰도: {signal['confidence']:.2f}")
            print(f"  강도: {signal['strength']:.2f}")
            
            filtered_signal = signal_filter.apply_comprehensive_filtering(
                signal, indicators, market_conditions, portfolio_state
            )
            
            if 'error' not in filtered_signal:
                comprehensive = filtered_signal['comprehensive_filtering']
                print(f"  최종 액션: {comprehensive['final_action']}")
                print(f"  최종 강도: {comprehensive['final_strength']:.2f}")
                print(f"  신호 상태: {comprehensive['signal_status']}")
                print(f"  종합 점수: {comprehensive['overall_filter_score']:.3f}")
            else:
                print(f"  오류: {filtered_signal['error']}")
        
        # 필터링 통계
        statistics = signal_filter.get_filtering_statistics()
        if 'error' not in statistics:
            print(f"\n필터링 통계:")
            print(f"  총 신호 수: {statistics['total_signals']}")
            print(f"  승인된 신호 수: {statistics['approved_signals']}")
            print(f"  승인율: {statistics['approval_rate']:.2%}")
        
        print(f"\n[성공] 다중 조건 기반 거래 신호 필터링 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 다중 조건 기반 거래 신호 필터링 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
