"""
자산별 특화 거래 전략 시스템
Asset-Specific Trading Strategy System

BTC, ETH, AAPL, MSFT 등 자산별 특성을 고려한 맞춤형 거래 전략
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

class AssetType(Enum):
    """자산 타입"""
    CRYPTOCURRENCY = "cryptocurrency"
    STOCK = "stock"
    COMMODITY = "commodity"
    FOREX = "forex"

class TradingStrategy(Enum):
    """거래 전략 타입"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    SCALPING = "scalping"

class AssetSpecificStrategy:
    """자산별 특화 거래 전략 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        자산별 특화 거래 전략 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 자산별 특성 정의
        self.asset_characteristics = {
            'BTC': {
                'type': AssetType.CRYPTOCURRENCY,
                'volatility': 'high',
                'liquidity': 'high',
                'correlation_with_market': 'medium',
                'trading_hours': '24/7',
                'preferred_strategies': [TradingStrategy.MOMENTUM, TradingStrategy.BREAKOUT],
                'risk_level': 'high',
                'position_sizing_multiplier': 0.8,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0
            },
            'ETH': {
                'type': AssetType.CRYPTOCURRENCY,
                'volatility': 'high',
                'liquidity': 'high',
                'correlation_with_market': 'high',
                'trading_hours': '24/7',
                'preferred_strategies': [TradingStrategy.MOMENTUM, TradingStrategy.TREND_FOLLOWING],
                'risk_level': 'high',
                'position_sizing_multiplier': 0.9,
                'stop_loss_multiplier': 1.3,
                'take_profit_multiplier': 1.8
            },
            'AAPL': {
                'type': AssetType.STOCK,
                'volatility': 'medium',
                'liquidity': 'high',
                'correlation_with_market': 'high',
                'trading_hours': 'market_hours',
                'preferred_strategies': [TradingStrategy.MEAN_REVERSION, TradingStrategy.TREND_FOLLOWING],
                'risk_level': 'medium',
                'position_sizing_multiplier': 1.2,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.5
            },
            'MSFT': {
                'type': AssetType.STOCK,
                'volatility': 'medium',
                'liquidity': 'high',
                'correlation_with_market': 'high',
                'trading_hours': 'market_hours',
                'preferred_strategies': [TradingStrategy.MEAN_REVERSION, TradingStrategy.TREND_FOLLOWING],
                'risk_level': 'medium',
                'position_sizing_multiplier': 1.1,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.4
            }
        }
        
        # 전략별 파라미터
        self.strategy_parameters = {
            TradingStrategy.MOMENTUM: {
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.5,
                'trend_confirmation_period': 5
            },
            TradingStrategy.MEAN_REVERSION: {
                'lookback_period': 50,
                'deviation_threshold': 2.0,
                'reversion_speed': 0.1,
                'support_resistance_period': 20
            },
            TradingStrategy.TREND_FOLLOWING: {
                'trend_period': 30,
                'trend_strength_threshold': 0.6,
                'pullback_threshold': 0.1,
                'trend_confirmation_period': 10
            },
            TradingStrategy.BREAKOUT: {
                'breakout_period': 20,
                'breakout_threshold': 0.01,
                'volume_confirmation': 2.0,
                'false_breakout_filter': 0.005
            },
            TradingStrategy.SCALPING: {
                'scalp_period': 5,
                'profit_target': 0.005,
                'stop_loss': 0.003,
                'max_hold_time': 30  # minutes
            }
        }
        
        # 자산별 성능 추적
        self.asset_performance = {}
        self.strategy_performance = {}
        
        logger.info(f"자산별 특화 거래 전략 시스템 초기화 완료")
        logger.info(f"지원 자산: {list(self.asset_characteristics.keys())}")
        logger.info(f"지원 전략: {[strategy.value for strategy in TradingStrategy]}")
    
    def get_asset_characteristics(self, asset: str) -> Dict[str, Any]:
        """
        자산 특성 정보 반환
        
        Args:
            asset: 자산 코드
            
        Returns:
            자산 특성 딕셔너리
        """
        try:
            if asset not in self.asset_characteristics:
                logger.warning(f"Unknown asset: {asset}")
                return {}
            
            return self.asset_characteristics[asset]
            
        except Exception as e:
            logger.error(f"자산 특성 조회 실패: {e}")
            return {}
    
    def calculate_asset_specific_indicators(self, asset: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        자산별 특화 지표 계산
        
        Args:
            asset: 자산 코드
            data: OHLCV 데이터
            
        Returns:
            자산별 특화 지표
        """
        try:
            logger.info(f"{asset} 자산별 특화 지표 계산 시작")
            
            if asset not in self.asset_characteristics:
                return {'error': f'Unknown asset: {asset}'}
            
            asset_char = self.asset_characteristics[asset]
            indicators = {}
            
            # 기본 기술적 지표
            indicators.update(self._calculate_basic_indicators(data))
            
            # 자산 타입별 특화 지표
            if asset_char['type'] == AssetType.CRYPTOCURRENCY:
                indicators.update(self._calculate_crypto_specific_indicators(data))
            elif asset_char['type'] == AssetType.STOCK:
                indicators.update(self._calculate_stock_specific_indicators(data))
            
            # 변동성 특성에 따른 지표
            if asset_char['volatility'] == 'high':
                indicators.update(self._calculate_high_volatility_indicators(data))
            elif asset_char['volatility'] == 'medium':
                indicators.update(self._calculate_medium_volatility_indicators(data))
            
            logger.info(f"{asset} 자산별 특화 지표 계산 완료")
            
            return indicators
            
        except Exception as e:
            logger.error(f"{asset} 자산별 특화 지표 계산 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기본 기술적 지표 계산"""
        try:
            indicators = {}
            
            # 이동평균
            indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            indicators['ema_12'] = data['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            macd_signal = pd.Series([macd_line]).ewm(span=9).mean().iloc[-1]
            indicators['macd'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_line - macd_signal
            
            # 볼린저 밴드
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            indicators['bb_upper'] = (bb_middle + (bb_std * 2)).iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            indicators['bb_lower'] = (bb_middle - (bb_std * 2)).iloc[-1]
            
            # 거래량 지표
            volume_sma = data['Volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['Volume'].iloc[-1] / volume_sma.iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"기본 지표 계산 실패: {e}")
            return {}
    
    def _calculate_crypto_specific_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """암호화폐 특화 지표 계산"""
        try:
            indicators = {}
            
            # 24시간 변동률
            indicators['price_change_24h'] = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            
            # 고가-저가 비율
            high_low_ratio = (data['High'].rolling(window=24).max() / data['Low'].rolling(window=24).min()).iloc[-1]
            indicators['high_low_ratio'] = high_low_ratio
            
            # 거래량 가격 추세 (VPT)
            vpt = (data['Volume'] * data['Close'].pct_change()).cumsum()
            indicators['vpt'] = vpt.iloc[-1]
            
            # 암호화폐 변동성 지수
            returns = data['Close'].pct_change()
            crypto_volatility = returns.rolling(window=24).std() * np.sqrt(24)
            indicators['crypto_volatility'] = crypto_volatility.iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"암호화폐 특화 지표 계산 실패: {e}")
            return {}
    
    def _calculate_stock_specific_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """주식 특화 지표 계산"""
        try:
            indicators = {}
            
            # 가격 대비 거래량 비율
            price_volume_ratio = data['Close'].iloc[-1] / data['Volume'].iloc[-1]
            indicators['price_volume_ratio'] = price_volume_ratio
            
            # 주가 모멘텀
            momentum_5d = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6]
            momentum_20d = (data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21]
            indicators['momentum_5d'] = momentum_5d
            indicators['momentum_20d'] = momentum_20d
            
            # 주가 안정성 지수
            price_stability = 1 / (data['Close'].pct_change().rolling(window=20).std().iloc[-1] + 0.001)
            indicators['price_stability'] = price_stability
            
            return indicators
            
        except Exception as e:
            logger.error(f"주식 특화 지표 계산 실패: {e}")
            return {}
    
    def _calculate_high_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """고변동성 자산 특화 지표 계산"""
        try:
            indicators = {}
            
            # 변동성 돌파 전략 지표
            atr = self._calculate_atr(data, period=14)
            indicators['atr'] = atr
            indicators['atr_ratio'] = atr / data['Close'].iloc[-1]
            
            # 급격한 가격 변동 감지
            price_spike = abs(data['Close'].pct_change().iloc[-1])
            indicators['price_spike'] = price_spike
            
            # 변동성 클러스터링
            volatility_cluster = data['Close'].pct_change().rolling(window=5).std().iloc[-1]
            indicators['volatility_cluster'] = volatility_cluster
            
            return indicators
            
        except Exception as e:
            logger.error(f"고변동성 지표 계산 실패: {e}")
            return {}
    
    def _calculate_medium_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """중변동성 자산 특화 지표 계산"""
        try:
            indicators = {}
            
            # 평균 회귵 지표
            z_score = (data['Close'].iloc[-1] - data['Close'].rolling(window=20).mean().iloc[-1]) / data['Close'].rolling(window=20).std().iloc[-1]
            indicators['z_score'] = z_score
            
            # 지지/저항 레벨
            resistance = data['High'].rolling(window=20).max().iloc[-1]
            support = data['Low'].rolling(window=20).min().iloc[-1]
            indicators['resistance'] = resistance
            indicators['support'] = support
            indicators['support_resistance_ratio'] = (data['Close'].iloc[-1] - support) / (resistance - support)
            
            return indicators
            
        except Exception as e:
            logger.error(f"중변동성 지표 계산 실패: {e}")
            return {}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Average True Range 계산"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            return atr.iloc[-1]
            
        except Exception as e:
            logger.error(f"ATR 계산 실패: {e}")
            return 0.0
    
    def generate_asset_specific_signal(self, asset: str, data: pd.DataFrame,
                                     market_regime: str = 'sideways') -> Dict[str, Any]:
        """
        자산별 특화 거래 신호 생성
        
        Args:
            asset: 자산 코드
            data: OHLCV 데이터
            market_regime: 시장 체제
            
        Returns:
            자산별 특화 거래 신호
        """
        try:
            logger.info(f"{asset} 자산별 특화 거래 신호 생성 시작")
            
            if asset not in self.asset_characteristics:
                return {'error': f'Unknown asset: {asset}'}
            
            asset_char = self.asset_characteristics[asset]
            
            # 자산별 특화 지표 계산
            indicators = self.calculate_asset_specific_indicators(asset, data)
            if 'error' in indicators:
                return indicators
            
            # 선호 전략에 따른 신호 생성
            signals = []
            for strategy in asset_char['preferred_strategies']:
                signal = self._generate_strategy_signal(strategy, indicators, asset_char, market_regime)
                if signal and 'error' not in signal:
                    signals.append(signal)
            
            # 신호 통합 및 최종 결정
            if signals:
                final_signal = self._integrate_signals(signals, asset_char)
            else:
                final_signal = {
                    'action': 'hold',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'reason': 'No valid signals generated'
                }
            
            # 자산별 특성 반영
            final_signal = self._apply_asset_specific_adjustments(final_signal, asset_char, indicators)
            
            final_signal.update({
                'asset': asset,
                'asset_type': asset_char['type'].value,
                'market_regime': market_regime,
                'timestamp': datetime.now().isoformat(),
                'indicators_used': list(indicators.keys())
            })
            
            logger.info(f"{asset} 자산별 특화 거래 신호 생성 완료")
            logger.info(f"  최종 액션: {final_signal['action']}")
            logger.info(f"  신뢰도: {final_signal['confidence']:.3f}")
            logger.info(f"  강도: {final_signal['strength']:.3f}")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"{asset} 자산별 특화 거래 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_strategy_signal(self, strategy: TradingStrategy, indicators: Dict[str, Any],
                                asset_char: Dict[str, Any], market_regime: str) -> Optional[Dict[str, Any]]:
        """전략별 신호 생성"""
        try:
            params = self.strategy_parameters[strategy]
            
            if strategy == TradingStrategy.MOMENTUM:
                return self._generate_momentum_signal(indicators, params, asset_char)
            elif strategy == TradingStrategy.MEAN_REVERSION:
                return self._generate_mean_reversion_signal(indicators, params, asset_char)
            elif strategy == TradingStrategy.TREND_FOLLOWING:
                return self._generate_trend_following_signal(indicators, params, asset_char)
            elif strategy == TradingStrategy.BREAKOUT:
                return self._generate_breakout_signal(indicators, params, asset_char)
            elif strategy == TradingStrategy.SCALPING:
                return self._generate_scalping_signal(indicators, params, asset_char)
            else:
                return None
                
        except Exception as e:
            logger.error(f"{strategy.value} 전략 신호 생성 실패: {e}")
            return None
    
    def _generate_momentum_signal(self, indicators: Dict[str, Any], params: Dict[str, Any],
                                asset_char: Dict[str, Any]) -> Dict[str, Any]:
        """모멘텀 전략 신호 생성"""
        try:
            # 모멘텀 계산
            momentum = indicators.get('momentum_5d', 0)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            # 신호 생성
            if momentum > params['momentum_threshold'] and volume_ratio > params['volume_threshold']:
                action = 'buy'
                confidence = min(0.9, abs(momentum) * 10)
                strength = min(0.9, volume_ratio / 2)
                reason = f"Momentum: {momentum:.3f}, Volume: {volume_ratio:.2f}"
            elif momentum < -params['momentum_threshold'] and volume_ratio > params['volume_threshold']:
                action = 'sell'
                confidence = min(0.9, abs(momentum) * 10)
                strength = min(0.9, volume_ratio / 2)
                reason = f"Negative momentum: {momentum:.3f}, Volume: {volume_ratio:.2f}"
            else:
                action = 'hold'
                confidence = 0.0
                strength = 0.0
                reason = "Insufficient momentum or volume"
            
            return {
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'strategy': 'momentum',
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"모멘텀 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_mean_reversion_signal(self, indicators: Dict[str, Any], params: Dict[str, Any],
                                      asset_char: Dict[str, Any]) -> Dict[str, Any]:
        """평균 회귵 전략 신호 생성"""
        try:
            z_score = indicators.get('z_score', 0)
            support_resistance_ratio = indicators.get('support_resistance_ratio', 0.5)
            
            # 신호 생성
            if z_score > params['deviation_threshold']:
                action = 'sell'
                confidence = min(0.9, abs(z_score) / 3)
                strength = min(0.9, abs(z_score) / 4)
                reason = f"Overbought: Z-score {z_score:.2f}"
            elif z_score < -params['deviation_threshold']:
                action = 'buy'
                confidence = min(0.9, abs(z_score) / 3)
                strength = min(0.9, abs(z_score) / 4)
                reason = f"Oversold: Z-score {z_score:.2f}"
            else:
                action = 'hold'
                confidence = 0.0
                strength = 0.0
                reason = f"Within normal range: Z-score {z_score:.2f}"
            
            return {
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'strategy': 'mean_reversion',
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"평균 회귵 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_trend_following_signal(self, indicators: Dict[str, Any], params: Dict[str, Any],
                                       asset_char: Dict[str, Any]) -> Dict[str, Any]:
        """트렌드 추종 전략 신호 생성"""
        try:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            current_price = indicators.get('current_price', 0)
            
            if sma_20 == 0 or sma_50 == 0 or current_price == 0:
                return {'action': 'hold', 'confidence': 0.0, 'strength': 0.0, 'strategy': 'trend_following', 'reason': 'Insufficient data'}
            
            # 트렌드 방향 확인
            trend_direction = 1 if sma_20 > sma_50 else -1
            price_above_sma20 = current_price > sma_20
            
            # 신호 생성
            if trend_direction > 0 and price_above_sma20:
                action = 'buy'
                confidence = 0.7
                strength = 0.6
                reason = f"Uptrend: SMA20 {sma_20:.2f} > SMA50 {sma_50:.2f}"
            elif trend_direction < 0 and not price_above_sma20:
                action = 'sell'
                confidence = 0.7
                strength = 0.6
                reason = f"Downtrend: SMA20 {sma_20:.2f} < SMA50 {sma_50:.2f}"
            else:
                action = 'hold'
                confidence = 0.0
                strength = 0.0
                reason = "No clear trend or price not following trend"
            
            return {
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'strategy': 'trend_following',
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"트렌드 추종 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_breakout_signal(self, indicators: Dict[str, Any], params: Dict[str, Any],
                                asset_char: Dict[str, Any]) -> Dict[str, Any]:
        """돌파 전략 신호 생성"""
        try:
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            current_price = indicators.get('current_price', 0)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            
            if bb_upper == 0 or bb_lower == 0 or current_price == 0:
                return {'action': 'hold', 'confidence': 0.0, 'strength': 0.0, 'strategy': 'breakout', 'reason': 'Insufficient data'}
            
            # 돌파 확인
            upper_breakout = current_price > bb_upper * (1 + params['breakout_threshold'])
            lower_breakout = current_price < bb_lower * (1 - params['breakout_threshold'])
            volume_confirmation = volume_ratio > params['volume_confirmation']
            
            # 신호 생성
            if upper_breakout and volume_confirmation:
                action = 'buy'
                confidence = min(0.9, volume_ratio / 3)
                strength = min(0.9, (current_price - bb_upper) / bb_upper * 10)
                reason = f"Upper breakout: Price {current_price:.2f} > BB Upper {bb_upper:.2f}, Volume {volume_ratio:.2f}"
            elif lower_breakout and volume_confirmation:
                action = 'sell'
                confidence = min(0.9, volume_ratio / 3)
                strength = min(0.9, (bb_lower - current_price) / bb_lower * 10)
                reason = f"Lower breakout: Price {current_price:.2f} < BB Lower {bb_lower:.2f}, Volume {volume_ratio:.2f}"
            else:
                action = 'hold'
                confidence = 0.0
                strength = 0.0
                reason = "No breakout or insufficient volume"
            
            return {
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'strategy': 'breakout',
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"돌파 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_scalping_signal(self, indicators: Dict[str, Any], params: Dict[str, Any],
                                asset_char: Dict[str, Any]) -> Dict[str, Any]:
        """스캘핑 전략 신호 생성"""
        try:
            # 스캘핑은 고빈도 거래이므로 간단한 신호 생성
            rsi = indicators.get('rsi', 50)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            # 신호 생성
            if rsi < 30 and macd_histogram > 0:
                action = 'buy'
                confidence = 0.6
                strength = 0.5
                reason = f"Scalping buy: RSI {rsi:.1f}, MACD {macd_histogram:.4f}"
            elif rsi > 70 and macd_histogram < 0:
                action = 'sell'
                confidence = 0.6
                strength = 0.5
                reason = f"Scalping sell: RSI {rsi:.1f}, MACD {macd_histogram:.4f}"
            else:
                action = 'hold'
                confidence = 0.0
                strength = 0.0
                reason = "No scalping opportunity"
            
            return {
                'action': action,
                'confidence': confidence,
                'strength': strength,
                'strategy': 'scalping',
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"스캘핑 신호 생성 실패: {e}")
            return {'error': str(e)}
    
    def _integrate_signals(self, signals: List[Dict[str, Any]], asset_char: Dict[str, Any]) -> Dict[str, Any]:
        """여러 신호 통합"""
        try:
            if not signals:
                return {'action': 'hold', 'confidence': 0.0, 'strength': 0.0, 'reason': 'No signals'}
            
            # 신호별 가중치 (선호 전략에 더 높은 가중치)
            strategy_weights = {strategy.value: 1.0 for strategy in TradingStrategy}
            for strategy in asset_char['preferred_strategies']:
                strategy_weights[strategy.value] = 1.5
            
            # 가중 평균 계산
            weighted_confidence = 0.0
            weighted_strength = 0.0
            total_weight = 0.0
            
            buy_signals = 0
            sell_signals = 0
            hold_signals = 0
            
            for signal in signals:
                weight = strategy_weights.get(signal.get('strategy', ''), 1.0)
                weighted_confidence += signal.get('confidence', 0) * weight
                weighted_strength += signal.get('strength', 0) * weight
                total_weight += weight
                
                if signal.get('action') == 'buy':
                    buy_signals += 1
                elif signal.get('action') == 'sell':
                    sell_signals += 1
                else:
                    hold_signals += 1
            
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
                final_strength = weighted_strength / total_weight
            else:
                final_confidence = 0.0
                final_strength = 0.0
            
            # 최종 액션 결정
            if buy_signals > sell_signals and buy_signals > hold_signals:
                final_action = 'buy'
            elif sell_signals > buy_signals and sell_signals > hold_signals:
                final_action = 'sell'
            else:
                final_action = 'hold'
            
            # 통합 이유
            reasons = [signal.get('reason', '') for signal in signals]
            final_reason = f"Integrated from {len(signals)} signals: {', '.join(reasons[:2])}"
            
            return {
                'action': final_action,
                'confidence': final_confidence,
                'strength': final_strength,
                'reason': final_reason,
                'signal_count': len(signals),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals
            }
            
        except Exception as e:
            logger.error(f"신호 통합 실패: {e}")
            return {'error': str(e)}
    
    def _apply_asset_specific_adjustments(self, signal: Dict[str, Any], asset_char: Dict[str, Any],
                                        indicators: Dict[str, Any]) -> Dict[str, Any]:
        """자산별 특성에 따른 신호 조정"""
        try:
            adjusted_signal = signal.copy()
            
            # 포지션 크기 조정
            position_multiplier = asset_char.get('position_sizing_multiplier', 1.0)
            adjusted_signal['position_size_multiplier'] = position_multiplier
            
            # 리스크 수준에 따른 신뢰도 조정
            risk_level = asset_char.get('risk_level', 'medium')
            if risk_level == 'high':
                adjusted_signal['confidence'] *= 0.9  # 고위험 자산은 신뢰도 약간 감소
            elif risk_level == 'low':
                adjusted_signal['confidence'] *= 1.1  # 저위험 자산은 신뢰도 약간 증가
            
            # 변동성에 따른 강도 조정
            volatility = asset_char.get('volatility', 'medium')
            if volatility == 'high':
                adjusted_signal['strength'] *= 0.8  # 고변동성 자산은 강도 감소
            elif volatility == 'low':
                adjusted_signal['strength'] *= 1.2  # 저변동성 자산은 강도 증가
            
            # 신뢰도와 강도 범위 제한
            adjusted_signal['confidence'] = max(0.0, min(1.0, adjusted_signal['confidence']))
            adjusted_signal['strength'] = max(0.0, min(1.0, adjusted_signal['strength']))
            
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"자산별 특성 조정 실패: {e}")
            return signal

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("자산별 특화 거래 전략 시스템 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {}
        
        # 자산별 특화 거래 전략 시스템 생성
        asset_strategy = AssetSpecificStrategy(config)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        test_assets = ['BTC', 'ETH', 'AAPL', 'MSFT']
        
        for asset in test_assets:
            print(f"\n{asset} 테스트:")
            
            # 테스트 데이터 생성
            test_data = pd.DataFrame({
                'Open': np.random.uniform(90, 110, n_samples),
                'High': np.random.uniform(100, 120, n_samples),
                'Low': np.random.uniform(80, 100, n_samples),
                'Close': np.random.uniform(90, 110, n_samples),
                'Volume': np.random.uniform(1000, 10000, n_samples)
            })
            
            # 자산 특성 확인
            characteristics = asset_strategy.get_asset_characteristics(asset)
            if characteristics:
                print(f"  자산 타입: {characteristics['type'].value}")
                print(f"  변동성: {characteristics['volatility']}")
                print(f"  선호 전략: {[s.value for s in characteristics['preferred_strategies']]}")
            
            # 자산별 특화 지표 계산
            indicators = asset_strategy.calculate_asset_specific_indicators(asset, test_data)
            if 'error' not in indicators:
                print(f"  RSI: {indicators.get('rsi', 0):.2f}")
                print(f"  MACD: {indicators.get('macd_histogram', 0):.4f}")
                print(f"  거래량 비율: {indicators.get('volume_ratio', 0):.2f}")
            
            # 자산별 특화 신호 생성
            signal = asset_strategy.generate_asset_specific_signal(asset, test_data, 'sideways')
            if 'error' not in signal:
                print(f"  최종 액션: {signal['action']}")
                print(f"  신뢰도: {signal['confidence']:.3f}")
                print(f"  강도: {signal['strength']:.3f}")
                print(f"  포지션 크기 배수: {signal.get('position_size_multiplier', 1.0):.2f}")
                print(f"  신호 수: {signal.get('signal_count', 0)}")
            else:
                print(f"  오류: {signal['error']}")
        
        print(f"\n[성공] 자산별 특화 거래 전략 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 자산별 특화 거래 전략 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
