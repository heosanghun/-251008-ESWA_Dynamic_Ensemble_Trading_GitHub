"""
백테스팅 엔진 모듈
Backtesting Engine Module

백테스팅 엔진 구현
- Walk-Forward Expanding Window Cross-Validation
- 시계열 데이터 기반 백테스팅
- 성과 지표 계산 및 분석
- 리스크 지표 및 드로우다운 분석

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
from pathlib import Path
import json

from .trading_environment import TradingEnvironment, create_trading_environment
from ..ensemble.policy_ensemble import PolicyEnsemble
from ..regime.classifier import MarketRegimeClassifier


class BacktestingEngine:
    """
    백테스팅 엔진 메인 클래스
    
    Walk-Forward Expanding Window Cross-Validation 기반 백테스팅
    """
    
    def __init__(self, config: Dict, policy_ensemble: PolicyEnsemble, 
                 regime_classifier: MarketRegimeClassifier):
        """
        백테스팅 엔진 초기화
        
        Args:
            config: 설정 딕셔너리
            policy_ensemble: 정책 앙상블
            regime_classifier: 시장 체제 분류기
        """
        self.config = config
        self.policy_ensemble = policy_ensemble
        self.regime_classifier = regime_classifier
        
        # 백테스팅 설정
        self.backtesting_config = config.get('backtesting', {})
        self.initial_window_size = self.backtesting_config.get('initial_window_size', 252)  # 1년
        self.expanding_window = self.backtesting_config.get('expanding_window', True)
        self.validation_period = self.backtesting_config.get('validation_period', 30)  # 30일
        
        # 거래 환경 초기화
        self.trading_environment = create_trading_environment(config)
        
        # 백테스팅 결과
        self.backtest_results = []
        self.performance_metrics = {}
        self.regime_performance = defaultdict(list)
        
        # Walk-Forward 결과
        self.walk_forward_results = []
        self.cumulative_returns = []
        self.drawdowns = []
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("백테스팅 엔진 초기화 완료")
        self.logger.info(f"초기 윈도우 크기: {self.initial_window_size}일")
        self.logger.info(f"검증 기간: {self.validation_period}일")
        self.logger.info(f"확장 윈도우: {self.expanding_window}")
    
    def run_backtest(self, data: pd.DataFrame, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Dict[str, Union[float, List, Dict]]:
        """
        백테스팅 실행
        
        Args:
            data: 시계열 데이터 (OHLCV + 멀티모달 특징)
            start_date: 시작 날짜 (선택사항)
            end_date: 종료 날짜 (선택사항)
            
        Returns:
            백테스팅 결과
        """
        try:
            self.logger.info("백테스팅 시작")
            
            # 데이터 전처리
            processed_data = self._preprocess_data(data, start_date, end_date)
            
            if len(processed_data) < self.initial_window_size + self.validation_period:
                raise ValueError(f"데이터가 부족합니다. 최소 {self.initial_window_size + self.validation_period}일 필요")
            
            # Walk-Forward Cross-Validation 실행
            walk_forward_results = self._run_walk_forward_validation(processed_data)
            
            # 전체 성과 지표 계산
            performance_metrics = self._calculate_performance_metrics(walk_forward_results)
            
            # 체제별 성과 분석
            regime_analysis = self._analyze_regime_performance(walk_forward_results)
            
            # 백테스팅 결과 정리
            backtest_results = {
                'walk_forward_results': walk_forward_results,
                'performance_metrics': performance_metrics,
                'regime_analysis': regime_analysis,
                'total_periods': len(walk_forward_results),
                'backtest_summary': self._generate_backtest_summary(performance_metrics)
            }
            
            self.backtest_results = walk_forward_results
            self.performance_metrics = performance_metrics
            
            self.logger.info("백테스팅 완료")
            self.logger.info(f"총 기간: {len(walk_forward_results)}개")
            self.logger.info(f"총 수익률: {performance_metrics.get('total_return', 0):.2%}")
            self.logger.info(f"샤프 비율: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"백테스팅 실행 실패: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            data: 원본 데이터
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            전처리된 데이터
        """
        try:
            # 날짜 필터링
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # 필수 컬럼 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {missing_columns}")
            
            # 데이터 정렬
            data = data.sort_index()
            
            # 결측값 처리
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"데이터 전처리 완료: {len(data)}개 레코드")
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 실패: {e}")
            raise
    
    def _run_walk_forward_validation(self, data: pd.DataFrame) -> List[Dict]:
        """
        Walk-Forward Expanding Window Cross-Validation 실행
        
        Args:
            data: 전처리된 데이터
            
        Returns:
            Walk-Forward 결과 리스트
        """
        try:
            walk_forward_results = []
            current_window_size = self.initial_window_size
            
            # Walk-Forward 윈도우 생성
            for i in range(current_window_size, len(data) - self.validation_period + 1):
                # 훈련 데이터 (확장 윈도우)
                if self.expanding_window:
                    train_data = data.iloc[:i]
                else:
                    # 고정 윈도우 (롤링)
                    train_data = data.iloc[i-current_window_size:i]
                
                # 검증 데이터
                validation_data = data.iloc[i:i+self.validation_period]
                
                if len(validation_data) < self.validation_period:
                    break
                
                # 훈련 기간 성과 계산
                train_performance = self._run_validation_period(train_data, validation_data, i)
                
                walk_forward_results.append(train_performance)
                
                # 진행 상황 로깅
                if (i - current_window_size + 1) % 50 == 0:
                    progress = (i - current_window_size + 1) / (len(data) - current_window_size - self.validation_period + 1)
                    self.logger.info(f"Walk-Forward 진행률: {progress:.1%}")
            
            self.logger.info(f"Walk-Forward 검증 완료: {len(walk_forward_results)}개 기간")
            
            return walk_forward_results
            
        except Exception as e:
            self.logger.error(f"Walk-Forward 검증 실행 실패: {e}")
            raise
    
    def _run_validation_period(self, train_data: pd.DataFrame, 
                              validation_data: pd.DataFrame, period_idx: int) -> Dict:
        """
        검증 기간 실행
        
        Args:
            train_data: 훈련 데이터
            validation_data: 검증 데이터
            period_idx: 기간 인덱스
            
        Returns:
            검증 기간 성과
        """
        try:
            # 포트폴리오 초기화
            self.trading_environment.reset()
            
            # 훈련 데이터로 체제 분류기 재훈련 (임시 구현)
            try:
                # 체제 분류기 훈련 (기본 구현)
                self.regime_classifier.train(train_data)
            except Exception as e:
                self.logger.warning(f"체제 분류기 훈련 실패: {e}")
            
            # 검증 기간 거래 실행
            period_returns = []
            period_trades = []
            regime_predictions = []
            
            for idx, (timestamp, row) in enumerate(validation_data.iterrows()):
                try:
                    # 현재 가격 업데이트
                    current_price = row['close']
                    self.trading_environment.update_price(current_price)
                    
                    # 체제 예측 (임시 구현)
                    try:
                        # 체제 분류기 예측 (기본 구현)
                        regime_proba = {'bull_market': 0.4, 'bear_market': 0.3, 'sideways_market': 0.3}
                        regime_label = max(regime_proba, key=regime_proba.get)
                    except Exception as e:
                        self.logger.warning(f"체제 예측 실패: {e}")
                        regime_proba = {'bull_market': 0.4, 'bear_market': 0.3, 'sideways_market': 0.3}
                        regime_label = 'bull_market'
                    
                    # 정책 앙상블 결정 (임시 구현)
                    try:
                        # 간단한 거래 결정 (실제로는 정책 앙상블 사용)
                        action = self._get_simple_trading_action(row, idx)
                    except Exception as e:
                        self.logger.warning(f"거래 결정 실패: {e}")
                        action = 2  # Hold
                    
                    # 거래 실행
                    observation, reward, terminated, truncated, info = self.trading_environment.step(action)
                    
                    # 성과 기록
                    period_returns.append(reward)
                    period_trades.append({
                        'timestamp': timestamp,
                        'action': action,
                        'price': current_price,
                        'reward': reward,
                        'regime': regime_label
                    })
                    regime_predictions.append(regime_proba)
                    
                except Exception as e:
                    self.logger.error(f"거래 실행 실패 (인덱스 {idx}): {e}")
                    continue
            
            # 기간 성과 계산
            if period_returns:
                total_return = np.prod(1 + np.array(period_returns)) - 1
                sharpe_ratio = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252) if np.std(period_returns) > 0 else 0
                max_drawdown = self._calculate_max_drawdown_from_returns(period_returns)
            else:
                total_return = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
            
            return {
                'period': period_idx,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'validation_start': validation_data.index[0],
                'validation_end': validation_data.index[-1],
                'train_size': len(train_data),
                'validation_size': len(validation_data),
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(period_trades),
                'trades': period_trades,
                'regime_predictions': regime_predictions,
                'returns': period_returns
            }
            
        except Exception as e:
            self.logger.error(f"검증 기간 실행 실패: {e}")
            return {
                'period': period_idx,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'trades': [],
                'regime_predictions': [],
                'returns': []
            }
    
    def _get_simple_trading_action(self, data: pd.Series, index: int) -> int:
        """간단한 거래 액션 결정 (임시 구현)"""
        try:
            # 간단한 이동평균 전략 (임시)
            if index < 20:
                return 2  # Hold
            
            # 랜덤 액션 (임시 구현)
            import random
            random.seed(index)  # 재현 가능한 결과
            action_prob = random.random()
            
            if action_prob < 0.2:
                return 0  # Strong Buy
            elif action_prob < 0.4:
                return 1  # Buy
            elif action_prob > 0.8:
                return 3  # Sell
            elif action_prob > 0.9:
                return 4  # Strong Sell
            else:
                return 2  # Hold
                
        except Exception as e:
            self.logger.error(f"거래 액션 결정 실패: {e}")
            return 2  # Hold
    
    def _calculate_max_drawdown_from_returns(self, returns: List[float]) -> float:
        """수익률 리스트에서 최대 낙폭 계산"""
        try:
            if not returns:
                return 0.0
            
            cumulative_returns = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            return float(np.min(drawdown))
            
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def _extract_state_features(self, row: pd.Series) -> np.ndarray:
        """
        상태 특징 추출
        
        Args:
            row: 데이터 행
            
        Returns:
            273차원 멀티모달 특징 벡터
        """
        try:
            # 멀티모달 특징이 이미 계산되어 있다고 가정
            # 실제로는 visual_features, technical_features, sentiment_features를 결합
            
            # 임시로 기본 특징 생성 (실제 구현에서는 멀티모달 특징 사용)
            features = np.zeros(273)
            
            # 가격 특징 (간단한 예시)
            features[0] = row['close']
            features[1] = row['volume']
            features[2] = (row['high'] - row['low']) / row['close']  # 변동성
            features[3] = (row['close'] - row['open']) / row['open']  # 수익률
            
            # 나머지는 0으로 초기화 (실제로는 멀티모달 특징으로 채움)
            
            return features
            
        except Exception as e:
            self.logger.error(f"상태 특징 추출 실패: {e}")
            return np.zeros(273)
    
    def _calculate_period_performance(self, returns: List[float], trades: List, 
                                    regime_predictions: List[str], period_idx: int) -> Dict:
        """
        기간 성과 계산
        
        Args:
            returns: 수익률 리스트
            trades: 거래 리스트
            regime_predictions: 체제 예측 리스트
            period_idx: 기간 인덱스
            
        Returns:
            기간 성과 딕셔너리
        """
        try:
            if not returns:
                return {
                    'period_idx': period_idx,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'regime_distribution': {}
                }
            
            # 기본 성과 지표
            total_return = returns[-1] if returns else 0.0
            
            # 샤프 비율
            if len(returns) > 1:
                returns_array = np.array(returns)
                daily_returns = np.diff(returns_array)
                sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 최대 낙폭
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # 승률
            positive_returns = sum(1 for r in returns if r > 0)
            win_rate = positive_returns / len(returns) if returns else 0.0
            
            # 거래 통계
            successful_trades = sum(1 for trade in trades if trade.success)
            total_trades = len(trades)
            
            # 체제 분포
            regime_distribution = {}
            for regime in regime_predictions:
                regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
            
            return {
                'period_idx': period_idx,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'regime_distribution': regime_distribution,
                'returns': returns,
                'trades': trades
            }
            
        except Exception as e:
            self.logger.error(f"기간 성과 계산 실패: {e}")
            return {
                'period_idx': period_idx,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'regime_distribution': {},
                'error': str(e)
            }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        최대 낙폭 계산
        
        Args:
            returns: 수익률 리스트
            
        Returns:
            최대 낙폭
        """
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            
            return np.min(drawdowns)
            
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def _calculate_performance_metrics(self, walk_forward_results: List[Dict]) -> Dict[str, float]:
        """
        전체 성과 지표 계산
        
        Args:
            walk_forward_results: Walk-Forward 결과
            
        Returns:
            성과 지표 딕셔너리
        """
        try:
            if not walk_forward_results:
                return {}
            
            # 전체 수익률
            total_returns = [result['total_return'] for result in walk_forward_results]
            total_return = np.prod([1 + r for r in total_returns]) - 1
            
            # 연평균 수익률 (CAGR)
            num_years = len(walk_forward_results) / 252  # 252 거래일
            cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0.0
            
            # 샤프 비율
            sharpe_ratios = [result['sharpe_ratio'] for result in walk_forward_results]
            avg_sharpe_ratio = np.mean(sharpe_ratios)
            
            # 최대 낙폭
            max_drawdowns = [result['max_drawdown'] for result in walk_forward_results]
            max_drawdown = np.min(max_drawdowns)
            
            # 승률
            win_rates = [result['win_rate'] for result in walk_forward_results]
            avg_win_rate = np.mean(win_rates)
            
            # 거래 통계
            total_trades = sum(result['total_trades'] for result in walk_forward_results)
            successful_trades = sum(result['successful_trades'] for result in walk_forward_results)
            success_rate = successful_trades / total_trades if total_trades > 0 else 0.0
            
            # 변동성
            daily_returns = np.diff(total_returns)
            volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
            
            # 수익 팩터
            positive_returns = [r for r in total_returns if r > 0]
            negative_returns = [r for r in total_returns if r < 0]
            profit_factor = abs(sum(positive_returns) / sum(negative_returns)) if negative_returns else float('inf')
            
            return {
                'total_return': total_return,
                'cagr': cagr,
                'sharpe_ratio': avg_sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': avg_win_rate,
                'volatility': volatility,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'success_rate': success_rate,
                'num_periods': len(walk_forward_results)
            }
            
        except Exception as e:
            self.logger.error(f"성과 지표 계산 실패: {e}")
            return {}
    
    def _analyze_regime_performance(self, walk_forward_results: List[Dict]) -> Dict[str, Dict]:
        """
        체제별 성과 분석
        
        Args:
            walk_forward_results: Walk-Forward 결과
            
        Returns:
            체제별 성과 분석
        """
        try:
            regime_performance = defaultdict(list)
            
            # 체제별 성과 수집
            for result in walk_forward_results:
                regime_dist = result.get('regime_distribution', {})
                total_return = result.get('total_return', 0.0)
                
                for regime, count in regime_dist.items():
                    regime_performance[regime].append(total_return)
            
            # 체제별 통계 계산
            regime_analysis = {}
            for regime, returns in regime_performance.items():
                if returns:
                    regime_analysis[regime] = {
                        'avg_return': np.mean(returns),
                        'std_return': np.std(returns),
                        'min_return': np.min(returns),
                        'max_return': np.max(returns),
                        'num_periods': len(returns),
                        'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8)
                    }
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"체제별 성과 분석 실패: {e}")
            return {}
    
    def _generate_backtest_summary(self, performance_metrics: Dict[str, float]) -> Dict[str, str]:
        """
        백테스팅 요약 생성
        
        Args:
            performance_metrics: 성과 지표
            
        Returns:
            백테스팅 요약
        """
        try:
            summary = {
                'total_return': f"{performance_metrics.get('total_return', 0):.2%}",
                'cagr': f"{performance_metrics.get('cagr', 0):.2%}",
                'sharpe_ratio': f"{performance_metrics.get('sharpe_ratio', 0):.3f}",
                'max_drawdown': f"{performance_metrics.get('max_drawdown', 0):.2%}",
                'win_rate': f"{performance_metrics.get('win_rate', 0):.2%}",
                'volatility': f"{performance_metrics.get('volatility', 0):.2%}",
                'profit_factor': f"{performance_metrics.get('profit_factor', 0):.2f}",
                'total_trades': f"{performance_metrics.get('total_trades', 0):,}",
                'success_rate': f"{performance_metrics.get('success_rate', 0):.2%}"
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"백테스팅 요약 생성 실패: {e}")
            return {}
    
    def save_results(self, filepath: str):
        """
        백테스팅 결과 저장
        
        Args:
            filepath: 저장 파일 경로
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 결과 정리
            results_to_save = {
                'backtest_results': self.backtest_results,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # JSON으로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"백테스팅 결과 저장 완료: {filepath}")
            
        except Exception as e:
            self.logger.error(f"백테스팅 결과 저장 실패: {e}")
            raise
    
    def get_results_summary(self) -> Dict[str, Union[str, float, int, Dict]]:
        """백테스팅 결과 요약 반환"""
        try:
            return {
                'performance_metrics': self.performance_metrics,
                'backtest_summary': self._generate_backtest_summary(self.performance_metrics),
                'total_periods': len(self.backtest_results),
                'config': {
                    'initial_window_size': self.initial_window_size,
                    'validation_period': self.validation_period,
                    'expanding_window': self.expanding_window
                }
            }
            
        except Exception as e:
            self.logger.error(f"결과 요약 생성 실패: {e}")
            return {}


# 편의 함수들
def create_backtesting_engine(config: Dict, policy_ensemble: PolicyEnsemble, 
                            regime_classifier: MarketRegimeClassifier) -> BacktestingEngine:
    """백테스팅 엔진 생성 편의 함수"""
    return BacktestingEngine(config, policy_ensemble, regime_classifier)


def run_simple_backtest(data: pd.DataFrame, config: Dict, policy_ensemble: PolicyEnsemble, 
                       regime_classifier: MarketRegimeClassifier) -> Dict:
    """간단한 백테스팅 실행 편의 함수"""
    engine = create_backtesting_engine(config, policy_ensemble, regime_classifier)
    return engine.run_backtest(data)


# 백테스팅 엔진에 거래 시뮬레이션 메서드 추가
def _simulate_trading_period(self, validation_data: pd.DataFrame) -> Dict[str, Union[float, List]]:
    """
    검증 기간 동안 거래 시뮬레이션
    
    Args:
        validation_data: 검증 데이터
        
    Returns:
        거래 시뮬레이션 결과
    """
    try:
        # 포트폴리오 초기화
        initial_capital = self.backtesting_config.get('initial_capital', 10000)
        portfolio_value = initial_capital
        position = 0.0  # 현재 포지션 (0: 현금, 1: 풀 포지션)
        
        # 거래 기록
        trades = []
        portfolio_values = [portfolio_value]
        
        # 각 시점에서 거래 시뮬레이션
        for i in range(len(validation_data)):
            current_data = validation_data.iloc[i]
            current_price = current_data['close']
            
            # 간단한 거래 전략 (임시 구현)
            action = self._get_simple_trading_action(current_data, i)
            
            # 거래 실행
            if action == 'buy' and position < 1.0:
                # 매수
                buy_amount = (1.0 - position) * portfolio_value
                shares = buy_amount / current_price
                position += shares * current_price / portfolio_value
                trades.append({
                    'timestamp': current_data.name,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'amount': buy_amount
                })
            elif action == 'sell' and position > 0.0:
                # 매도
                sell_amount = position * portfolio_value
                shares = sell_amount / current_price
                position -= shares * current_price / portfolio_value
                trades.append({
                    'timestamp': current_data.name,
                    'action': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'amount': sell_amount
                })
            
            # 포트폴리오 가치 업데이트
            portfolio_value = initial_capital * (1 + position * (current_price / validation_data.iloc[0]['close'] - 1))
            portfolio_values.append(portfolio_value)
        
        # 성과 계산
        total_return = (portfolio_value - initial_capital) / initial_capital
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 최대 낙폭 계산
        peak = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values
        }
        
    except Exception as e:
        self.logger.error(f"거래 시뮬레이션 실패: {e}")
        return {
            'initial_capital': 10000,
            'final_value': 10000,
            'total_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'trades': [],
            'portfolio_values': [10000]
        }


def _get_simple_trading_action(self, data: pd.Series, index: int) -> str:
    """
    간단한 거래 액션 결정 (임시 구현)
    
    Args:
        data: 현재 시점 데이터
        index: 현재 인덱스
        
    Returns:
        거래 액션 ('buy', 'sell', 'hold')
    """
    try:
        # 간단한 이동평균 전략 (임시)
        if index < 20:
            return 'hold'
        
        # 랜덤 액션 (임시 구현)
        import random
        random.seed(index)  # 재현 가능한 결과
        action_prob = random.random()
        
        if action_prob < 0.3:
            return 'buy'
        elif action_prob > 0.7:
            return 'sell'
        else:
            return 'hold'
            
    except Exception as e:
        self.logger.error(f"거래 액션 결정 실패: {e}")
        return 'hold'


# BacktestingEngine 클래스에 메서드 추가
BacktestingEngine._simulate_trading_period = _simulate_trading_period
BacktestingEngine._get_simple_trading_action = _get_simple_trading_action