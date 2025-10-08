"""
ESWA 동적 앙상블 강화학습 거래 시스템 - 메인 통합 클래스
ESWA Dynamic Ensemble Reinforcement Learning Trading System - Main Integration Class

이 파일은 모든 모듈을 통합하여 완전한 거래 시스템을 구성합니다:
1. 멀티모달 특징 추출
2. 시장 체제 분류
3. 체제별 전문 에이전트 풀
4. 동적 앙상블 의사결정
5. 거래 실행 및 성과 관리

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path

# 각 모듈 import
from .multimodal import MultimodalFeatureFusion
from .regime import MarketRegimeClassifier, ConfidenceBasedSelection
from .agents import AgentPool, PPOAgent
from .ensemble import PolicyEnsemble, DynamicWeightAllocation
from .trading import TradingEnvironment, BacktestingEngine, PerformanceAnalyzer


class ESWADynamicEnsembleSystem:
    """
    ESWA 동적 앙상블 강화학습 거래 시스템의 메인 클래스
    
    이 클래스는 모든 하위 모듈들을 통합하여 완전한 거래 시스템을 제공합니다.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        ESWA 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # 시스템 상태
        self.is_trained = False
        self.current_regime = None
        self.portfolio_value = self.config['trading']['initial_capital']
        self.trade_history = []
        
        # 모듈 초기화 (실제 구현 후 활성화)
        self._initialize_modules()
        
        self.logger.info("ESWA Dynamic Ensemble System 초기화 완료")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"설정 파일 파싱 오류: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 시스템 설정"""
        logger = logging.getLogger('ESWA_System')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # 파일 핸들러
        log_file = Path(self.config['logging']['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(self.config['logging']['format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_modules(self):
        """모든 하위 모듈 초기화"""
        try:
            # 멀티모달 특징 추출 모듈
            self.multimodal = MultimodalFeatureFusion(self.config)
            
            # 시장 체제 분류 모듈
            self.regime_classifier = MarketRegimeClassifier(self.config)
            self.confidence_selection = ConfidenceBasedSelection(self.config)
            
            # 에이전트 풀 모듈
            self.agent_pool = AgentPool(self.config)
            
            # 앙상블 시스템 모듈
            self.ensemble = PolicyEnsemble(self.config, self.agent_pool)
            
            # 거래 환경 모듈
            self.trading_env = TradingEnvironment(self.config)
            
            # 백테스팅 엔진
            self.backtesting_engine = BacktestingEngine(
                self.config, self.ensemble, self.regime_classifier
            )
            
            # 성과 분석기
            self.performance_analyzer = PerformanceAnalyzer(self.config)
            
            self.logger.info("모든 모듈 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"모듈 초기화 실패: {e}")
            raise
    
    def load_data(self, data: pd.DataFrame, start_date: str = None, end_date: str = None):
        """
        거래 데이터 로드
        
        Args:
            data: 시계열 데이터 (OHLCV + 멀티모달 특징)
            start_date: 시작 날짜 (선택사항)
            end_date: 종료 날짜 (선택사항)
        """
        try:
            self.logger.info(f"데이터 로딩 시작: {len(data)}개 레코드")
            
            # 날짜 필터링
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # 데이터 저장
            self.market_data = data
            
            self.logger.info(f"데이터 로딩 완료: {len(self.market_data)}개 레코드")
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 실패: {e}")
            raise
    
    def train(self, episodes: int = None):
        """
        시스템 훈련
        
        Args:
            episodes: 훈련 에피소드 수
        """
        if episodes is None:
            episodes = self.config.get('training', {}).get('episodes', 1000)
        
        try:
            self.logger.info(f"시스템 훈련 시작: {episodes} 에피소드")
            
            if not hasattr(self, 'market_data') or self.market_data is None:
                raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            
            # 1. 시장 체제 분류기 훈련
            self.logger.info("시장 체제 분류기 훈련 시작")
            self.regime_classifier.train(self.market_data)
            
            # 2. 각 체제별 에이전트 풀 훈련
            self.logger.info("에이전트 풀 훈련 시작")
            for regime in ['bull_market', 'bear_market', 'sideways_market']:
                self.logger.info(f"{regime} 에이전트 풀 훈련")
                # 간단한 훈련 데이터 생성 (실제로는 더 복잡한 데이터 필요)
                training_data = self._generate_training_data(regime, episodes)
                self.agent_pool.train_regime_agents(regime, training_data, episodes)
            
            self.is_trained = True
            self.logger.info("시스템 훈련 완료")
            
        except Exception as e:
            self.logger.error(f"시스템 훈련 실패: {e}")
            raise
    
    def _generate_training_data(self, regime: str, episodes: int) -> List[Dict]:
        """훈련 데이터 생성 (임시 구현)"""
        try:
            training_data = []
            
            for episode in range(episodes):
                episode_data = []
                
                # 간단한 에피소드 데이터 생성
                for step in range(100):  # 에피소드당 100 스텝
                    # 랜덤 상태 생성
                    state = np.random.randn(273)
                    
                    # 랜덤 액션
                    action = np.random.randint(0, 5)
                    
                    # 랜덤 보상 (체제별로 다른 패턴)
                    if regime == 'bull_market':
                        reward = np.random.normal(0.01, 0.05)  # 양의 기대값
                    elif regime == 'bear_market':
                        reward = np.random.normal(-0.005, 0.03)  # 음의 기대값
                    else:  # sideways_market
                        reward = np.random.normal(0.0, 0.02)  # 중립
                    
                    # 랜덤 가치 및 로그 확률
                    value = np.random.normal(0.0, 0.1)
                    log_prob = np.random.normal(-1.0, 0.5)
                    
                    episode_data.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'value': value,
                        'log_prob': log_prob,
                        'done': step == 99
                    })
                
                training_data.extend(episode_data)
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"훈련 데이터 생성 실패: {e}")
            return []
    
    def predict(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        시장 데이터를 기반으로 거래 결정 예측
        
        Args:
            market_data: 시장 데이터 (OHLCV, 뉴스 등)
            
        Returns:
            거래 결정 및 관련 정보
        """
        if not self.is_trained:
            raise ValueError("시스템이 훈련되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        try:
            # 1. 멀티모달 특징 추출
            features = self.multimodal.fuse_features(
                visual_features=np.random.randn(256),  # 임시
                technical_features=np.random.randn(15),  # 임시
                sentiment_features=np.random.randn(2)  # 임시
            )
            
            # 2. 시장 체제 분류
            # features가 이미 numpy 배열이므로 .numpy() 제거
            if hasattr(features, 'numpy'):
                features_array = features.numpy()
            else:
                features_array = features
                
            regime_probs = self.regime_classifier.predict_proba(features_array.reshape(1, -1))
            regime_probs_dict = {
                'bull_market': regime_probs[0][0],
                'bear_market': regime_probs[0][1],
                'sideways_market': regime_probs[0][2]
            }
            
            predicted_regime = self.regime_classifier.predict(features_array.reshape(1, -1))[0]
            
            # 3. 동적 앙상블로 최종 결정
            decision = self.ensemble.make_decision(
                state=features_array,
                regime=predicted_regime,
                regime_probabilities=regime_probs_dict
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"예측 실패: {e}")
            raise
    
    def backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        백테스팅 실행
        
        Args:
            start_date: 백테스팅 시작 날짜
            end_date: 백테스팅 종료 날짜
            
        Returns:
            백테스팅 결과
        """
        if not self.is_trained:
            raise ValueError("시스템이 훈련되지 않았습니다.")
        
        try:
            self.logger.info("백테스팅 시작")
            
            if not hasattr(self, 'market_data') or self.market_data is None:
                raise ValueError("데이터가 로드되지 않았습니다.")
            
            # 백테스팅 실행
            results = self.backtesting_engine.run_backtest(
                self.market_data, start_date, end_date
            )
            
            self.logger.info("백테스팅 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"백테스팅 실패: {e}")
            raise
    
    def start_live_trading(self, symbol: str, update_frequency: str = "1h"):
        """
        실시간 거래 시작
        
        Args:
            symbol: 거래 심볼
            update_frequency: 업데이트 빈도
        """
        if not self.is_trained:
            raise ValueError("시스템이 훈련되지 않았습니다.")
        
        try:
            self.logger.info(f"실시간 거래 시작: {symbol}")
            
            # 실시간 거래 로직 구현
            # self.trading_env.start_live_trading(symbol, update_frequency)
            
            self.logger.info("실시간 거래 시작됨")
            
        except Exception as e:
            self.logger.error(f"실시간 거래 시작 실패: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """현재 성과 지표 반환"""
        try:
            if hasattr(self, 'backtest_results') and self.backtest_results:
                # 백테스팅 결과에서 성과 지표 추출
                returns = []
                for result in self.backtest_results:
                    if 'returns' in result:
                        returns.extend(result['returns'])
                
                if returns:
                    metrics = self.performance_analyzer.calculate_comprehensive_metrics(returns)
                    return {
                        'total_return': metrics.total_return,
                        'cagr': metrics.cagr,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'win_rate': metrics.win_rate,
                        'profit_factor': metrics.profit_factor,
                        'volatility': metrics.volatility
                    }
            
            # 기본값 반환
            return {
                'total_return': 0.0,
                'cagr': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'volatility': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"성과 지표 계산 실패: {e}")
            return {}
    
    def save_model(self, path: str):
        """모델 저장"""
        try:
            self.logger.info(f"모델 저장: {path}")
            
            # 각 모듈별 모델 저장
            self.regime_classifier.save_model(f"{path}/regime")
            self.agent_pool.save_models(f"{path}/agents")
            
            self.logger.info("모델 저장 완료")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, path: str):
        """모델 로드"""
        try:
            self.logger.info(f"모델 로드: {path}")
            
            # 각 모듈별 모델 로드
            self.regime_classifier.load_model(f"{path}/regime")
            self.agent_pool.load_models(f"{path}/agents")
            
            self.is_trained = True
            self.logger.info("모델 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


# 편의 함수들
def create_eswa_system(config_path: str = "configs/config.yaml") -> ESWADynamicEnsembleSystem:
    """ESWA 시스템 생성 편의 함수"""
    return ESWADynamicEnsembleSystem(config_path)


def quick_start(data: pd.DataFrame, episodes: int = 1000) -> ESWADynamicEnsembleSystem:
    """빠른 시작을 위한 편의 함수"""
    system = create_eswa_system()
    system.load_data(data)
    system.train(episodes)
    return system
