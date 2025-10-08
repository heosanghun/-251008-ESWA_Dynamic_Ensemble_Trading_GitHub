"""
성과 모니터링 시스템 모듈
Performance Monitoring System Module

실시간 성과 모니터링 및 이상 상황 감지
- 실시간 성과 모니터링
- 이상 상황 감지 및 알림
- 성과 추세 분석 및 예측
- 체제 전환 감지 및 대응
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta
import threading
import queue
from collections import deque
import json


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceTrend(Enum):
    """성과 추세"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


@dataclass
class PerformanceAlert:
    """성과 알림"""
    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    value: float
    threshold: float
    recommendation: str


@dataclass
class PerformanceSnapshot:
    """성과 스냅샷"""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    regime: str
    regime_confidence: float
    num_trades: int
    win_rate: float


@dataclass
class TrendAnalysis:
    """추세 분석"""
    trend: PerformanceTrend
    trend_strength: float
    change_rate: float
    confidence: float
    prediction: Dict[str, float]


class PerformanceMonitor:
    """
    성과 모니터링 시스템 메인 클래스
    
    실시간 성과 모니터링 및 이상 상황 감지
    """
    
    def __init__(self, config: Dict):
        """
        성과 모니터링 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 모니터링 설정
        self.monitoring_config = config.get('performance_monitoring', {})
        
        # 모니터링 주기
        self.monitoring_interval = self.monitoring_config.get('interval', 60)  # 초
        self.history_length = self.monitoring_config.get('history_length', 1000)
        
        # 알림 설정
        self.alert_config = self.monitoring_config.get('alerts', {})
        self.alert_thresholds = self.alert_config.get('thresholds', {})
        
        # 추세 분석 설정
        self.trend_config = self.monitoring_config.get('trend_analysis', {})
        self.trend_window = self.trend_config.get('window', 30)
        self.prediction_horizon = self.trend_config.get('prediction_horizon', 7)
        
        # 체제 전환 감지 설정
        self.regime_config = self.monitoring_config.get('regime_detection', {})
        self.regime_change_threshold = self.regime_config.get('change_threshold', 0.3)
        
        # 데이터 저장소
        self.performance_history = deque(maxlen=self.history_length)
        self.alert_history = deque(maxlen=self.history_length)
        self.trend_history = deque(maxlen=self.history_length)
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_queue = queue.Queue()
        
        # 콜백 함수들
        self.alert_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("성과 모니터링 시스템 초기화 완료")
        self.logger.info(f"모니터링 주기: {self.monitoring_interval}초")
        self.logger.info(f"히스토리 길이: {self.history_length}")
        self.logger.info(f"추세 분석 윈도우: {self.trend_window}")
        self.logger.info(f"체제 전환 임계값: {self.regime_change_threshold}")
    
    def start_monitoring(self, data_source: Callable) -> None:
        """
        실시간 모니터링 시작
        
        Args:
            data_source: 성과 데이터 소스 함수
        """
        try:
            if self.is_monitoring:
                self.logger.warning("모니터링이 이미 실행 중입니다.")
                return
            
            self.is_monitoring = True
            self.data_source = data_source
            
            # 모니터링 스레드 시작
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.logger.info("실시간 성과 모니터링 시작")
            
        except Exception as e:
            self.logger.error(f"모니터링 시작 실패: {e}")
            raise
    
    def stop_monitoring(self) -> None:
        """실시간 모니터링 중지"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("실시간 성과 모니터링 중지")
            
        except Exception as e:
            self.logger.error(f"모니터링 중지 실패: {e}")
    
    def _monitoring_loop(self) -> None:
        """모니터링 루프"""
        try:
            while self.is_monitoring:
                # 성과 데이터 수집
                performance_data = self.data_source()
                
                if performance_data:
                    # 성과 스냅샷 생성
                    snapshot = self._create_performance_snapshot(performance_data)
                    
                    # 성과 히스토리에 추가
                    self.performance_history.append(snapshot)
                    
                    # 이상 상황 감지
                    alerts = self._detect_anomalies(snapshot)
                    
                    # 알림 처리
                    for alert in alerts:
                        self._process_alert(alert)
                    
                    # 추세 분석
                    trend_analysis = self._analyze_trends()
                    if trend_analysis:
                        self.trend_history.append(trend_analysis)
                    
                    # 체제 전환 감지
                    regime_change = self._detect_regime_change(snapshot)
                    if regime_change:
                        self._process_regime_change(regime_change)
                    
                    # 콜백 함수 실행
                    self._execute_callbacks(snapshot, alerts, trend_analysis)
                
                # 다음 모니터링까지 대기
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            self.logger.error(f"모니터링 루프 오류: {e}")
            self.is_monitoring = False
    
    def _create_performance_snapshot(self, data: Dict) -> PerformanceSnapshot:
        """성과 스냅샷 생성"""
        try:
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                portfolio_value=data.get('portfolio_value', 0.0),
                total_return=data.get('total_return', 0.0),
                daily_return=data.get('daily_return', 0.0),
                sharpe_ratio=data.get('sharpe_ratio', 0.0),
                max_drawdown=data.get('max_drawdown', 0.0),
                volatility=data.get('volatility', 0.0),
                regime=data.get('regime', 'unknown'),
                regime_confidence=data.get('regime_confidence', 0.0),
                num_trades=data.get('num_trades', 0),
                win_rate=data.get('win_rate', 0.0)
            )
        except Exception as e:
            self.logger.error(f"성과 스냅샷 생성 실패: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                portfolio_value=0.0,
                total_return=0.0,
                daily_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                regime='unknown',
                regime_confidence=0.0,
                num_trades=0,
                win_rate=0.0
            )
    
    def _detect_anomalies(self, snapshot: PerformanceSnapshot) -> List[PerformanceAlert]:
        """이상 상황 감지"""
        try:
            alerts = []
            
            # 최대 낙폭 알림
            max_drawdown_threshold = self.alert_thresholds.get('max_drawdown', 0.1)
            if abs(snapshot.max_drawdown) > max_drawdown_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=snapshot.timestamp,
                    level=AlertLevel.WARNING,
                    category='max_drawdown',
                    message=f"최대 낙폭이 임계값을 초과했습니다: {snapshot.max_drawdown:.2%}",
                    value=snapshot.max_drawdown,
                    threshold=max_drawdown_threshold,
                    recommendation="포지션 크기를 줄이고 리스크를 모니터링하세요."
                ))
            
            # 변동성 알림
            volatility_threshold = self.alert_thresholds.get('volatility', 0.3)
            if snapshot.volatility > volatility_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=snapshot.timestamp,
                    level=AlertLevel.WARNING,
                    category='volatility',
                    message=f"변동성이 임계값을 초과했습니다: {snapshot.volatility:.2%}",
                    value=snapshot.volatility,
                    threshold=volatility_threshold,
                    recommendation="포트폴리오 다각화를 고려하세요."
                ))
            
            # 샤프 비율 알림
            sharpe_threshold = self.alert_thresholds.get('sharpe_ratio', 0.5)
            if snapshot.sharpe_ratio < sharpe_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=snapshot.timestamp,
                    level=AlertLevel.INFO,
                    category='sharpe_ratio',
                    message=f"샤프 비율이 낮습니다: {snapshot.sharpe_ratio:.2f}",
                    value=snapshot.sharpe_ratio,
                    threshold=sharpe_threshold,
                    recommendation="투자 전략을 재검토하세요."
                ))
            
            # 일일 수익률 알림
            daily_return_threshold = self.alert_thresholds.get('daily_return', 0.05)
            if abs(snapshot.daily_return) > daily_return_threshold:
                level = AlertLevel.ERROR if abs(snapshot.daily_return) > daily_return_threshold * 2 else AlertLevel.WARNING
                alerts.append(PerformanceAlert(
                    timestamp=snapshot.timestamp,
                    level=level,
                    category='daily_return',
                    message=f"일일 수익률이 크게 변동했습니다: {snapshot.daily_return:.2%}",
                    value=snapshot.daily_return,
                    threshold=daily_return_threshold,
                    recommendation="시장 상황을 주의 깊게 모니터링하세요."
                ))
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"이상 상황 감지 실패: {e}")
            return []
    
    def _analyze_trends(self) -> Optional[TrendAnalysis]:
        """추세 분석"""
        try:
            if len(self.performance_history) < self.trend_window:
                return None
            
            # 최근 성과 데이터 추출
            recent_snapshots = list(self.performance_history)[-self.trend_window:]
            
            # 수익률 추세 분석
            returns = [s.daily_return for s in recent_snapshots]
            return_trend = self._calculate_trend(returns)
            
            # 샤프 비율 추세 분석
            sharpe_ratios = [s.sharpe_ratio for s in recent_snapshots]
            sharpe_trend = self._calculate_trend(sharpe_ratios)
            
            # 변동성 추세 분석
            volatilities = [s.volatility for s in recent_snapshots]
            volatility_trend = self._calculate_trend(volatilities)
            
            # 전체 추세 결정
            overall_trend = self._determine_overall_trend(return_trend, sharpe_trend, volatility_trend)
            
            # 추세 강도 계산
            trend_strength = self._calculate_trend_strength(returns, sharpe_ratios, volatilities)
            
            # 변화율 계산
            change_rate = self._calculate_change_rate(recent_snapshots)
            
            # 신뢰도 계산
            confidence = self._calculate_trend_confidence(returns, sharpe_ratios, volatilities)
            
            # 예측 생성
            prediction = self._generate_prediction(recent_snapshots)
            
            return TrendAnalysis(
                trend=overall_trend,
                trend_strength=trend_strength,
                change_rate=change_rate,
                confidence=confidence,
                prediction=prediction
            )
            
        except Exception as e:
            self.logger.error(f"추세 분석 실패: {e}")
            return None
    
    def _detect_regime_change(self, snapshot: PerformanceSnapshot) -> Optional[Dict]:
        """체제 전환 감지"""
        try:
            if len(self.performance_history) < 2:
                return None
            
            # 이전 체제와 비교
            previous_snapshot = self.performance_history[-1]
            
            # 체제 변경 감지
            if snapshot.regime != previous_snapshot.regime:
                confidence_change = abs(snapshot.regime_confidence - previous_snapshot.regime_confidence)
                
                if confidence_change > self.regime_change_threshold:
                    return {
                        'timestamp': snapshot.timestamp,
                        'previous_regime': previous_snapshot.regime,
                        'current_regime': snapshot.regime,
                        'confidence_change': confidence_change,
                        'previous_confidence': previous_snapshot.regime_confidence,
                        'current_confidence': snapshot.regime_confidence
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"체제 전환 감지 실패: {e}")
            return None
    
    def _process_alert(self, alert: PerformanceAlert) -> None:
        """알림 처리"""
        try:
            # 알림 히스토리에 추가
            self.alert_history.append(alert)
            
            # 로깅
            self.logger.log(
                getattr(logging, alert.level.value.upper()),
                f"[{alert.category}] {alert.message}"
            )
            
            # 알림 큐에 추가
            self.alert_queue.put(alert)
            
        except Exception as e:
            self.logger.error(f"알림 처리 실패: {e}")
    
    def _process_regime_change(self, regime_change: Dict) -> None:
        """체제 전환 처리"""
        try:
            # 체제 전환 알림 생성
            alert = PerformanceAlert(
                timestamp=regime_change['timestamp'],
                level=AlertLevel.INFO,
                category='regime_change',
                message=f"시장 체제가 변경되었습니다: {regime_change['previous_regime']} → {regime_change['current_regime']}",
                value=regime_change['confidence_change'],
                threshold=self.regime_change_threshold,
                recommendation="새로운 체제에 맞는 전략으로 조정하세요."
            )
            
            self._process_alert(alert)
            
        except Exception as e:
            self.logger.error(f"체제 전환 처리 실패: {e}")
    
    def _execute_callbacks(self, snapshot: PerformanceSnapshot, 
                          alerts: List[PerformanceAlert], 
                          trend_analysis: Optional[TrendAnalysis]) -> None:
        """콜백 함수 실행"""
        try:
            # 성과 콜백 실행
            for callback in self.performance_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    self.logger.error(f"성과 콜백 실행 실패: {e}")
            
            # 알림 콜백 실행
            for alert in alerts:
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"알림 콜백 실행 실패: {e}")
            
        except Exception as e:
            self.logger.error(f"콜백 실행 실패: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """추세 계산 (선형 회귀 기울기)"""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # 선형 회귀
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
            
        except Exception as e:
            self.logger.error(f"추세 계산 실패: {e}")
            return 0.0
    
    def _determine_overall_trend(self, return_trend: float, 
                               sharpe_trend: float, 
                               volatility_trend: float) -> PerformanceTrend:
        """전체 추세 결정"""
        try:
            # 추세 점수 계산
            trend_score = 0.0
            
            if return_trend > 0:
                trend_score += 1.0
            elif return_trend < 0:
                trend_score -= 1.0
            
            if sharpe_trend > 0:
                trend_score += 1.0
            elif sharpe_trend < 0:
                trend_score -= 1.0
            
            if volatility_trend < 0:  # 변동성 감소는 좋음
                trend_score += 0.5
            elif volatility_trend > 0:
                trend_score -= 0.5
            
            # 추세 결정
            if trend_score > 1.0:
                return PerformanceTrend.IMPROVING
            elif trend_score < -1.0:
                return PerformanceTrend.DECLINING
            elif abs(trend_score) < 0.5:
                return PerformanceTrend.STABLE
            else:
                return PerformanceTrend.VOLATILE
                
        except Exception as e:
            self.logger.error(f"전체 추세 결정 실패: {e}")
            return PerformanceTrend.STABLE
    
    def _calculate_trend_strength(self, returns: List[float], 
                                sharpe_ratios: List[float], 
                                volatilities: List[float]) -> float:
        """추세 강도 계산"""
        try:
            # 각 지표의 추세 강도 계산
            return_strength = abs(self._calculate_trend(returns))
            sharpe_strength = abs(self._calculate_trend(sharpe_ratios))
            volatility_strength = abs(self._calculate_trend(volatilities))
            
            # 평균 추세 강도
            avg_strength = (return_strength + sharpe_strength + volatility_strength) / 3
            
            return float(avg_strength)
            
        except Exception as e:
            self.logger.error(f"추세 강도 계산 실패: {e}")
            return 0.0
    
    def _calculate_change_rate(self, snapshots: List[PerformanceSnapshot]) -> float:
        """변화율 계산"""
        try:
            if len(snapshots) < 2:
                return 0.0
            
            first_value = snapshots[0].portfolio_value
            last_value = snapshots[-1].portfolio_value
            
            if first_value == 0:
                return 0.0
            
            change_rate = (last_value - first_value) / first_value
            return float(change_rate)
            
        except Exception as e:
            self.logger.error(f"변화율 계산 실패: {e}")
            return 0.0
    
    def _calculate_trend_confidence(self, returns: List[float], 
                                  sharpe_ratios: List[float], 
                                  volatilities: List[float]) -> float:
        """추세 신뢰도 계산"""
        try:
            # 각 지표의 일관성 계산
            return_consistency = 1.0 - (np.std(returns) / (np.mean(np.abs(returns)) + 1e-8))
            sharpe_consistency = 1.0 - (np.std(sharpe_ratios) / (np.mean(np.abs(sharpe_ratios)) + 1e-8))
            volatility_consistency = 1.0 - (np.std(volatilities) / (np.mean(volatilities) + 1e-8))
            
            # 평균 신뢰도
            avg_confidence = (return_consistency + sharpe_consistency + volatility_consistency) / 3
            
            return float(max(0.0, min(1.0, avg_confidence)))
            
        except Exception as e:
            self.logger.error(f"추세 신뢰도 계산 실패: {e}")
            return 0.5
    
    def _generate_prediction(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, float]:
        """예측 생성"""
        try:
            # 간단한 선형 예측
            if len(snapshots) < 2:
                return {'portfolio_value': 0.0, 'total_return': 0.0}
            
            # 포트폴리오 가치 추세
            values = [s.portfolio_value for s in snapshots]
            value_trend = self._calculate_trend(values)
            
            # 수익률 추세
            returns = [s.daily_return for s in snapshots]
            return_trend = self._calculate_trend(returns)
            
            # 예측값 계산
            current_value = values[-1]
            predicted_value = current_value * (1 + value_trend * self.prediction_horizon)
            
            current_return = returns[-1]
            predicted_return = current_return + return_trend * self.prediction_horizon
            
            return {
                'portfolio_value': float(predicted_value),
                'total_return': float(predicted_return)
            }
            
        except Exception as e:
            self.logger.error(f"예측 생성 실패: {e}")
            return {'portfolio_value': 0.0, 'total_return': 0.0}
    
    def add_alert_callback(self, callback: Callable) -> None:
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable) -> None:
        """성과 콜백 추가"""
        self.performance_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약 반환"""
        try:
            if not self.performance_history:
                return {}
            
            latest_snapshot = self.performance_history[-1]
            recent_alerts = list(self.alert_history)[-10:]  # 최근 10개 알림
            
            return {
                'latest_performance': latest_snapshot,
                'recent_alerts': recent_alerts,
                'monitoring_status': self.is_monitoring,
                'total_snapshots': len(self.performance_history),
                'total_alerts': len(self.alert_history)
            }
            
        except Exception as e:
            self.logger.error(f"성과 요약 생성 실패: {e}")
            return {}
    
    def export_data(self, filepath: str) -> None:
        """데이터 내보내기"""
        try:
            data = {
                'performance_history': [
                    {
                        'timestamp': s.timestamp.isoformat(),
                        'portfolio_value': s.portfolio_value,
                        'total_return': s.total_return,
                        'daily_return': s.daily_return,
                        'sharpe_ratio': s.sharpe_ratio,
                        'max_drawdown': s.max_drawdown,
                        'volatility': s.volatility,
                        'regime': s.regime,
                        'regime_confidence': s.regime_confidence,
                        'num_trades': s.num_trades,
                        'win_rate': s.win_rate
                    }
                    for s in self.performance_history
                ],
                'alert_history': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'level': a.level.value,
                        'category': a.category,
                        'message': a.message,
                        'value': a.value,
                        'threshold': a.threshold,
                        'recommendation': a.recommendation
                    }
                    for a in self.alert_history
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            self.logger.error(f"데이터 내보내기 실패: {e}")


# 편의 함수들
def create_performance_monitor(config: Dict) -> PerformanceMonitor:
    """성과 모니터링 시스템 생성 편의 함수"""
    return PerformanceMonitor(config)


def start_performance_monitoring(data_source: Callable, config: Dict) -> PerformanceMonitor:
    """성과 모니터링 시작 편의 함수"""
    monitor = create_performance_monitor(config)
    monitor.start_monitoring(data_source)
    return monitor
