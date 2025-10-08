"""
성과 시각화 및 보고서 생성 모듈
Performance Visualization and Report Generation Module

성과 지표 시각화 및 보고서 생성
- 성과 차트 생성 (수익률, 드로우다운, 샤프 비율 등)
- 체제별 성과 비교 차트
- 거래 히스토리 분석 차트
- HTML 보고서 생성

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import json


class PerformanceVisualizer:
    """
    성과 시각화 클래스
    
    다양한 성과 지표를 시각화하고 보고서를 생성
    """
    
    def __init__(self, config: Dict):
        """
        성과 시각화기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 시각화 설정
        self.viz_config = config.get('visualization', {})
        self.figure_size = self.viz_config.get('figure_size', (12, 8))
        self.dpi = self.viz_config.get('dpi', 300)
        self.style = self.viz_config.get('style', 'seaborn-v0_8')
        
        # 색상 팔레트
        self.colors = {
            'bull_market': '#2E8B57',  # Sea Green
            'bear_market': '#DC143C',  # Crimson
            'sideways_market': '#4682B4',  # Steel Blue
            'portfolio': '#FF6347',  # Tomato
            'benchmark': '#696969'  # Dim Gray
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 스타일 설정
        plt.style.use(self.style)
        sns.set_palette("husl")
        
        self.logger.info("성과 시각화기 초기화 완료")
    
    def plot_cumulative_returns(self, returns: List[float], 
                              benchmark_returns: Optional[List[float]] = None,
                              title: str = "누적 수익률") -> plt.Figure:
        """
        누적 수익률 차트 생성
        
        Args:
            returns: 수익률 리스트
            benchmark_returns: 벤치마크 수익률 (선택사항)
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # 누적 수익률 계산
            cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
            
            # 날짜 인덱스 생성 (임시)
            dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
            
            # 포트폴리오 수익률 플롯
            ax.plot(dates, cumulative_returns, 
                   label='ESWA Portfolio', color=self.colors['portfolio'], linewidth=2)
            
            # 벤치마크 수익률 플롯 (있는 경우)
            if benchmark_returns is not None:
                benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns)) - 1
                ax.plot(dates, benchmark_cumulative, 
                       label='Benchmark', color=self.colors['benchmark'], linewidth=2, linestyle='--')
            
            # 차트 설정
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Returns', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Y축을 퍼센트로 표시
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # 날짜 포맷팅
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            self.logger.info("누적 수익률 차트 생성 완료")
            return fig
            
        except Exception as e:
            self.logger.error(f"누적 수익률 차트 생성 실패: {e}")
            raise
    
    def plot_drawdown(self, returns: List[float], 
                     title: str = "드로우다운 분석") -> plt.Figure:
        """
        드로우다운 차트 생성
        
        Args:
            returns: 수익률 리스트
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_size[0], self.figure_size[1]*1.5))
            
            returns_array = np.array(returns)
            
            # 누적 수익률
            cumulative_returns = np.cumprod(1 + returns_array)
            dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
            
            ax1.plot(dates, cumulative_returns, color=self.colors['portfolio'], linewidth=2)
            ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative Returns', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
            
            # 드로우다운
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            
            ax2.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
            ax2.plot(dates, drawdowns, color='red', linewidth=1)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Drawdown', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # 최대 드로우다운 표시
            max_dd_idx = np.argmin(drawdowns)
            max_dd = drawdowns[max_dd_idx]
            ax2.axhline(y=max_dd, color='red', linestyle='--', alpha=0.7)
            ax2.text(dates[max_dd_idx], max_dd, f'Max DD: {max_dd:.1%}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 날짜 포맷팅
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            self.logger.info("드로우다운 차트 생성 완료")
            return fig
            
        except Exception as e:
            self.logger.error(f"드로우다운 차트 생성 실패: {e}")
            raise
    
    def plot_regime_performance(self, regime_returns: Dict[str, List[float]], 
                              title: str = "체제별 성과 비교") -> plt.Figure:
        """
        체제별 성과 비교 차트 생성
        
        Args:
            regime_returns: 체제별 수익률 딕셔너리
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figure_size[0]*1.5, self.figure_size[1]))
            
            # 1. 체제별 누적 수익률
            for regime, returns in regime_returns.items():
                if returns:
                    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
                    dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
                    ax1.plot(dates, cumulative_returns, 
                            label=regime.replace('_', ' ').title(), 
                            color=self.colors.get(regime, '#000000'), linewidth=2)
            
            ax1.set_title('Cumulative Returns by Regime', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative Returns', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # 2. 체제별 수익률 분포
            regime_data = []
            regime_labels = []
            for regime, returns in regime_returns.items():
                if returns:
                    regime_data.append(returns)
                    regime_labels.append(regime.replace('_', ' ').title())
            
            if regime_data:
                ax2.boxplot(regime_data, labels=regime_labels)
                ax2.set_title('Return Distribution by Regime', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Daily Returns', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # 3. 체제별 성과 지표
            metrics_data = []
            metrics_labels = []
            for regime, returns in regime_returns.items():
                if returns:
                    returns_array = np.array(returns)
                    total_return = np.prod(1 + returns_array) - 1
                    sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
                    max_drawdown = self._calculate_max_drawdown(returns_array)
                    
                    metrics_data.append([total_return, sharpe_ratio, abs(max_drawdown)])
                    metrics_labels.append(regime.replace('_', ' ').title())
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data, 
                                        index=metrics_labels,
                                        columns=['Total Return', 'Sharpe Ratio', 'Max Drawdown'])
                
                # 정규화된 메트릭스 히트맵
                metrics_normalized = metrics_df.div(metrics_df.max())
                sns.heatmap(metrics_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                           ax=ax3, cbar_kws={'label': 'Normalized Score'})
                ax3.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
            
            # 4. 체제별 거래 빈도
            regime_counts = {regime: len(returns) for regime, returns in regime_returns.items() if returns}
            if regime_counts:
                regimes = list(regime_counts.keys())
                counts = list(regime_counts.values())
                colors = [self.colors.get(regime, '#000000') for regime in regimes]
                
                ax4.bar([r.replace('_', ' ').title() for r in regimes], counts, color=colors)
                ax4.set_title('Trading Frequency by Regime', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Number of Trades', fontsize=12)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            self.logger.info("체제별 성과 비교 차트 생성 완료")
            return fig
            
        except Exception as e:
            self.logger.error(f"체제별 성과 비교 차트 생성 실패: {e}")
            raise
    
    def plot_trade_analysis(self, trade_history: List[Dict], 
                          title: str = "거래 분석") -> plt.Figure:
        """
        거래 분석 차트 생성
        
        Args:
            trade_history: 거래 히스토리
            title: 차트 제목
            
        Returns:
            matplotlib Figure 객체
        """
        try:
            if not trade_history:
                # 빈 차트 생성
                fig, ax = plt.subplots(figsize=self.figure_size)
                ax.text(0.5, 0.5, 'No Trade Data Available', 
                       ha='center', va='center', fontsize=16, transform=ax.transAxes)
                ax.set_title(title, fontsize=16, fontweight='bold')
                return fig
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figure_size[0]*1.5, self.figure_size[1]))
            
            # 거래 데이터를 DataFrame으로 변환
            df = pd.DataFrame(trade_history)
            
            # 1. 액션별 거래 분포
            action_counts = df['action'].value_counts()
            action_names = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
            action_labels = [action_names[i] if i < len(action_names) else f'Action {i}' 
                           for i in action_counts.index]
            
            ax1.pie(action_counts.values, labels=action_labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Trade Distribution by Action', fontsize=14, fontweight='bold')
            
            # 2. 시간별 거래 빈도
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                hourly_counts = df['hour'].value_counts().sort_index()
                
                ax2.bar(hourly_counts.index, hourly_counts.values, alpha=0.7)
                ax2.set_title('Trading Frequency by Hour', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Hour of Day', fontsize=12)
                ax2.set_ylabel('Number of Trades', fontsize=12)
                ax2.grid(True, alpha=0.3)
            
            # 3. 거래 성공률
            if 'success' in df.columns:
                success_rate = df['success'].mean()
                ax3.bar(['Successful', 'Failed'], 
                       [success_rate, 1 - success_rate], 
                       color=['green', 'red'], alpha=0.7)
                ax3.set_title(f'Trade Success Rate: {success_rate:.1%}', 
                             fontsize=14, fontweight='bold')
                ax3.set_ylabel('Proportion', fontsize=12)
                ax3.set_ylim(0, 1)
            
            # 4. 수수료 및 슬리피지 분석
            if 'fees_paid' in df.columns and 'slippage_cost' in df.columns:
                total_fees = df['fees_paid'].sum()
                total_slippage = df['slippage_cost'].sum()
                
                ax4.bar(['Fees', 'Slippage'], [total_fees, total_slippage], 
                       color=['orange', 'purple'], alpha=0.7)
                ax4.set_title('Trading Costs', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Total Cost', fontsize=12)
                ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.2f}'))
            
            plt.tight_layout()
            
            self.logger.info("거래 분석 차트 생성 완료")
            return fig
            
        except Exception as e:
            self.logger.error(f"거래 분석 차트 생성 실패: {e}")
            raise
    
    def create_interactive_dashboard(self, backtest_results: Dict[str, Any], 
                                   output_path: str) -> str:
        """
        인터랙티브 대시보드 생성
        
        Args:
            backtest_results: 백테스팅 결과
            output_path: 출력 파일 경로
            
        Returns:
            생성된 HTML 파일 경로
        """
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Cumulative Returns', 'Drawdown', 
                              'Monthly Returns', 'Rolling Sharpe Ratio',
                              'Regime Distribution', 'Trade Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. 누적 수익률
            if 'walk_forward_results' in backtest_results:
                returns = []
                for result in backtest_results['walk_forward_results']:
                    if 'returns' in result:
                        returns.extend(result['returns'])
                
                if returns:
                    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
                    dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='D')
                    
                    fig.add_trace(
                        go.Scatter(x=dates, y=cumulative_returns, 
                                 name='Portfolio', line=dict(color='blue')),
                        row=1, col=1
                    )
            
            # 2. 드로우다운
            if returns:
                returns_array = np.array(returns)
                cumulative_returns = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                
                fig.add_trace(
                    go.Scatter(x=dates, y=drawdowns, 
                             name='Drawdown', fill='tonexty', 
                             line=dict(color='red')),
                    row=1, col=2
                )
            
            # 3. 월별 수익률
            if returns:
                monthly_returns = pd.Series(returns).resample('M').apply(lambda x: np.prod(1 + x) - 1)
                fig.add_trace(
                    go.Bar(x=monthly_returns.index, y=monthly_returns.values,
                          name='Monthly Returns'),
                    row=2, col=1
                )
            
            # 4. 롤링 샤프 비율
            if returns:
                rolling_sharpe = pd.Series(returns).rolling(30).apply(
                    lambda x: np.mean(x) / (np.std(x) + 1e-8) * np.sqrt(252)
                )
                fig.add_trace(
                    go.Scatter(x=dates, y=rolling_sharpe, 
                             name='Rolling Sharpe', line=dict(color='green')),
                    row=2, col=2
                )
            
            # 5. 체제 분포
            regime_dist = backtest_results.get('regime_analysis', {})
            if regime_dist:
                regimes = list(regime_dist.keys())
                counts = [len(regime_dist[regime]) for regime in regimes]
                
                fig.add_trace(
                    go.Pie(labels=regimes, values=counts, name="Regime Distribution"),
                    row=3, col=1
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title_text="ESWA Dynamic Ensemble Trading System - Performance Dashboard",
                showlegend=True,
                height=1200,
                template="plotly_white"
            )
            
            # HTML 파일로 저장
            html_path = Path(output_path) / "performance_dashboard.html"
            pyo.plot(fig, filename=str(html_path), auto_open=False)
            
            self.logger.info(f"인터랙티브 대시보드 생성 완료: {html_path}")
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"인터랙티브 대시보드 생성 실패: {e}")
            raise
    
    def generate_html_report(self, backtest_results: Dict[str, Any], 
                           output_path: str) -> str:
        """
        HTML 보고서 생성
        
        Args:
            backtest_results: 백테스팅 결과
            output_path: 출력 파일 경로
            
        Returns:
            생성된 HTML 파일 경로
        """
        try:
            performance_metrics = backtest_results.get('performance_metrics', {})
            backtest_summary = backtest_results.get('backtest_summary', {})
            
            # HTML 템플릿
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>ESWA 동적 앙상블 거래 시스템 - 성과 보고서</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .metric-card {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                    .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                    .positive {{ color: #27ae60; }}
                    .negative {{ color: #e74c3c; }}
                    .neutral {{ color: #f39c12; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #3498db; color: white; }}
                    .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>ESWA 동적 앙상블 강화학습 거래 시스템</h1>
                    <h2>성과 보고서</h2>
                    
                    <p><strong>생성 일시:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>주요 성과 지표</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value {'positive' if performance_metrics.get('total_return', 0) > 0 else 'negative'}">
                                {performance_metrics.get('total_return', 0):.2%}
                            </div>
                            <div class="metric-label">총 수익률</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if performance_metrics.get('cagr', 0) > 0 else 'negative'}">
                                {performance_metrics.get('cagr', 0):.2%}
                            </div>
                            <div class="metric-label">연평균 수익률 (CAGR)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if performance_metrics.get('sharpe_ratio', 0) > 1 else 'neutral' if performance_metrics.get('sharpe_ratio', 0) > 0 else 'negative'}">
                                {performance_metrics.get('sharpe_ratio', 0):.3f}
                            </div>
                            <div class="metric-label">샤프 비율</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value negative">
                                {performance_metrics.get('max_drawdown', 0):.2%}
                            </div>
                            <div class="metric-label">최대 낙폭</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if performance_metrics.get('win_rate', 0) > 0.5 else 'negative'}">
                                {performance_metrics.get('win_rate', 0):.2%}
                            </div>
                            <div class="metric-label">승률</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if performance_metrics.get('profit_factor', 0) > 1 else 'negative'}">
                                {performance_metrics.get('profit_factor', 0):.2f}
                            </div>
                            <div class="metric-label">수익 팩터</div>
                        </div>
                    </div>
                    
                    <h2>거래 통계</h2>
                    <table>
                        <tr>
                            <th>지표</th>
                            <th>값</th>
                        </tr>
                        <tr>
                            <td>총 거래 수</td>
                            <td>{backtest_summary.get('total_trades', 0):,}</td>
                        </tr>
                        <tr>
                            <td>성공률</td>
                            <td>{backtest_summary.get('success_rate', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td>변동성</td>
                            <td>{performance_metrics.get('volatility', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td>검증 기간 수</td>
                            <td>{backtest_results.get('total_periods', 0)}</td>
                        </tr>
                    </table>
                    
                    <h2>시스템 정보</h2>
                    <p><strong>알고리즘:</strong> ESWA (Ensemble of Specialized Weak Agents)</p>
                    <p><strong>강화학습:</strong> PPO (Proximal Policy Optimization)</p>
                    <p><strong>앙상블 방법:</strong> 동적 가중치 할당 (Softmax, T=10)</p>
                    <p><strong>시장 체제:</strong> Bull Market, Bear Market, Sideways Market</p>
                    <p><strong>검증 방법:</strong> Walk-Forward Expanding Window Cross-Validation</p>
                    
                    <div class="footer">
                        <p>ESWA Dynamic Ensemble Trading System v1.0.0</p>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # HTML 파일 저장
            html_path = Path(output_path) / "performance_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML 보고서 생성 완료: {html_path}")
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"HTML 보고서 생성 실패: {e}")
            raise
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            return np.min(drawdowns)
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0
    
    def save_all_charts(self, backtest_results: Dict[str, Any], 
                       output_dir: str) -> List[str]:
        """
        모든 차트를 저장
        
        Args:
            backtest_results: 백테스팅 결과
            output_dir: 출력 디렉토리
            
        Returns:
            저장된 파일 경로 리스트
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # 백테스팅 결과에서 데이터 추출
            returns = []
            regime_returns = {}
            trade_history = []
            
            if 'walk_forward_results' in backtest_results:
                for result in backtest_results['walk_forward_results']:
                    if 'returns' in result:
                        returns.extend(result['returns'])
                    if 'trades' in result:
                        trade_history.extend(result['trades'])
                    if 'regime_distribution' in result:
                        regime = result.get('regime', 'unknown')
                        if regime not in regime_returns:
                            regime_returns[regime] = []
                        regime_returns[regime].extend(result.get('returns', []))
            
            # 1. 누적 수익률 차트
            if returns:
                fig1 = self.plot_cumulative_returns(returns)
                file1 = output_path / "cumulative_returns.png"
                fig1.savefig(file1, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig1)
                saved_files.append(str(file1))
            
            # 2. 드로우다운 차트
            if returns:
                fig2 = self.plot_drawdown(returns)
                file2 = output_path / "drawdown_analysis.png"
                fig2.savefig(file2, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig2)
                saved_files.append(str(file2))
            
            # 3. 체제별 성과 비교 차트
            if regime_returns:
                fig3 = self.plot_regime_performance(regime_returns)
                file3 = output_path / "regime_performance.png"
                fig3.savefig(file3, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig3)
                saved_files.append(str(file3))
            
            # 4. 거래 분석 차트
            if trade_history:
                fig4 = self.plot_trade_analysis(trade_history)
                file4 = output_path / "trade_analysis.png"
                fig4.savefig(file4, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig4)
                saved_files.append(str(file4))
            
            # 5. 인터랙티브 대시보드
            dashboard_path = self.create_interactive_dashboard(backtest_results, str(output_path))
            saved_files.append(dashboard_path)
            
            # 6. HTML 보고서
            report_path = self.generate_html_report(backtest_results, str(output_path))
            saved_files.append(report_path)
            
            self.logger.info(f"모든 차트 저장 완료: {len(saved_files)}개 파일")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"차트 저장 실패: {e}")
            raise


# 편의 함수들
def create_performance_visualizer(config: Dict) -> PerformanceVisualizer:
    """성과 시각화기 생성 편의 함수"""
    return PerformanceVisualizer(config)


def generate_quick_report(backtest_results: Dict[str, Any], output_dir: str) -> List[str]:
    """빠른 보고서 생성 편의 함수"""
    visualizer = PerformanceVisualizer({})
    return visualizer.save_all_charts(backtest_results, output_dir)
