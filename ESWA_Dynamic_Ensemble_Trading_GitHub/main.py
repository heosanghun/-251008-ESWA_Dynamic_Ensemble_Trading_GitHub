"""
ESWA 동적 앙상블 강화학습 거래 시스템 - 메인 실행 스크립트
ESWA Dynamic Ensemble Reinforcement Learning Trading System - Main Execution Script

이 스크립트는 ESWA 시스템의 전체 워크플로우를 실행합니다:
1. 데이터 로딩 및 전처리
2. 시스템 훈련
3. 백테스팅 실행
4. 성과 분석 및 시각화
5. 결과 저장

작성자: AI Assistant
버전: 1.0.0
날짜: 2025-09-04
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.eswa_system import ESWADynamicEnsembleSystem, create_eswa_system


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger('ESWA_Main')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러
    log_file = project_root / "logs" / f"eswa_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def generate_sample_data(num_days: int = 1000) -> pd.DataFrame:
    """샘플 데이터 생성 (실제 구현에서는 실제 데이터 사용)"""
    logger = logging.getLogger('ESWA_Main')
    logger.info(f"샘플 데이터 생성: {num_days}일")
    
    try:
        # 날짜 범위 생성
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # OHLCV 데이터 생성 (간단한 랜덤 워크)
        np.random.seed(42)
        n_points = len(dates)
        
        # 가격 데이터 (로그 정규 분포)
        returns = np.random.normal(0.0001, 0.02, n_points)  # 시간당 수익률
        prices = 50000 * np.exp(np.cumsum(returns))  # BTC 가격 시뮬레이션
        
        # OHLCV 데이터 생성
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 간단한 OHLCV 생성
            volatility = np.random.uniform(0.005, 0.02)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        
        logger.info(f"샘플 데이터 생성 완료: {len(df)}개 레코드")
        return df
        
    except Exception as e:
        logger.error(f"샘플 데이터 생성 실패: {e}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    logger = logging.getLogger('ESWA_Main')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"설정 파일 로드 완료: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        raise


def run_training(system: ESWADynamicEnsembleSystem, config: Dict[str, Any]) -> bool:
    """시스템 훈련 실행"""
    logger = logging.getLogger('ESWA_Main')
    
    try:
        logger.info("시스템 훈련 시작")
        
        # 훈련 파라미터
        episodes = config.get('training', {}).get('episodes', 1000)
        
        # 훈련 실행
        system.train(episodes)
        
        logger.info("시스템 훈련 완료")
        return True
        
    except Exception as e:
        logger.error(f"시스템 훈련 실패: {e}")
        return False


def run_backtesting(system: ESWADynamicEnsembleSystem, config: Dict[str, Any]) -> Dict[str, Any]:
    """백테스팅 실행"""
    logger = logging.getLogger('ESWA_Main')
    
    try:
        logger.info("백테스팅 시작")
        
        # 백테스팅 파라미터
        backtest_config = config.get('backtesting', {})
        start_date = backtest_config.get('start_date')
        end_date = backtest_config.get('end_date')
        
        # 백테스팅 실행
        results = system.backtest(start_date, end_date)
        
        logger.info("백테스팅 완료")
        return results
        
    except Exception as e:
        logger.error(f"백테스팅 실패: {e}")
        return {}


def save_results(results: Dict[str, Any], output_dir: str):
    """결과 저장"""
    logger = logging.getLogger('ESWA_Main')
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 결과를 JSON으로 저장
        results_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"결과 저장 완료: {results_file}")
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")


def print_performance_summary(results: Dict[str, Any]):
    """성과 요약 출력"""
    logger = logging.getLogger('ESWA_Main')
    
    try:
        if not results:
            logger.warning("출력할 결과가 없습니다.")
            return
        
        performance_metrics = results.get('performance_metrics', {})
        backtest_summary = results.get('backtest_summary', {})
        
        print("\n" + "="*60)
        print("ESWA 동적 앙상블 거래 시스템 - 성과 요약")
        print("="*60)
        
        if performance_metrics:
            print(f"총 수익률:     {performance_metrics.get('total_return', 0):.2%}")
            print(f"CAGR:          {performance_metrics.get('cagr', 0):.2%}")
            print(f"샤프 비율:     {performance_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"최대 낙폭:     {performance_metrics.get('max_drawdown', 0):.2%}")
            print(f"승률:          {performance_metrics.get('win_rate', 0):.2%}")
            print(f"수익 팩터:     {performance_metrics.get('profit_factor', 0):.2f}")
            print(f"변동성:        {performance_metrics.get('volatility', 0):.2%}")
        
        if backtest_summary:
            print(f"총 거래 수:    {backtest_summary.get('total_trades', 0):,}")
            print(f"성공률:        {backtest_summary.get('success_rate', 0):.2%}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"성과 요약 출력 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='ESWA 동적 앙상블 강화학습 거래 시스템')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--data-days', type=int, default=1000,
                       help='생성할 샘플 데이터 일수')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='훈련 에피소드 수')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로깅 레벨')
    parser.add_argument('--skip-training', action='store_true',
                       help='훈련 건너뛰기 (기존 모델 사용)')
    parser.add_argument('--skip-backtesting', action='store_true',
                       help='백테스팅 건너뛰기')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("ESWA 동적 앙상블 강화학습 거래 시스템 시작")
        logger.info(f"설정 파일: {args.config}")
        logger.info(f"데이터 일수: {args.data_days}")
        logger.info(f"훈련 에피소드: {args.episodes}")
        
        # 1. 설정 파일 로드
        config = load_config(args.config)
        
        # 2. ESWA 시스템 초기화
        logger.info("ESWA 시스템 초기화")
        system = create_eswa_system(args.config)
        
        # 3. 데이터 로딩
        logger.info("데이터 로딩")
        sample_data = generate_sample_data(args.data_days)
        system.load_data(sample_data)
        
        # 4. 시스템 훈련
        if not args.skip_training:
            success = run_training(system, config)
            if not success:
                logger.error("훈련 실패로 인해 프로그램 종료")
                return 1
        else:
            logger.info("훈련 건너뛰기")
        
        # 5. 백테스팅 실행
        results = {}
        if not args.skip_backtesting:
            results = run_backtesting(system, config)
            if not results:
                logger.error("백테스팅 실패")
                return 1
        else:
            logger.info("백테스팅 건너뛰기")
        
        # 6. 결과 저장
        if results:
            save_results(results, args.output_dir)
        
        # 7. 성과 요약 출력
        print_performance_summary(results)
        
        logger.info("ESWA 시스템 실행 완료")
        return 0
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
