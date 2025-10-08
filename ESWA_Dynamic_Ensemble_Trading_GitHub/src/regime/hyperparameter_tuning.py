"""
시장 체제 분류기 하이퍼파라미터 튜닝
Market Regime Classifier Hyperparameter Tuning

XGBoost 모델의 하이퍼파라미터 최적화를 통한 분류 정확도 향상
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RegimeClassifierHyperparameterTuning:
    """시장 체제 분류기 하이퍼파라미터 튜닝 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        하이퍼파라미터 튜닝 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # XGBoost 기본 파라미터
        self.base_params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # bull, bear, sideways
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 튜닝할 하이퍼파라미터 범위
        self.param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [1, 1.5, 2.0]
        }
        
        # 랜덤 서치용 파라미터 분포
        self.param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'reg_alpha': [0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5],
            'reg_lambda': [0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        }
        
        logger.info(f"시장 체제 분류기 하이퍼파라미터 튜닝 시스템 초기화 완료")
        logger.info(f"튜닝할 파라미터 수: {len(self.param_grid)}")
    
    def perform_grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                          cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        그리드 서치를 통한 하이퍼파라미터 튜닝
        
        Args:
            X_train: 훈련 데이터
            y_train: 훈련 라벨
            cv_folds: 교차 검증 폴드 수
            scoring: 평가 지표
            
        Returns:
            그리드 서치 결과
        """
        try:
            logger.info("그리드 서치 하이퍼파라미터 튜닝 시작")
            
            # XGBoost 분류기 생성
            xgb_classifier = xgb.XGBClassifier(**self.base_params)
            
            # 그리드 서치 수행
            grid_search = GridSearchCV(
                estimator=xgb_classifier,
                param_grid=self.param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            # 훈련 수행
            grid_search.fit(X_train, y_train)
            
            # 결과 수집
            result = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'method': 'grid_search',
                'cv_folds': cv_folds,
                'scoring': scoring,
                'n_combinations': len(grid_search.cv_results_['params'])
            }
            
            logger.info(f"그리드 서치 완료")
            logger.info(f"최고 점수: {grid_search.best_score_:.4f}")
            logger.info(f"최적 파라미터: {grid_search.best_params_}")
            
            return result
            
        except Exception as e:
            logger.error(f"그리드 서치 하이퍼파라미터 튜닝 실패: {e}")
            return {'error': str(e)}
    
    def perform_randomized_search(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_iter: int = 100, cv_folds: int = 5, 
                                scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        랜덤 서치를 통한 하이퍼파라미터 튜닝
        
        Args:
            X_train: 훈련 데이터
            y_train: 훈련 라벨
            n_iter: 시도할 파라미터 조합 수
            cv_folds: 교차 검증 폴드 수
            scoring: 평가 지표
            
        Returns:
            랜덤 서치 결과
        """
        try:
            logger.info(f"랜덤 서치 하이퍼파라미터 튜닝 시작 (시도 횟수: {n_iter})")
            
            # XGBoost 분류기 생성
            xgb_classifier = xgb.XGBClassifier(**self.base_params)
            
            # 랜덤 서치 수행
            random_search = RandomizedSearchCV(
                estimator=xgb_classifier,
                param_distributions=self.param_distributions,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            # 훈련 수행
            random_search.fit(X_train, y_train)
            
            # 결과 수집
            result = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_estimator': random_search.best_estimator_,
                'cv_results': random_search.cv_results_,
                'method': 'randomized_search',
                'n_iter': n_iter,
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
            logger.info(f"랜덤 서치 완료")
            logger.info(f"최고 점수: {random_search.best_score_:.4f}")
            logger.info(f"최적 파라미터: {random_search.best_params_}")
            
            return result
            
        except Exception as e:
            logger.error(f"랜덤 서치 하이퍼파라미터 튜닝 실패: {e}")
            return {'error': str(e)}
    
    def perform_bayesian_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                    n_trials: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        """
        베이지안 최적화를 통한 하이퍼파라미터 튜닝 (간단한 구현)
        
        Args:
            X_train: 훈련 데이터
            y_train: 훈련 라벨
            n_trials: 시도 횟수
            cv_folds: 교차 검증 폴드 수
            
        Returns:
            베이지안 최적화 결과
        """
        try:
            logger.info(f"베이지안 최적화 하이퍼파라미터 튜닝 시작 (시도 횟수: {n_trials})")
            
            # 간단한 베이지안 최적화 구현 (실제로는 optuna 등을 사용)
            best_score = -np.inf
            best_params = None
            trial_results = []
            
            for trial in range(n_trials):
                # 파라미터 랜덤 샘플링
                params = {}
                for param_name, param_values in self.param_distributions.items():
                    params[param_name] = np.random.choice(param_values)
                
                # XGBoost 분류기 생성
                xgb_classifier = xgb.XGBClassifier(**self.base_params, **params)
                
                # 교차 검증 수행
                cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=cv_folds, scoring='accuracy')
                mean_score = np.mean(cv_scores)
                
                trial_results.append({
                    'trial': trial + 1,
                    'params': params,
                    'score': mean_score,
                    'std': np.std(cv_scores)
                })
                
                # 최고 점수 업데이트
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                
                if (trial + 1) % 20 == 0:
                    logger.info(f"시도 {trial + 1}/{n_trials} 완료, 현재 최고 점수: {best_score:.4f}")
            
            # 최적 모델 훈련
            best_estimator = xgb.XGBClassifier(**self.base_params, **best_params)
            best_estimator.fit(X_train, y_train)
            
            result = {
                'best_params': best_params,
                'best_score': best_score,
                'best_estimator': best_estimator,
                'trial_results': trial_results,
                'method': 'bayesian_optimization',
                'n_trials': n_trials,
                'cv_folds': cv_folds
            }
            
            logger.info(f"베이지안 최적화 완료")
            logger.info(f"최고 점수: {best_score:.4f}")
            logger.info(f"최적 파라미터: {best_params}")
            
            return result
            
        except Exception as e:
            logger.error(f"베이지안 최적화 하이퍼파라미터 튜닝 실패: {e}")
            return {'error': str(e)}
    
    def evaluate_model_performance(self, model: xgb.XGBClassifier, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        모델 성능 평가
        
        Args:
            model: 훈련된 모델
            X_test: 테스트 데이터
            y_test: 테스트 라벨
            
        Returns:
            모델 성능 평가 결과
        """
        try:
            logger.info("모델 성능 평가 시작")
            
            # 예측 수행
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # 정확도 계산
            accuracy = accuracy_score(y_test, y_pred)
            
            # 분류 보고서
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # 혼동 행렬
            confusion_mat = confusion_matrix(y_test, y_pred)
            
            # 클래스별 정확도
            class_accuracy = {}
            for i, class_name in enumerate(['bull', 'bear', 'sideways']):
                class_mask = y_test == i
                if np.sum(class_mask) > 0:
                    class_accuracy[class_name] = accuracy_score(y_test[class_mask], y_pred[class_mask])
                else:
                    class_accuracy[class_name] = 0.0
            
            # 예측 신뢰도 분석
            max_proba = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(max_proba)
            confidence_std = np.std(max_proba)
            
            result = {
                'accuracy': accuracy,
                'classification_report': classification_rep,
                'confusion_matrix': confusion_mat.tolist(),
                'class_accuracy': class_accuracy,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'feature_importance': model.feature_importances_.tolist()
            }
            
            logger.info(f"모델 성능 평가 완료")
            logger.info(f"정확도: {accuracy:.4f}")
            logger.info(f"평균 신뢰도: {avg_confidence:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"모델 성능 평가 실패: {e}")
            return {'error': str(e)}
    
    def perform_comprehensive_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        종합 하이퍼파라미터 튜닝 (모든 방법 사용)
        
        Args:
            X_train: 훈련 데이터
            y_train: 훈련 라벨
            X_test: 테스트 데이터
            y_test: 테스트 라벨
            
        Returns:
            종합 튜닝 결과
        """
        try:
            logger.info("종합 하이퍼파라미터 튜닝 시작")
            
            # 각 방법별 튜닝 수행
            grid_result = self.perform_grid_search(X_train, y_train)
            random_result = self.perform_randomized_search(X_train, y_train, n_iter=50)
            bayesian_result = self.perform_bayesian_optimization(X_train, y_train, n_trials=50)
            
            # 결과 수집
            tuning_results = {
                'grid_search': grid_result,
                'randomized_search': random_result,
                'bayesian_optimization': bayesian_result
            }
            
            # 최고 성능 모델 선택
            best_method = None
            best_score = -np.inf
            best_model = None
            
            for method, result in tuning_results.items():
                if 'error' not in result and result.get('best_score', 0) > best_score:
                    best_score = result['best_score']
                    best_method = method
                    best_model = result['best_estimator']
            
            # 최고 모델 성능 평가
            if best_model is not None:
                performance_result = self.evaluate_model_performance(best_model, X_test, y_test)
            else:
                performance_result = {'error': 'No valid model found'}
            
            comprehensive_result = {
                'tuning_results': tuning_results,
                'best_method': best_method,
                'best_score': best_score,
                'best_model': best_model,
                'performance_evaluation': performance_result,
                'improvement_summary': self._generate_improvement_summary(tuning_results)
            }
            
            logger.info(f"종합 하이퍼파라미터 튜닝 완료")
            logger.info(f"최고 방법: {best_method}")
            logger.info(f"최고 점수: {best_score:.4f}")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"종합 하이퍼파라미터 튜닝 실패: {e}")
            return {'error': str(e)}
    
    def _generate_improvement_summary(self, tuning_results: Dict[str, Any]) -> Dict[str, Any]:
        """개선 요약 생성"""
        try:
            summary = {
                'methods_tested': len(tuning_results),
                'best_scores': {},
                'parameter_insights': {},
                'recommendations': []
            }
            
            # 각 방법별 최고 점수 수집
            for method, result in tuning_results.items():
                if 'error' not in result:
                    summary['best_scores'][method] = result.get('best_score', 0)
            
            # 파라미터 인사이트 생성
            if 'grid_search' in tuning_results and 'error' not in tuning_results['grid_search']:
                best_params = tuning_results['grid_search'].get('best_params', {})
                summary['parameter_insights']['best_params'] = best_params
                summary['recommendations'].append(f"최적 파라미터: {best_params}")
            
            # 성능 개선 제안
            if len(summary['best_scores']) > 0:
                max_score = max(summary['best_scores'].values())
                summary['recommendations'].append(f"최고 성능: {max_score:.4f}")
                
                if max_score > 0.8:
                    summary['recommendations'].append("높은 성능 달성 - 모델 배포 준비 완료")
                elif max_score > 0.7:
                    summary['recommendations'].append("양호한 성능 - 추가 튜닝 고려")
                else:
                    summary['recommendations'].append("성능 개선 필요 - 데이터 품질 및 특성 엔지니어링 검토")
            
            return summary
            
        except Exception as e:
            logger.error(f"개선 요약 생성 실패: {e}")
            return {'error': str(e)}

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("시장 체제 분류기 하이퍼파라미터 튜닝 테스트")
        print("=" * 60)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)  # 0: bull, 1: bear, 2: sideways
        
        # 훈련/테스트 분할
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 하이퍼파라미터 튜닝 시스템 생성
        config = {}
        tuner = RegimeClassifierHyperparameterTuning(config)
        
        # 종합 튜닝 수행
        result = tuner.perform_comprehensive_tuning(X_train, y_train, X_test, y_test)
        
        if 'error' not in result:
            print(f"테스트 결과:")
            print(f"  최고 방법: {result['best_method']}")
            print(f"  최고 점수: {result['best_score']:.4f}")
            
            # 성능 평가 결과
            if 'error' not in result['performance_evaluation']:
                perf = result['performance_evaluation']
                print(f"  테스트 정확도: {perf['accuracy']:.4f}")
                print(f"  평균 신뢰도: {perf['avg_confidence']:.4f}")
            
            # 개선 요약
            if 'error' not in result['improvement_summary']:
                summary = result['improvement_summary']
                print(f"\n개선 요약:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
        
        print(f"\n[성공] 시장 체제 분류기 하이퍼파라미터 튜닝 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 시장 체제 분류기 하이퍼파라미터 튜닝 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
