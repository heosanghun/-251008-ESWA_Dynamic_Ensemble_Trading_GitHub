"""
시장 체제 분류기 모델 재훈련 시스템
Market Regime Classifier Model Retraining System

주기적 모델 재훈련 및 성능 모니터링을 통한 분류 정확도 향상
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RegimeClassifierModelRetraining:
    """시장 체제 분류기 모델 재훈련 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        모델 재훈련 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 모델 재훈련 파라미터
        self.retraining_frequency = config.get('retraining_frequency', 30)  # 일
        self.min_samples_for_retraining = config.get('min_samples_for_retraining', 1000)
        self.performance_threshold = config.get('performance_threshold', 0.7)
        self.performance_degradation_threshold = config.get('performance_degradation_threshold', 0.05)
        
        # 모델 저장 경로
        self.model_save_path = config.get('model_save_path', 'models/regime_classifier')
        self.backup_model_path = config.get('backup_model_path', 'models/regime_classifier_backup')
        
        # 데이터 전처리
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # 모델 성능 추적
        self.performance_history = []
        self.last_retraining_date = None
        
        logger.info(f"시장 체제 분류기 모델 재훈련 시스템 초기화 완료")
        logger.info(f"재훈련 주기: {self.retraining_frequency}일")
        logger.info(f"최소 샘플 수: {self.min_samples_for_retraining}")
        logger.info(f"성능 임계값: {self.performance_threshold}")
    
    def prepare_training_data(self, data: pd.DataFrame, 
                            feature_columns: List[str],
                            target_column: str = 'regime') -> Tuple[np.ndarray, np.ndarray]:
        """
        훈련 데이터 준비
        
        Args:
            data: 원본 데이터
            feature_columns: 특성 컬럼 목록
            target_column: 타겟 컬럼명
            
        Returns:
            특성 배열과 타겟 배열
        """
        try:
            logger.info("훈련 데이터 준비 시작")
            
            # 결측값 제거
            clean_data = data.dropna(subset=feature_columns + [target_column])
            
            if len(clean_data) < self.min_samples_for_retraining:
                raise ValueError(f"데이터 부족: {len(clean_data)} < {self.min_samples_for_retraining}")
            
            # 특성과 타겟 분리
            X = clean_data[feature_columns].values
            y = clean_data[target_column].values
            
            # 타겟 라벨 인코딩 (문자열을 숫자로 변환)
            if y.dtype == object:
                unique_labels = np.unique(y)
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                y = np.array([label_mapping[label] for label in y])
                self.label_mapping = label_mapping
            else:
                self.label_mapping = None
            
            logger.info(f"훈련 데이터 준비 완료: {X.shape[0]}개 샘플, {X.shape[1]}개 특성")
            
            return X, y
            
        except Exception as e:
            logger.error(f"훈련 데이터 준비 실패: {e}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   model_params: Optional[Dict[str, Any]] = None) -> xgb.XGBClassifier:
        """
        모델 훈련
        
        Args:
            X: 특성 배열
            y: 타겟 배열
            model_params: 모델 파라미터
            
        Returns:
            훈련된 모델
        """
        try:
            logger.info("모델 훈련 시작")
            
            # 기본 파라미터 설정
            if model_params is None:
                model_params = {
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 42,
                    'n_jobs': -1
                }
            
            # XGBoost 분류기 생성
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,  # bull, bear, sideways
                **model_params
            )
            
            # 데이터 스케일링
            if not self.is_scaler_fitted:
                X_scaled = self.scaler.fit_transform(X)
                self.is_scaler_fitted = True
            else:
                X_scaled = self.scaler.transform(X)
            
            # 훈련/검증 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 모델 훈련
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # 검증 성능 평가
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            logger.info(f"모델 훈련 완료")
            logger.info(f"검증 정확도: {val_accuracy:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"모델 훈련 실패: {e}")
            raise
    
    def evaluate_model_performance(self, model: xgb.XGBClassifier, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        모델 성능 평가
        
        Args:
            model: 훈련된 모델
            X_test: 테스트 데이터
            y_test: 테스트 라벨
            
        Returns:
            성능 평가 결과
        """
        try:
            logger.info("모델 성능 평가 시작")
            
            # 데이터 스케일링
            X_test_scaled = self.scaler.transform(X_test)
            
            # 예측 수행
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # 정확도 계산
            accuracy = accuracy_score(y_test, y_pred)
            
            # 분류 보고서
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # 혼동 행렬
            confusion_mat = confusion_matrix(y_test, y_pred)
            
            # 클래스별 정확도
            class_accuracy = {}
            class_names = ['bull', 'bear', 'sideways']
            for i, class_name in enumerate(class_names):
                class_mask = y_test == i
                if np.sum(class_mask) > 0:
                    class_accuracy[class_name] = accuracy_score(y_test[class_mask], y_pred[class_mask])
                else:
                    class_accuracy[class_name] = 0.0
            
            # 예측 신뢰도 분석
            max_proba = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(max_proba)
            confidence_std = np.std(max_proba)
            
            # 특성 중요도
            feature_importance = model.feature_importances_
            
            result = {
                'accuracy': accuracy,
                'classification_report': classification_rep,
                'confusion_matrix': confusion_mat.tolist(),
                'class_accuracy': class_accuracy,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'feature_importance': feature_importance.tolist(),
                'evaluation_date': datetime.now().isoformat()
            }
            
            logger.info(f"모델 성능 평가 완료")
            logger.info(f"정확도: {accuracy:.4f}")
            logger.info(f"평균 신뢰도: {avg_confidence:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"모델 성능 평가 실패: {e}")
            return {'error': str(e)}
    
    def save_model(self, model: xgb.XGBClassifier, 
                  performance_result: Dict[str, Any],
                  model_version: str = None) -> str:
        """
        모델 저장
        
        Args:
            model: 저장할 모델
            performance_result: 성능 평가 결과
            model_version: 모델 버전 (선택사항)
            
        Returns:
            저장된 모델 경로
        """
        try:
            logger.info("모델 저장 시작")
            
            # 모델 저장 디렉토리 생성
            os.makedirs(self.model_save_path, exist_ok=True)
            
            # 모델 버전 생성
            if model_version is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_version = f"regime_classifier_{timestamp}"
            
            # 모델 파일 경로
            model_file_path = os.path.join(self.model_save_path, f"{model_version}.pkl")
            
            # 모델과 관련 정보 저장
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'label_mapping': getattr(self, 'label_mapping', None),
                'performance_result': performance_result,
                'training_date': datetime.now().isoformat(),
                'model_version': model_version
            }
            
            with open(model_file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # 성능 히스토리 업데이트
            self.performance_history.append({
                'model_version': model_version,
                'accuracy': performance_result.get('accuracy', 0),
                'avg_confidence': performance_result.get('avg_confidence', 0),
                'training_date': datetime.now().isoformat()
            })
            
            # 마지막 재훈련 날짜 업데이트
            self.last_retraining_date = datetime.now()
            
            logger.info(f"모델 저장 완료: {model_file_path}")
            
            return model_file_path
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        모델 로드
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            로드된 모델 데이터
        """
        try:
            logger.info(f"모델 로드 시작: {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 스케일러와 라벨 매핑 복원
            self.scaler = model_data['scaler']
            self.is_scaler_fitted = True
            if 'label_mapping' in model_data:
                self.label_mapping = model_data['label_mapping']
            
            logger.info(f"모델 로드 완료")
            
            return model_data
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def should_retrain(self, current_performance: float = None) -> Dict[str, Any]:
        """
        재훈련 필요성 판단
        
        Args:
            current_performance: 현재 성능 (선택사항)
            
        Returns:
            재훈련 필요성 판단 결과
        """
        try:
            logger.info("재훈련 필요성 판단 시작")
            
            retrain_reasons = []
            should_retrain = False
            
            # 1. 시간 기반 재훈련 (주기적)
            if self.last_retraining_date is None:
                retrain_reasons.append("초기 모델 훈련 필요")
                should_retrain = True
            else:
                days_since_retraining = (datetime.now() - self.last_retraining_date).days
                if days_since_retraining >= self.retraining_frequency:
                    retrain_reasons.append(f"재훈련 주기 도달 ({days_since_retraining}일 경과)")
                    should_retrain = True
            
            # 2. 성능 기반 재훈련
            if current_performance is not None:
                if current_performance < self.performance_threshold:
                    retrain_reasons.append(f"성능 임계값 미달 ({current_performance:.4f} < {self.performance_threshold})")
                    should_retrain = True
                
                # 성능 저하 감지
                if len(self.performance_history) > 0:
                    recent_performance = self.performance_history[-1]['accuracy']
                    if recent_performance - current_performance > self.performance_degradation_threshold:
                        retrain_reasons.append(f"성능 저하 감지 ({recent_performance:.4f} -> {current_performance:.4f})")
                        should_retrain = True
            
            result = {
                'should_retrain': should_retrain,
                'reasons': retrain_reasons,
                'last_retraining_date': self.last_retraining_date.isoformat() if self.last_retraining_date else None,
                'days_since_retraining': (datetime.now() - self.last_retraining_date).days if self.last_retraining_date else None,
                'current_performance': current_performance,
                'performance_threshold': self.performance_threshold
            }
            
            logger.info(f"재훈련 필요성 판단 완료: {should_retrain}")
            if retrain_reasons:
                logger.info(f"재훈련 이유: {', '.join(retrain_reasons)}")
            
            return result
            
        except Exception as e:
            logger.error(f"재훈련 필요성 판단 실패: {e}")
            return {'should_retrain': False, 'error': str(e)}
    
    def perform_retraining(self, data: pd.DataFrame, 
                         feature_columns: List[str],
                         target_column: str = 'regime',
                         model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        모델 재훈련 수행
        
        Args:
            data: 훈련 데이터
            feature_columns: 특성 컬럼 목록
            target_column: 타겟 컬럼명
            model_params: 모델 파라미터
            
        Returns:
            재훈련 결과
        """
        try:
            logger.info("모델 재훈련 수행 시작")
            
            # 훈련 데이터 준비
            X, y = self.prepare_training_data(data, feature_columns, target_column)
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 모델 훈련
            model = self.train_model(X_train, y_train, model_params)
            
            # 성능 평가
            performance_result = self.evaluate_model_performance(model, X_test, y_test)
            
            # 모델 저장
            model_path = self.save_model(model, performance_result)
            
            result = {
                'success': True,
                'model_path': model_path,
                'performance_result': performance_result,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'retraining_date': datetime.now().isoformat()
            }
            
            logger.info(f"모델 재훈련 수행 완료")
            logger.info(f"훈련 샘플 수: {len(X_train)}")
            logger.info(f"테스트 정확도: {performance_result.get('accuracy', 0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"모델 재훈련 수행 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        try:
            if not self.performance_history:
                return {'message': 'No performance history available'}
            
            recent_performance = self.performance_history[-1]
            best_performance = max(self.performance_history, key=lambda x: x['accuracy'])
            
            summary = {
                'total_retrainings': len(self.performance_history),
                'recent_accuracy': recent_performance['accuracy'],
                'best_accuracy': best_performance['accuracy'],
                'best_model_version': best_performance['model_version'],
                'last_retraining_date': self.last_retraining_date.isoformat() if self.last_retraining_date else None,
                'performance_trend': self._calculate_performance_trend()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_trend(self) -> str:
        """성능 트렌드 계산"""
        try:
            if len(self.performance_history) < 2:
                return 'insufficient_data'
            
            recent_accuracies = [p['accuracy'] for p in self.performance_history[-5:]]
            
            if len(recent_accuracies) >= 2:
                trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
                if trend > 0.01:
                    return 'improving'
                elif trend < -0.01:
                    return 'declining'
                else:
                    return 'stable'
            else:
                return 'insufficient_data'
                
        except Exception as e:
            logger.error(f"성능 트렌드 계산 실패: {e}")
            return 'unknown'

def main():
    """테스트 함수"""
    try:
        print("=" * 60)
        print("시장 체제 분류기 모델 재훈련 시스템 테스트")
        print("=" * 60)
        
        # 테스트 설정
        config = {
            'retraining_frequency': 30,
            'min_samples_for_retraining': 100,
            'performance_threshold': 0.7,
            'performance_degradation_threshold': 0.05,
            'model_save_path': 'models/regime_classifier',
            'backup_model_path': 'models/regime_classifier_backup'
        }
        
        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # 특성 데이터 생성
        feature_data = np.random.randn(n_samples, n_features)
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        
        # 타겟 데이터 생성 (시장 체제)
        regime_labels = np.random.choice(['bull', 'bear', 'sideways'], n_samples)
        
        # DataFrame 생성
        test_data = pd.DataFrame(feature_data, columns=feature_columns)
        test_data['regime'] = regime_labels
        
        # 모델 재훈련 시스템 생성
        retraining_system = RegimeClassifierModelRetraining(config)
        
        # 재훈련 필요성 판단
        retrain_check = retraining_system.should_retrain()
        print(f"재훈련 필요성: {retrain_check['should_retrain']}")
        if retrain_check['reasons']:
            print(f"재훈련 이유: {', '.join(retrain_check['reasons'])}")
        
        # 모델 재훈련 수행
        if retrain_check['should_retrain']:
            retraining_result = retraining_system.perform_retraining(
                test_data, feature_columns, 'regime'
            )
            
            if retraining_result['success']:
                print(f"재훈련 성공!")
                print(f"  모델 경로: {retraining_result['model_path']}")
                print(f"  테스트 정확도: {retraining_result['performance_result']['accuracy']:.4f}")
                print(f"  평균 신뢰도: {retraining_result['performance_result']['avg_confidence']:.4f}")
        
        # 성능 요약
        performance_summary = retraining_system.get_performance_summary()
        if 'error' not in performance_summary:
            print(f"\n성능 요약:")
            print(f"  총 재훈련 횟수: {performance_summary['total_retrainings']}")
            print(f"  최근 정확도: {performance_summary['recent_accuracy']:.4f}")
            print(f"  최고 정확도: {performance_summary['best_accuracy']:.4f}")
            print(f"  성능 트렌드: {performance_summary['performance_trend']}")
        
        print(f"\n[성공] 시장 체제 분류기 모델 재훈련 시스템 테스트 완료!")
        
    except Exception as e:
        print(f"[실패] 시장 체제 분류기 모델 재훈련 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
