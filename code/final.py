import argparse
import os
import time
import re
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix,roc_auc_score
import scipy.stats as ss
import nsml
from nsml import DATASET_PATH
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin,TransformerMixin
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb ## XGBoost 불러오기
from abc import ABCMeta, abstractmethod
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier        
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import GradientBoostingClassifier as GBC
import tensorflow as tf
from joblib import Parallel
import torch
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli
from catboost import CatBoostClassifier 

def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        with open(os.path.join(path, 'model_lgb.pkl'), 'wb') as fp:
            joblib.dump(model, fp)
        print('Model saved')

    def load(path, *args, **kwargs):
        with open(os.path.join(path, 'model_lgb.pkl'), 'rb') as fp:
            temp_class = joblib.load(fp)
        nsml.copy(temp_class, model)
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)
    
    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

# 추론
def inference(path, model, **kwargs):
    data = preproc_data(pd.read_csv(path), train=False)
    
    pred_proba = model.predict_proba(data)[:, 1]
    pred_labels = np.where(pred_proba >= .5, 1, 0).reshape(-1)

    # output format
    # [(proba, label), (proba, label), ..., (proba, label)]
    results = [(proba, label) for proba, label in zip(pred_proba, pred_labels)]
    
    return results


# 데이터 전처리
def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()

        # NaN 값 mean으로 채우기
        data = data.fillna(data.mean())
        # pd.set_option('display.max_columns',None)
        # print(data)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        
        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E','GGT','DBP','AST','Ht','BUN','SBP','Cr','CrCl','TC','Wt']
        X = data.drop(columns=DROP_COLS).copy()
        y = label
        print(X.columns)

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y, 
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )
        
        print('개수 : ', len(X_train))
        lof = LocalOutlierFactor(contamination=0.01)
        y_pred_train = lof.fit_predict(X_train)

        #Remove outliers where 1 represent inliers and -1 represent outliers:
        X_train = X_train[np.where(y_pred_train == 1, True, False)]
        y_train = y_train[np.where(y_pred_train == 1, True, False)]
        smote = SMOTE()

        print('개수 : ', len(X_train))

        X_train, y_train = smote.fit_resample(X_train, y_train) 
        print("##############",X_train.shape )
        # Min-max scaling
        scaler = StandardScaler()

        X_cols = X.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_cols)
        X_train['gender_enc'] = X_train['gender_enc'].astype('int')
        print(scaler.mean_, scaler.scale_)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_cols)
        X_val['gender_enc'] = X_val['gender_enc'].astype('int')

        dataset['X_train'] = X_train
        dataset['y_train'] = y_train
        dataset['X_val'] = X_val
        dataset['y_val'] = y_val

        return dataset
    
    else:
        # NaN 값 0으로 채우기
        data = data.fillna(data.mean())

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E','GGT','DBP','AST','Ht','BUN','SBP','Cr','CrCl','TC','Wt']
        data = data.drop(columns=DROP_COLS).copy()

        scaler = StandardScaler()
        scaler.mean_= [51.1890166, 25.26426203, 66.01060534 , 5.7202224 , 99.46551724, 147.88794851 , 127.13320811, 52.84602789 , 4.38810748, 29.379934, 101.7804642, 0.26746944]

        scaler.scale_ = [9.22591618, 3.25402551, 9.21985451, 0.3748249, 11.97321657, 90.27301931, 31.58233882, 12.3875143 ,  0.2059728  ,17.16697477 ,72.66694131 , 0.44263929]

        X_cols = data.columns
        X_test = pd.DataFrame(scaler.transform(data), columns=X_cols)
        X_test['gender_enc'] = X_test['gender_enc'].astype('int')

        return X_test


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--seed', type=int, default=0)
    config = args.parse_args()

    time_init = time.time()

    np.random.seed(config.seed)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
    }
    modele = xgb.XGBClassifier()
    rf_clf = RandomForestClassifier()
    svc = SVC(probability=True)
    nb = GaussianNB()
    qda = QDA()
    dt = DT()
    knn = KNN()

    lda = LDA()
    modele = LGBMClassifier(**params)
    log_clf = LogisticRegression(C=0.1)
    gbc = GBC()
    ngb = NGBClassifier(Dist=k_categorical(2), verbose=False)
    ngb._estimator_type="classifier"

    cat = CatBoostClassifier(verbose=False)

    clf2 = SVC(probability=True)
    clf3 = xgb.XGBClassifier(use_label_encoder=False,  eval_metric='auc')
    # modele= VotingClassifier(estimators=[('cat',cat),('ngb',ngb),('rf_clf',rf_clf),('lda', lda),('clf1', clf1),('log_clf', log_clf), ('gbc', gbc),('clf2',clf2)], voting='soft',verbose=True)
    #model = TabNetClassifier(num_classes = 2,feature_columns=None,num_features=22)
    # nsml.bind() should be called before nsml.paused()
    def _collect_probas2(self,X):
        print("###############################")
        return np.asarray([ss.zscore(clf.predict_proba(X).astype(float)) for clf in self.estimators_])

    print(VotingClassifier._collect_probas)
    VotingClassifier._collect_probas = _collect_probas2
    print(VotingClassifier._collect_probas)

    bind_model(modele)

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    # test mode
    if config.pause:
        nsml.paused(scope=locals())

    # training mode
    if config.mode == 'train':
        data_path = DATASET_PATH + '/train/train_data'
        label_path = DATASET_PATH + '/train/train_label'

        raw_data = pd.read_csv(data_path)
        raw_labels = np.loadtxt(label_path, dtype=np.int16)
        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=0)
        
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        time_dl_init = time.time()
        print('Time to dataset initialization: ', time_dl_init - time_init)
        print("_____________")
        
        from sklearn.model_selection import GridSearchCV
        params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
                }

        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        
        folds = 3
        param_comb = 5

        skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
        random_search = RandomizedSearchCV(modele, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=1001)
        random_search.fit(X_train, y_train)
        print('최적 하이퍼 파라미터: ', random_search.best_params_)
        print('최고 예측 정확도: {:.4f}'.format(random_search.best_score_))
        print("_____________")
        
        print(roc_auc_score(y_val, modele.predict_proba(X_val)[:,1]))
        #print( model.predict_proba(X_val)[:,1])
        # models = [xgb_clf, log_clf, rf_clf, svc, gbc, nb, lda, qda, dt, knn,clf1,ngb,cat]
        # for model in models:
        #     model.fit(X_train, y_train)
        #     print(model, roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
        # print(xgb_clf.feature_importances_)
        #ann.fit(X_train, y_train, batch_size = 256, epochs = 200)
        #print(model, roc_auc_score(y_val, ann.predict(X_val)))
        nsml.save(0) # name of checkpoint; 'model_lgb.pkl' will be saved

        final_time = time.time()
        print("Time to dataset initialization: ", time_dl_init - time_init)
        print("Time spent on training :",final_time - time_dl_init)
        print("Total time: ", final_time - time_init)
        
        print("Done")