import argparse
import os
import time
import re
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb ## XGBoost 불러오기
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import nsml
from nsml import DATASET_PATH
from tabnet import TabNet, TabNetClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf

import torch
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

        # NaN 값 0으로 채우기
        #data = data.fillna(0)
        data = data.fillna(data.mean())
        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 컬럼명에 특수 JSON 문자 포함시 발생하는 오류 방지
        data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E','GGT','DBP','AST','Ht','BUN','SBP']
        X = data.drop(columns=DROP_COLS).copy()
        y = label
        print(X.columns)

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y, 
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )
        
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(contamination='auto')
        y_pred_train = lof.fit_predict(X_train)
  
        #Remove outliers where 1 represent inliers and -1 represent outliers:
        X_train = X_train[np.where(y_pred_train == 1, True, False)]
        y_train = y_train[np.where(y_pred_train == 1, True, False)]
        smote = SMOTE()
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

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E','GGT','DBP','AST','Ht','BUN','SBP']
        data = data.drop(columns=DROP_COLS).copy()
        # Training set mean, var
        scaler = StandardScaler()
        scaler.mean_= [ 51.05994488 , 69.64180974 , 25.11283041 , 65.58977587,   5.71682213,
        99.72002756, 201.78796469, 147.66602901, 128.09603557 , 52.39126243,
        4.39564935 ,  0.80891229 , 97.93576766 , 28.7212754   ,98.94400673,
        0.25930179]

        scaler.scale_ = [ 9.00147176, 11.61695691 , 3.00969656,  9.5075606   ,0.3655724 , 12.0663808,
        35.19931509, 87.07020583 ,30.57076698, 12.60658857 , 0.21775347 , 0.16284464,
        19.78686617, 15.29547106 ,70.55417441 , 0.43825149]

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
    import xgboost
    xgb_clf = xgboost.XGBClassifier()

    from sklearn.linear_model import LogisticRegression
    log_clf = LogisticRegression()

    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier()


    svc = SVC(probability=True)

    from sklearn.ensemble import GradientBoostingClassifier as GBC
    gbc = GBC()

    from sklearn.naive_bayes import GaussianNB 
    nb = GaussianNB()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA()

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    qda = QDA()

    from sklearn.tree import DecisionTreeClassifier as DT
    dt = DT()

    from sklearn.neighbors import KNeighborsClassifier as KNN

    knn = KNN()

    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))
    ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    ann.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy')



    clf1 = LGBMClassifier(**params)
    clf2 = SVC( probability=True)
    clf3 = xgb.XGBClassifier(use_label_encoder=False,  eval_metric='auc')
    model= VotingClassifier(estimators=[('lda', lda),('clf1', clf1),('log_clf', log_clf), ('gbc', gbc)], voting='soft',verbose=True)
    #model = TabNetClassifier(num_classes = 2,feature_columns=None,num_features=22)
    # nsml.bind() should be called before nsml.paused()
    bind_model(model)

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
        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.01, seed=0)
        
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        time_dl_init = time.time()
        print('Time to dataset initialization: ', time_dl_init - time_init)
        print("_____________")
        model.fit(X_train, y_train)
        print("_____________")
        


        #print(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
        #print( model.predict_proba(X_val)[:,1])
        models = [xgb_clf, log_clf, rf_clf, svc, gbc, nb, lda, qda, dt, knn,clf1]
        for model in models:
            model.fit(X_train, y_train)
            print(model, roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
        print(xgb_clf.feature_importances_)
        #ann.fit(X_train, y_train, batch_size = 256, epochs = 200)
        #print(model, roc_auc_score(y_val, ann.predict(X_val)))
        nsml.save(0) # name of checkpoint; 'model_lgb.pkl' will be saved

        final_time = time.time()
        print("Time to dataset initialization: ", time_dl_init - time_init)
        print("Time spent on training :",final_time - time_dl_init)
        print("Total time: ", final_time - time_init)
        
        print("Done")