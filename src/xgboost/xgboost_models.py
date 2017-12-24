import matplotlib.pylab as plt
import pandas as pd
import xgboost as xgb
from matplotlib.pylab import rcParams
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier

rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('train_modified.csv')
test = pd.read_csv('test_modified.csv')

target = 'Disbursed'
IDcol = 'ID'

train['Disbursed'].value_counts()
predictors = [x for x in train.columns if x not in [target, IDcol]]


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


def get_basic_result():
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb1, train, test, predictors)


def get_first_grid_search():
    param_test1 = {
        'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=128, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(train[predictors], train[target])
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


if __name__ == "__main__":
    get_basic_result()
    get_first_grid_search()
