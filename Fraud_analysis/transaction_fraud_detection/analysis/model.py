import pickle
import pandas as pd
import numpy as np

import gc
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    return pd.read_csv(file_path, index_col=0)

def scale_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_val_scaled, columns=X_val.columns)


def train_model(X, y, params, NFOLDS=3, proporcao_treino=0.8):
    folds = KFold(n_splits=NFOLDS)
    columns = X.columns
    splits = folds.split(X, y)

    y_oof = np.zeros(X.shape[0])
    score = 0

    feature_importances = pd.DataFrame()
    feature_importances['feature'] = columns

    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        clf = lgb.train(params, dtrain, 10000, valid_sets=[dtrain, dvalid])

        feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

        y_pred_valid = clf.predict(X_valid)
        y_oof[valid_index] = y_pred_valid
        print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

        score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    print(f"\nMean AUC = {score}")
    print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

    return clf, feature_importances

def predict_and_evaluate(model, data, threshold=0.7):
    X = data.drop(['isFraud'], axis=1)
    y = data['isFraud']

    prediction = model.predict(X)
    print("Predictions:", prediction)

    if len(prediction) == len(X):
        data['score'] = prediction
        data['fraud'] = data['score'].apply(lambda x: 1 if x > threshold else 0)

    return data

def save_model(model, filename='model/model.pickle'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")

def main():
    # Carregar dados
    df_train = load_data('dataframe_to_train_model.csv')
    df_train = df_train[['isFraud','card1', 'TransactionDT', 'addr1', 'TransactionAmt', 'D10','V310', 'card2', 'P_emaildomain',
                        'D15', 'D1', 'V312', 'V315', 'V318', 'V285', 'card4', 'V314', 'card5', 'V313', 'V283',
                        'V130', 'V320', 'ProductCD']] 


    # Separar dados de treino e teste
    proporcao_treino = 0.85
    treino, teste = train_test_split(df_train, test_size=1-proporcao_treino, random_state=42)

    # Treinar modelo com todos os dados
    X_train, X_val, y_train, y_val = train_test_split(treino.drop(['isFraud'], axis=1), treino['isFraud'], test_size=1-proporcao_treino, random_state=42)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

    params = {'num_leaves': 546,
              'min_child_weight': 0.03454472573214212,
              'feature_fraction': 0.1797454081646243,
              'bagging_fraction': 0.2181193142567742,
              'min_data_in_leaf': 106,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': 0.005883242363721497,
              "boosting_type": "gbdt",
              "bagging_seed": 11,
              "metric": 'auc',
              "verbosity": -1,
              'reg_alpha': 0.3299927210061127,
              'reg_lambda': 0.3885237330340494,
              'random_state': 42,
    }

    model, feature_importances = train_model(X_train_scaled, y_train, params)
    save_model(model)
    model = pickle.load(open('model/model.pickle', 'rb'))
    predicted_data = predict_and_evaluate(model, teste)
    print(predicted_data.head())

if __name__ == "__main__":
    main()
