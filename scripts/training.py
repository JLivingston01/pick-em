import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt

from joblib import dump

def main() -> None:
    data = pd.read_csv("data/model_data.csv")

    model_data = data[data['season']<2023].copy()

    exclusions = ['week','away_team','home_team','away_score','home_score','result','season',]

    first_features = [
        i for i in model_data.columns if 
        (i not in exclusions)&
        ('_team_id_' not in i)
    ]
    target = 'result'

    Xtrain,Xtest,ytrain,ytest=train_test_split(model_data[first_features],model_data[target],test_size=.2,random_state=42,shuffle=True)

    corr = pd.concat([Xtrain,ytrain],axis=1).corr(numeric_only=True)[target].sort_values()
    corr.sort_values(ascending=False)
    feats_used = corr[(abs(corr)>.05)&(abs(corr)<1)]

    features = [i for i in feats_used.index if i != target]
    Xtrain,Xtest,ytrain,ytest=train_test_split(model_data[features],model_data[target],test_size=.2,random_state=42,shuffle=True)

    model = Pipeline(steps=[
        ('scaler',MinMaxScaler()),
        ('learner',MLPRegressor(random_state=42,hidden_layer_sizes=(100),activation='logistic',
                                alpha=0,batch_size=1,solver='sgd',learning_rate_init=.0001,
                                tol=1e-8,
                                n_iter_no_change=100,
                        max_iter=4000,learning_rate='constant',shuffle=True,verbose=True,
                        early_stopping=True,
                        validation_fraction=.1))
    ]).fit(Xtrain,ytrain)

    yfit = pd.Series(model.predict(Xtrain),Xtrain.index)
    ypred = pd.Series(model.predict(Xtest),Xtest.index)


    plt.scatter(ytrain,yfit)
    plt.scatter(ytest,ypred)
    plt.title(str(r2_score(ytest,ypred)))
    plt.savefig("artifacts/validation/validation.jpg")

    dump(model,"artifacts/model.joblib")

    return

if __name__=='__main__':
    main()
