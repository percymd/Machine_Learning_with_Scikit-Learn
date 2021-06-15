import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=='__main__':

    dt_heard = pd.read_csv('./datasets/heart.csv')
    print(dt_heard['target'].describe())

    X = dt_heard.drop(['target'], axis=1)
    y = dt_heard['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print('='*64)
    print(accuracy_score(boost_pred, y_test))