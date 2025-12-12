import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv('/Users/formalraindbow43/python_for_andan/Homeworks/data/data.adult.csv')


df = df.replace('?', pd.NA).dropna()


y = (df['>50K,<=50K'] == '>50K').astype(int)
df = df.drop('>50K,<=50K', axis=1)


X = pd.get_dummies(df, drop_first=True)


model = GradientBoostingClassifier(n_estimators=67, max_depth=5, 
                                   max_features='sqrt', random_state=42)
model.fit(X, y)


model_data = {
    'model': model,
    'columns': X.columns.tolist()
}

joblib.dump(model_data, 'model_income.pkl')
