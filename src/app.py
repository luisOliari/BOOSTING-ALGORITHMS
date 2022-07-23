from utils import db_connect
engine = db_connect()

# your code here

# Step 0. Load libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import seaborn as sns
import sklearn
import pickle
import xgboost as xgb

from xgboost import XGBRegressor
from xgboost import XGBModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier  # random forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Step 1. Load the dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, index_col=0)

# Get basic info
df.info()

# descripción del dataset:
df.describe()

# elimino la columna Cabin
df=df.drop(columns='Cabin')

# imputamos en los faltantes de la edad con la meida y la moda en embarked:
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# correlaciones entre numéricas
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')

# 1.3 Transform if needed
# esa transformación afecta a todo el dataset (antes de dividirlo)
X=df.drop(columns=['Ticket','Name','Survived'])
y=df['Survived']

X[['Sex','Embarked']]=X[['Sex','Embarked']].astype('category')

X['Sex']=X['Sex'].cat.codes

X['Embarked']=X['Embarked'].cat.codes

# 2.1 Split the dataset so to avoid bias
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1107)

# se convierte a categórica
X[['Sex','Embarked']]=X[['Sex','Embarked']].astype('category')

# Boosting Algorithms:

# Fit a Gradient Boosting model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

plt.show()

D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

param = {
    'eta': 0.3,
    'max_depth': 3,
    'objective': 'multi:softprob',
    'num_class': 3}

steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)

clf1= xgb.XGBClassifier()
parameters = {
    "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

grid = GridSearchCV(clf1,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, y_train)

# score con regularizaciones
clf1.fit(X_train, y_train)
y_pred_1 = clf1.predict(X_test)
accuracy_score(y_test, y_pred_1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_1, labels=clf1.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=clf1.classes_)
disp.plot()
































