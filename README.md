import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
       
data=pd.read_csv("C:\\Users\\sandeep\\Desktop\\heart.csv"     
        
data.info()

data.head(10)

for col in data.columns:
    print(f'{col} has {data[col].nunique()} unique values.')

cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for col in cat_cols:
    print(f'The unique values in {col} are: {data[col].unique()}')

data.isnull().sum()

corr = data.corr(method = 'spearman')
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('Spearman Correlation Heatmap')
plt.show()

pairplot_cols = num_cols
pairplot_cols.append('HeartDisease')
figure = plt.figure(figsize=(15,8))
sns.pairplot(data[pairplot_cols], hue='HeartDisease', palette='GnBu')
plt.show()

data.loc[data['Cholesterol'] == 0, 'Cholesterol'].count()

data.loc[(data['Cholesterol'] == 0) & (data['HeartDisease'] == 1), 'Cholesterol'].count()

data.drop('Cholesterol', axis=1, inplace=True)
num_cols.remove('Cholesterol')

fig, axes = plt.subplots(2, 3, figsize=(20,12))
for i, col in zip(range(6), cat_cols):
    sns.stripplot(ax=axes[i//3][i%3], x=col, y='Age', data=data, palette='GnBu', hue='HeartDisease', jitter=True)
    axes[i//3][i%3].set_title(f'{col} Countplot')
    
eda_num_cols = ['RestingBP', 'MaxHR', 'Oldpeak']

fig, axes = plt.subplots(1, 3, figsize=(20,7))
for i, col in zip(range(3), eda_num_cols):
    sns.scatterplot(ax=axes[i], x='Age', y=col, hue="HeartDisease", style="Sex", data=data.iloc[0:889,:], palette="GnBu")
    
num_cols.remove('HeartDisease')

print(cat_cols)
print(num_cols)

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
scaler = MinMaxScaler()
onehotencoder = OneHotEncoder()
ordinalencoder = OrdinalEncoder()
preprocessor = ColumnTransformer(
    transformers = [
        ('onehotcat', onehotencoder, ['ChestPainType', 'ST_Slope', 'RestingECG']),
        ('labelcat', ordinalencoder, ['Sex', 'ExerciseAngina']),
        ('num', scaler, num_cols),
    ]
)

from sklearn.model_selection import train_test_split
X = data.iloc[:,:10]
y = data['HeartDisease']
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectPercentile, chi2

def fit(clf, params, preprocessor=preprocessor, X_train=X_train, y_train=y_train):
    params['selector__percentile'] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    params['selector__score_func'] = [chi2]
    selector = SelectPercentile()
    pipeline = Pipeline([( 'preprocessor', preprocessor),
                        ('selector', selector),
                       ('lr', clf)])
    grid = GridSearchCV(pipeline, params, cv=KFold(n_splits=10), return_train_score=True, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid

def make_predictions(model, X_test=X_test):
    return model.predict(X_test)

def best_scores(model):
    print(f'The mean cross validation test score is: {model.cv_results_.keys()}')
    print(f'The best parameters are: {model.best_params_}')
    print(f'The best score that we got is: {model.best_score_}')
    return None

def plot_confusion_matrix(y_pred):
    print('00: True Negatives\n01: False Positives\n10: False Negatives\n11: True Positives\n')
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap='GnBu', alpha=0.75)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large') 
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('Actuals', fontsize=14)
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()
    return None

def check_scores(y_pred):
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred))
    return None
    
lr_params = {'lr__C':[0.001,.009,0.01,.09,1,5,10,25], 'lr__penalty':['l1', 'l2']} #lasso and ridge regression
lr_clf = LogisticRegression(solver='saga', max_iter=5000)
lr_model = fit(lr_clf, lr_params)

best_scores(lr_model)

lr_y_pred = make_predictions(lr_model)
check_scores(lr_y_pred)

plot_confusion_matrix(lr_y_pred)
