import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('C:/Users/HP/OneDrive - 가천대학교/바탕 화면/논문/data.csv')
object_cols = df.select_dtypes(include = 'object').columns

label_encoder = LabelEncoder()
for col in object_cols:
	df[col] = label_encoder.fit_transform(df[col])

print(df.shape)
print(df.info())
print(df.isnull().sum())
df = df.dropna()

sns.pairplot(df, hue = 'death01')

corr = df.corr()
sns.heatmap(corr, cmap = 'coolwarm') 

df.hist(figsize = (15,20))

df['death01'].value_counts().plot(kind = 'bar')
plt.xlabel("TetraPlegia / ParaPlegia")
plt.ylabel("Count")
plt.title("Classification of Plegia")

cat = ['Patient', 'RNASeqCluster', 'MethylationCluster', 'miRNACluster', 'CNCluster', 'RPPACluster', 'OncosignCluster', 'COCCluster', 'histological_type', 'neoplasm_histologic_grade', 'tumor_tissue_site', 'laterality', 'tumor_location', 'gender', 'age_at_initial_pathologic', 'race', 'ethnicity', 'death01']
df[cat] = df[cat].apply(LabelEncoder().fit_transform)

X = df.drop('death01', axis = 1)
y = df['death01']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = QuantileTransformer()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors =4, metric = 'euclidean', p = 1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Class Names : ", list(df['death01'].unique()))

error_rate = []
for i in range(1, 40):
	knn = KNeighborsClassifier(n_neighbors = i)
	knn.fit(X_train, y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10, 6))
plt.plot(range(1, 40), error_rate, color = 'red', linestyle = 'dashed', marker = 'o', markerfacecolor = 'green', markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean', p = 2)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Class Names : ", list(df['death01'].unique()))

h2o.init()
hf = h2o.H2OFrame(df)
train, valid = hf.split_frame(ratios = [.80], seed = 1234)

featureColumns = train.columns
targetColumn = "death01"
featureColumns.remove(targetColumn)

aml = H2OAutoML(max_models = 12, seed = 1234, balance_classes = True)
aml.train(x = featureColumns, y = targetColumn, training_frame = train, validation_frame = valid)

lb = aml.leaderboard
print(lb.head(rows = lb.nrows))

model = aml.leader
predicted_y = model.predict(valid[featureColumns])
valid_dataset = valid.as_data_frame()

predicted_data = model.predict(valid[featureColumns]).as_data_frame()

acc = metrics.accuracy_score(valid_dataset[targetColumn], np.round(abs(predicted_data['predict'])))
classReport = metrics.classification_report(valid_dataset[targetColumn], np.round(abs(predicted_data['predict'])))
confMatrix = metrics.confusion_matrix(valid_dataset[targetColumn], np.round(abs(predicted_data['predict'])))

print('Testing Results of the trained model: ')
print('Accuracy : ', acc)
print('Confusion Matrix :\n', confMatrix)
print('Classification Report :\n',classReport)

from sklearn.metrics import roc_curve, auc

models = [
    LogisticRegression(random_state=0),
    RandomForestClassifier(random_state=0),
    XGBClassifier(use_label_encoder=False, eval_metric='error'),
    svm.SVC(probability=True)
]
	
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM']

plt.figure(figsize=(10, 8))

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

models = [
    KNeighborsClassifier(n_neighbors = 12),
    SVC(kernel = 'linear', probability=True),
    DecisionTreeClassifier(max_features = 3, random_state = 0),
    RandomForestClassifier(n_estimators = 100, random_state = 0)
]

model_names = ['KNN', 'SVC', 'Decision Tree', 'Random Forest']

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (name, roc_auc))
    
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of Classification')
plt.ylabel('Frequency')
plt.legend(loc='upper center')
plt.show()