import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('C:/Users/HP/OneDrive - 가천대학교/바탕 화면/논문/data.csv')     
print("NaN values in dataset:")
print(dataset.isnull().sum())

rcParams['figure.figsize'] = 20, 20

numeric_rows = dataset.select_dtypes(include=[np.number])
plt.matshow(numeric_rows.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns, rotation=90)
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()

dataset.hist(bins=30, figsize=(20, 15))
plt.suptitle('Histogram of Each Feature')
plt.show()

rcParams['figure.figsize'] = 8, 6
plt.bar(dataset['death01'].unique(), dataset['death01'].value_counts(), color=['red', 'green', 'blue'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of Each Target Class')
plt.show()

dataset = pd.get_dummies(dataset, drop_first=True)

imputer = SimpleImputer(strategy='mean')
dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

standardScaler = StandardScaler()
columns_to_scale = dataset.columns.drop('death01')
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

y = dataset['death01']
X = dataset.drop(['death01'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

knn_scores = []
for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

plt.plot(range(1, 21), knn_scores, color='red')
for i in range(1, 21):
    plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1], 2)))
plt.xticks(range(1, 21))
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier Scores for Different K Values')
plt.show()

svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svc_classifier = SVC(kernel=kernel)
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))

plt.bar(kernels, svc_scores, color=['red', 'green', 'blue', 'cyan'])
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], round(svc_scores[i], 2))
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier Scores for Different Kernels')
plt.show()

dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features=i, random_state=0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))

plt.plot(range(1, len(X.columns) + 1), dt_scores, color='green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, round(dt_scores[i-1], 2)))
plt.xticks(range(1, len(X.columns) + 1))
plt.xlabel('Max Features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier Scores for Different Number of Maximum Features')
plt.show()

rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for n in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=n, random_state=0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

plt.bar(range(len(estimators)), rf_scores, color=['red', 'green', 'blue', 'cyan', 'magenta'], width=0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], round(rf_scores[i], 2))
plt.xticks(ticks=range(len(estimators)), labels=[str(estimator) for estimator in estimators])
plt.xlabel('Number of Estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier Scores for Different Number of Estimators')
plt.show()

models = [
    KNeighborsClassifier(n_neighbors=12),
    SVC(kernel='linear', probability=True),
    DecisionTreeClassifier(max_features=3, random_state=0),
    RandomForestClassifier(n_estimators=100, random_state=0)
]
model_names = ['KNN', 'SVC', 'Decision Tree', 'Random Forest']

plt.figure(figsize=(10, 8))

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC curve (area = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=10, alpha=0.5, color='g')
    plt.title('Histogram of Predicted Probabilities - ' + name)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.show()

accuracy_scores = []
f1_scores = []

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    print(f"For {name}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(model_names, accuracy_scores, marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(model_names, f1_scores, marker='o', linestyle='-', color='r', label='F1 Score')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Comparing Accuracy and F1 Scores of Models')
plt.legend()
plt.show()
