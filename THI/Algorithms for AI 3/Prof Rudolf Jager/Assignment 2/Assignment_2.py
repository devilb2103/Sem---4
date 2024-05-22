# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize

# %%
data = pd.read_csv("../Assignment 1/Wine_Test_02_6_8_red.csv")
data.head()

# %% [markdown]
# ## Data preprocessing

# %%
data.isnull().sum()

# %% [markdown]
# we can see that there is no null values

# %%
data["quality"].value_counts()

# %% [markdown]
# here we notice that there is class imbalance

# %% [markdown]
# Training with Oversampling minority classes using SMTE algorithm since class imbalance will heavily impact knn performance

# %%
def overSample(data: pd.DataFrame):

    X = data.drop('quality', axis=1)
    y = data['quality']

    # training without randomstate to get random samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    return X_res, X_test, y_res, y_test

# %%
len(list(range(3, 70, 2)))

# %% [markdown]
# ## Method 1

# %%
accuracy = []

for i in range(5):
    iter_accuracy = []
    for j in range(3, 70, 2):
        X_res, X_test, y_res, y_test = overSample(data)
        model = KNeighborsClassifier(n_neighbors=j, metric="euclidean", algorithm="ball_tree").fit(X_res, y_res)
        acc = mean_absolute_error(model.predict(X_test), y_test)
        iter_accuracy.append(acc)
    accuracy.append(iter_accuracy)

accuracy = np.array(accuracy)

# %%
avg_accuracy = np.zeros((1, accuracy.shape[1]))
avg_accuracy = np.average(accuracy, axis=0)

# %%
plt.plot(np.array(range(len(avg_accuracy))) + 3, avg_accuracy, label="average error across n neighbors for 5 iterations"),
plt.scatter(np.argmin(np.array(range(len(avg_accuracy)))) + 3, min(avg_accuracy), c="red", zorder=100, label=f"max accuracy = {np.round(min(avg_accuracy), 2)}% (n={np.argmin(np.array(range(len(avg_accuracy)))) + 3})")
plt.xlabel("n_neighbors"), plt.ylabel("mean absolute error")
plt.grid()
plt.legend()

# %%
n_value = np.argmin(np.array(range(len(avg_accuracy)))) + 3

accuracy = []

for i in range(5):
    X_res, X_test, y_res, y_test = overSample(data)
    model = KNeighborsClassifier(n_neighbors=n_value, metric="euclidean", algorithm="ball_tree").fit(X_res, y_res)
    acc = mean_absolute_error(model.predict(X_test), y_test)
    accuracy.append(iter_accuracy)

accuracy = np.array(accuracy)
print(f"average accuracy: {round(np.mean(accuracy), 2) * 100}%")

# %% [markdown]
# #### Plot ROC curves for One Iteration

# %%
X = data.drop('quality', axis=1)
y = label_binarize(data['quality'], classes=[0,1,2])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_value, metric="euclidean", algorithm="ball_tree")).fit(X_res, y_res)
y_score = model.predict_proba(X_test)

# %%
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# %%
# Plot the ROC curves
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
plt.title('ROC - AUC Plot')
plt.legend()
plt.show()


# %% [markdown]
# Higher the roc curve sticks to the top left of the plot, better the results. This is because we want the ratio between true positive to false positive rates to be as high as possible. In this case, we can assess the quality of training for indivisual classes. We see that for this iteration, class 2 has been understood by the model well, followed by class 1, and then class 0.

# %% [markdown]
# ## Method 2

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': range(3, 70, 2),
    'metric': ['euclidean', 'manhattan'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

accuracy = []

for i in range(5):
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=2, verbose=1)
    X_res, X_test, y_res, y_test = overSample(data)
    grid_search.fit(X_res, y_res)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    accuracy.append(grid_search.best_score_)


# %%
print(f"Mean Accuracy: {round(np.mean(accuracy), 2)*100}%")

# %% [markdown]
# ## Remarks

# %% [markdown]
# Mean accuracy of Method 1 = **70%**
# 
# Mean accuracy of Method 2 = **74%**
# 
# The difference in performance here is significant. This is because GridSearchCV automates the searching of the best hyperparameter combination which yields better results than hardcoding hyperparameters like it was done in method 1.


