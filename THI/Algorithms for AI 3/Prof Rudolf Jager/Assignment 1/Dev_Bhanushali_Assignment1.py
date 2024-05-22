# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split

# %% [markdown]
# # Original Data

# %%
data = pd.read_csv("./Wine_Test_02.csv")
data.head()

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# check for NA values

# %%
data.isnull().sum()

# %% [markdown]
# ## **A)** Plot histogram of each attribute regarding Y=0, Y=1 and Y=2, and display the number of samples (Y) for each quality classes.

# %% [markdown]
# segregation of dataframe rows as per the categorical classes of the quality feature

# %%
class_0 = data[(data["quality"] == 0)]
class_1 = data[(data["quality"] == 1)]
class_2 = data[(data["quality"] == 2)]

quality_data = {"Poor Quality": class_0, "Medium Quality": class_1, "Premium Quality": class_2}

# %% [markdown]
# #### Quality-Wise feature distribution

# %%
plt.figure(figsize=(30, 15))
plt.title("Quality-Wise Feature Distribution", pad=45, fontdict={"size":20})
for quality in quality_data.keys():
    cols = quality_data[quality].columns[:-1]
    class_data = quality_data[quality]

    plt.ylabel("Frequency")
    plt.axis("off")

    for i in range(4):
        for j in range(3):
            idx = (i*3) + j
            col_name = cols[idx - 1]
            if(idx + 1 == 12):
                plt.subplot(3,4,idx + 1), plt.hist(class_data[cols[idx - 1]], density=True, alpha=0), plt.title(col_name)
            else:
                plt.subplot(3,4,idx + 1), plt.hist(class_data[cols[idx]], alpha=0.7, histtype="step", linewidth=2, label=str(quality)), plt.title(col_name)
                plt.legend(["Poor Quality", "Medium Quality", "Premium Quality"])

# %% [markdown]
# #### Class frequency plot

# %%
class_sample_distribution = {}
for key in quality_data.keys():
    class_sample_distribution[key] = len(quality_data[key])

# %%
plt.figure(figsize=(4.5,3))
plt.title("Class frequency plot")
plt.xlabel("Class"), plt.ylabel("Frequency")
plt.bar(class_sample_distribution.keys(), class_sample_distribution.values(), width=0.4)

# %% [markdown]
# #### Answering Questions

# %% [markdown]
# **What can you say regarding the quality (Y) classes distribution?**
# 
# It is clear that the classes are unequally distributed. There is a large number of medium quality wine samples and a very low number of samples of the poor and premium quality wine.

# %% [markdown]
# **What is your conclusion regarding the expected performance of the classifier?**
# 
# Due to the biased nature of class sample frequencies, it is expected that the model will generalize poorly on real life data. The model may have high accuracy, in this case by classifiying the majority class correctly but has low classification accuracy for the poor and premium class when seen indivisually.

# %% [markdown]
# ## **B)** Perform one run of modeling and test. Compare the obtained test accuracy by using:
# 1. One versus All Classifier
# And
# 2. One versus One Classifier

# %% [markdown]
# generate dataset for train and test using stratified sampling to keep same ratios of target classes in train and test

# %%
X, y = data[data.columns[:-1]], data[data.columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3)

# %% [markdown]
# feature scaling

# %%
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# %% [markdown]
# #### **One VS One Classification**

# %%
ovo = OneVsOneClassifier(
    LogisticRegression(max_iter=500, penalty='l2')
).fit(x_train, y_train)

y_pred = ovo.predict(x_test)

# %% [markdown]
# **confusion matrix**

# %%
cf_martix_df = pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_test), index = ["Poor", "Medium", "Premium"],
                  columns = ["Poor", "Medium", "Premium"])
plt.figure(figsize = (5,3))
heatmap = sns.heatmap(cf_martix_df, annot=True, fmt="g")
heatmap.set_xlabel("Predicted Class")
heatmap.set_ylabel("Actual Class")
plt.plot()

# %% [markdown]
# here we can see that all predictions for the poor class were incorrect and more than half the predictions of premium class were also incorrect which is not desirable, even though we have a high accuracy (below) supported majorly by the medium quality class due to heavy class imbalance (Model can classify the medium quality class correctly often due to the high number of samples)

# %% [markdown]
# **classification report**

# %%
print(classification_report(y_pred=y_pred, y_true=y_test, zero_division=True))

# %% [markdown]
# #### **One VS Rest Classification**

# %%
ovo = OneVsRestClassifier(
    LogisticRegression(penalty='l2')
).fit(x_train, y_train)

y_pred = ovo.predict(x_test)

# %% [markdown]
# **confusion matrix**

# %%
cf_martix_df = pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_test), index = ["Poor", "Medium", "Premium"],
                  columns = ["Poor", "Medium", "Premium"])
plt.figure(figsize = (5,3))
heatmap = sns.heatmap(cf_martix_df, annot=True, fmt="g")
heatmap.set_xlabel("Predicted Class")
heatmap.set_ylabel("Actual Class")
plt.plot()

# %% [markdown]
# here we can see that all predictions for the poor class were incorrect and more than half the predictions of premium class were also incorrect which is not desirable, even though we have a high accuracy supported majorly by the medium class due to heavy class imbalance

# %% [markdown]
# **classification report**

# %%
print(classification_report(y_pred=y_pred, y_true=y_test, zero_division=True))

# %% [markdown]
# # New Data

# %%
data_new = pd.read_csv("./Wine_Test_02_6_8_red.csv")
data_new.head()

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# check for NULL values

# %%
data_new.isnull().sum()

# %% [markdown]
# ## **A)** Plot histogram of each attribute regarding Y=0, Y=1 and Y=2, and display the number of samples (Y) for each quality classes.

# %% [markdown]
# segregation of dataframe rows as per the categorical classes of the quality feature

# %%
class_0 = data_new[(data_new["quality"] == 0)]
class_1 = data_new[(data_new["quality"] == 1)]
class_2 = data_new[(data_new["quality"] == 2)]

quality_data = {"Poor Quality": class_0, "Medium Quality": class_1, "Premium Quality": class_2}

# %% [markdown]
# #### Quality-Wise feature distribution

# %%
plt.figure(figsize=(30, 15))
plt.title("Quality-Wise Feature Distribution", pad=45, fontdict={"size":20})
for quality in quality_data.keys():
    cols = quality_data[quality].columns[:-1]
    class_data = quality_data[quality]

    plt.ylabel("Frequency")
    plt.axis("off")

    for i in range(4):
        for j in range(3):
            idx = (i*3) + j
            col_name = cols[idx - 1]
            if(idx + 1 == 12):
                plt.subplot(3,4,idx + 1), plt.hist(class_data[cols[idx - 1]], density=True, alpha=0), plt.title(col_name)
            else:
                plt.subplot(3,4,idx + 1), plt.hist(class_data[cols[idx]], alpha=0.7, histtype="step", linewidth=2, label=str(quality)), plt.title(col_name)
                plt.legend(["Poor Quality", "Medium Quality", "Premium Quality"])
    

# %% [markdown]
# #### Class frequency plot

# %%
class_sample_distribution = {}
for key in quality_data.keys():
    class_sample_distribution[key] = len(quality_data[key])

# %%
class_sample_distribution

# %%
plt.figure(figsize=(4.5,3))
plt.title("Class frequency plot")
plt.xlabel("Class"), plt.ylabel("Frequency")
plt.bar(class_sample_distribution.keys(), class_sample_distribution.values(), width=0.4)

# %% [markdown]
# #### Answering Questions

# %% [markdown]
# **What can you say regarding the quality (Y) classes distribution?**
# 
# the class distribution is still imbalanced but is somewhat better than the older dataset since there is lesser difference between the sample count of each class unlike the previous dataset

# %% [markdown]
# **What is your conclusion regarding the expected performance of the classifier?**
# 
# Due to the biased nature of class samples, it is expected that the model will generalize poorly on real life data. The model may have high accuracy, in this case by classifiying the majority class correctly but will end up having low classification accuracy for the poor and premium class when seen indivisually.
# 
# It is probable that this dataset will lead to better results compared to the previous dataset due to better sampling ratios but because the samples overall are lesser than the previous dataset, worse performance can also be expected.

# %% [markdown]
# ## **B)** Perform one run of modeling and test. Compare the obtained test accuracy by using:
# 1. One versus All Classifier
# And
# 2. One versus One Classifier

# %% [markdown]
# generate dataset for train and test using stratified sampling to keep same ratios of classes in train and test

# %%
X_new, y_new = data_new[data_new.columns[:-1]], data_new[data_new.columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, stratify=y_new, random_state=3)

# %%
(data_new[data_new.columns[-1]] == y_new).value_counts()
y_new.value_counts()

# %% [markdown]
# feature scaling

# %%
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# %% [markdown]
# #### **One VS One Classification**

# %%
ovo = OneVsOneClassifier(
    LogisticRegression(max_iter=500, penalty='l2')
).fit(x_train, y_train)

y_pred = ovo.predict(x_test)

# %% [markdown]
# **confusion matrix**

# %%
cf_martix_df = pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_test), index = ["Poor", "Medium", "Premium"],
                  columns = ["Poor", "Medium", "Premium"])
plt.figure(figsize = (5,3))
heatmap = sns.heatmap(cf_martix_df, annot=True, fmt="g")
heatmap.set_xlabel("Predicted Class")
heatmap.set_ylabel("Actual Class")
plt.plot()

# %% [markdown]
# here we can see that all predictions for the poor class were incorrect and more than half the predictions of premium class were also incorrect which is not desirable.

# %% [markdown]
# **classification report**

# %%
print(classification_report(y_pred=y_pred, y_true=y_test, zero_division=True))

# %% [markdown]
# #### **One VS Rest Classification**

# %%
ovo = OneVsRestClassifier(
    LogisticRegression(penalty='l2')
).fit(x_train, y_train)

y_pred = ovo.predict(x_test)

# %% [markdown]
# **confusion matrix**

# %%
cf_martix_df = pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_test), index = ["Poor", "Medium", "Premium"],
                  columns = ["Poor", "Medium", "Premium"])
plt.figure(figsize = (5,3))
heatmap = sns.heatmap(cf_martix_df, annot=True, fmt="g")
heatmap.set_xlabel("Predicted Class")
heatmap.set_ylabel("Actual Class")
plt.plot()

# %% [markdown]
# here we can see that all predictions for the poor class were incorrect and more close to half the predictions of premium class were also incorrect which is not desirable.

# %% [markdown]
# **classification report**

# %%
print(classification_report(y_pred=y_pred, y_true=y_test, zero_division=True))

# %% [markdown]
# # Observations

# %% [markdown]
# It can be seen that the accuracy decreases in the second dataset which is reasonable due to the lower number of samples across classes, but the model also learns slightly better for minority classes in the second dataset since the sampling ratios across class are not that varying and there are lesser true negatives for the predictions for the premium class.

# %% [markdown]
# It can be concluded that with better class distribution, it is likely that the performance will increase in contrast to when the class distributions are highly biased


