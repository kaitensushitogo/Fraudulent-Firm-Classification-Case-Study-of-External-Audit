
# coding: utf-8

# # <font color='red'> Project 1
# 
# ### Project description:
# - Please read the Data Set Information section to learn about this dataset. 
# - Data description is also provided for thi dataset.
# - Read data into Jupyter notebook, use pandas to import data into a data frame
# - Preprocess data: Explore data, check for missing data and apply data scaling. Justify the type of scaling used.
# 
# ### Regression Task:
# - Apply all the regression models you've learned so far. If your model has a scaling parameter(s) use Grid Search to find the best scaling parameter. Use plots and graphs to help you get a better glimpse of the results. 
# - Then use cross validation to find average training and testing score. 
# - Your submission should have at least the following regression models: KNN repressor, linear regression, Ridge, Lasso, polynomial regression, SVM both simple and with kernels. 
# - Finally find the best regressor for this dataset and train your model on the entire dataset using the best parameters and predict buzz for the test_set.
# 
# ### Classification task:
# - Decide aboute a good evaluation strategy and justify your choice.
# - Find best parameters for following classification models: KNN classifcation, Logistic Regression, Linear Supprt Vector Machine, Kerenilzed Support Vector Machine, Decision Tree. 
# - Which model gives the best results?
# 
# ### Deliverables:
# - Submit IPython notebook. Use markdown to provide an inline comments for this project.
# - Submit only one notebook. Before submitting, make sure everything runs as expected. To check that, restart the kernel (in the menubar, select Kernel > Restart) and then run all cells (in the menubar, select Cell > Run All).
# - Visualization encouraged. 
# 
# ### Questions regarding project:
# - Post your queries related to project on discussion board on e-learning. There is high possibility that your classmate has also faced the same problem and knows the solution. This is an effort to encourage collaborative learning and also making all the information available to everyone. We will also answer queries there. We will not be answering any project related queries through mail.

# ---
# ### Data Set Information:
# This dataset is taken from a research explained here. 
# 
# The goal of the research is to help the auditors by building a classification model that can predict the fraudulent firm on the basis the present and historical risk factors. The information about the sectors and the counts of firms are listed respectively as Irrigation (114), Public Health (77), Buildings and Roads (82), Forest (70), Corporate (47), Animal Husbandry (95), Communication (1), Electrical (4), Land (5), Science and Technology (3), Tourism (1), Fisheries (41), Industries (37), Agriculture (200).
# 
# There are two csv files to present data. Please merge these two datasets into one dataframe. All the steps should be done in Python. Please don't make any changes in csv files. Consider ``Audit_Risk`` as target columns for regression tasks, and ``Risk`` as the target column for classification tasks. 
# 
# ### Attribute Information:
# Many risk factors are examined from various areas like past records of audit office, audit-paras, environmental conditions reports, firm reputation summary, on-going issues report, profit-value records, loss-value records, follow-up reports etc. After in-depth interview with the auditors, important risk factors are evaluated and their probability of existence is calculated from the present and past records.
# 
# 
# ### Relevant Papers:
# Hooda, Nishtha, Seema Bawa, and Prashant Singh Rana. 'Fraudulent Firm Classification: A Case Study of an External Audit.' Applied Artificial Intelligence 32.1 (2018): 48-64.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading datasets
audit_risk = pd.read_csv('audit_risk.csv')
trial = pd.read_csv('trial.csv')


# In[3]:


# Exploring datasets
print(audit_risk.shape)
print(trial.shape)

audit_risk.info()


# In[4]:


audit_risk.describe()


# In[5]:


trial.info()


# In[6]:


trial.describe()


# In[7]:


# Remove categorical values in the column of LOCATION_ID
# Did not assign new values because they could be already one of the LOCATION_ID
audit_risk = audit_risk[audit_risk.LOCATION_ID != 'LOHARU']
audit_risk = audit_risk[audit_risk.LOCATION_ID != 'NUH']
audit_risk = audit_risk[audit_risk.LOCATION_ID != 'SAFIDON']

trial = trial[trial.LOCATION_ID != 'LOHARU']
trial = trial[trial.LOCATION_ID != 'NUH']
trial = trial[trial.LOCATION_ID != 'SAFIDON']


# In[8]:


# Changing dtype(object to integer)
audit_risk['LOCATION_ID'] = pd.to_numeric(audit_risk['LOCATION_ID'])
trial['LOCATION_ID'] = pd.to_numeric(trial['LOCATION_ID'])

# Removing duplicated columns
# We removed duplicated parameters with different scales in advance 
trial.drop(['Sector_score','LOCATION_ID','PARA_A','SCORE_A',
            'PARA_B','SCORE_B','TOTAL','numbers','District','Money_Value','History','Score','Risk'], axis=1, inplace = True)


# In[9]:


# Dealing with missing values
print(audit_risk.isnull().sum().sort_values(ascending = False))
audit_risk['Money_Value'] = audit_risk.fillna(audit_risk['Money_Value'].mean())


# In[10]:


trial.isnull().sum().sort_values(ascending = False)


# In[11]:


# Checking the size
print(audit_risk.shape)
print(trial.shape)


# In[12]:


# Merging datasets
df = pd.concat([audit_risk, trial], axis=1)
df.shape


# In[13]:


df.info()


# In[14]:


# Correlation within variables
plt.figure(figsize=(12,8))
sns.heatmap(df.drop(['Detection_Risk'], axis=1).corr(), cmap = 'coolwarm')


# In[15]:


# Correlation with target variables
cor = pd.DataFrame(df.drop("Audit_Risk", axis=1).apply(lambda x: x.corr(df.Audit_Risk)).sort_values(ascending = False)).rename(columns = {0:'Correlation'})
cor.dropna()


# # Regression Models Set Up, Spliting & Scaling Dataset

# In[16]:


# Environment setting for regression models
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler # Used Standard Scaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import classification_report


# In[17]:


regression_X = df.drop(['Audit_Risk','Risk'], axis=1)
regression_y = df['Audit_Risk']


# In[18]:


# Splitting training and test data
X_train_org, X_test_org, y_train, y_test = train_test_split(regression_X, regression_y, test_size = 0.3, random_state = 101)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# # Regression Model (1) - Linear Regression

# In[19]:


lreg = LinearRegression()
param_grid = {'fit_intercept':[True,False], 'normalize':[True,False]}

#cv =5
grid = GridSearchCV(lreg, param_grid, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[20]:


lreg_accuracy_train = grid.best_estimator_.score(X_test, y_test)
lreg_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Linear Regression - Train Accuracy: %.2f'%lreg_accuracy_train)
print('Linear Regression - Test Accuracy: %.2f '%lreg_accuracy_test)


# In[21]:


report_table = [['Linear', '', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# # Regression Model (2) - Ridge Regression

# In[22]:


param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# create and fit a ridge regression model, testing each alpha
# cv = 5
model = Ridge()
grid = GridSearchCV(model, param_grid, cv = 5)
grid.fit(X_train, y_train)
# summarize the results of the grid search
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[23]:


ridge_accuracy_train = grid.best_estimator_.score(X_train, y_train)
ridge_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Ridge Regression - Train Accuracy: %.2f'%ridge_accuracy_train)
print('Ridge Regression - Test Accuracy: %.2f '%ridge_accuracy_test)


# In[24]:


report_table = report_table + [['ridge', 'alpha = 1000', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# # Regression Model (3) - Lasso

# In[25]:


param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# create and fit a ridge regression model, testing each alpha
# cv = 5
model = Lasso()
grid = GridSearchCV(model, param_grid, cv = 5)
grid.fit(X_train, y_train)
# summarize the results of the grid search
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[26]:


lasso_accuracy_train = grid.best_estimator_.score(X_train, y_train)
lasso_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Lasso Regression - Train Accuracy: %.2f'%lasso_accuracy_train)
print('Lasso Regression - Test Accuracy: %.2f '%lasso_accuracy_test)


# In[27]:


report_table = report_table + [['lasso', 'alpha = 1.0', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# # Regression Model (4) - KNN Regressor

# In[28]:


knn = KNeighborsRegressor()
param_grid = {'n_neighbors':np.arange(1,11,1)}

#cv =5
grid = GridSearchCV(knn, param_grid, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[29]:


knnreg_accuracy_train = grid.best_estimator_.score(X_train, y_train)
knnreg_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('KNN Regression - Train Accuracy: %.2f'%knnreg_accuracy_train)
print('KNN Regression - Test Accuracy: %.2f '%knnreg_accuracy_test)


# In[30]:


report_table =report_table +  [['knn', 'n = 4', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# # Regression Model (5) - Polynomial Regression

# In[31]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

param_grid = {'polynomialfeatures__degree': np.arange(2), 
        'linearregression__fit_intercept': [True, False], 
        'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=3)

grid.fit(X_train, y_train)
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))


# In[32]:


poly_accuracy_train = grid.best_estimator_.score(X_train, y_train)
poly_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Polynomial Regression - Train Accuracy: %.2f'%poly_accuracy_train)
print('Polynomial Regression - Test Accuracy: %.2f '%poly_accuracy_test)


# In[33]:


report_table =report_table +  [['Polynomial', 'degree=1', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# # Regression Model (6) - Linear SVR

# In[34]:


model = LinearSVR()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10], 'epsilon':[0.001,0.01,0.1,1,10]}

#cv = 5
grid = GridSearchCV(model, parameters, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[35]:


lsvr_accuracy_train = grid.best_estimator_.score(X_train, y_train)
lsvr_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Linear SVR - Train Accuracy: %.2f'%lsvr_accuracy_train)
print('Linear SVR - Test Accuracy: %.2f '%lsvr_accuracy_test)


# In[36]:


report_table =report_table +  [['SVR', 'degree=1', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# # Regression Model (6) - rbf Kernel SVR

# In[37]:


model = SVR(kernel='rbf')
parameters = {'C':[0.001, 0.01, 0.1, 1,10],'gamma':[0.0001,0.001, 0.01, 0.1, 1],'epsilon':[0.01, 0.1, 1]}

#cv =5
grid = GridSearchCV(model, parameters, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[38]:


rbf_svr_accuracy_train = grid.best_estimator_.score(X_train, y_train)
rbf_svr_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('RBF SVR - Train Accuracy: %.2f'%rbf_svr_accuracy_train)
print('RBF SVR - Test Accuracy: %.2f '%rbf_svr_accuracy_test)


# In[39]:


report_table = report_table + [['rbf Kernel SVR', 'C=10', grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test)]]


# Show Report

# In[40]:


report = pd.DataFrame(report_table,columns = ['Model name', 'Model parameter', 'Train accuracy', 'Test accuracy'])
report.index = report['Model name']
Final_Report= report.drop(['Polynomial'])# as polynomial regression with the degree of 1 shows a negative test score and train score, we will just drop it.


# In[41]:


Final_Report


# In[42]:


sns.barplot(y =Final_Report.index, x = 'Test accuracy',data = Final_Report)


# # Classification Models Set Up and Spliting Dataset

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score,classification_report


# In[44]:


clf_X = df.drop(['Audit_Risk','Risk'], axis=1)
clf_y = df['Risk']


# In[45]:


# Splitting training and test data
X_train_org, X_test_org, y_train, y_test = train_test_split(clf_X, clf_y, test_size = 0.3, random_state = 101)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# # Classification Model (1) - KNN Classification

# In[46]:


knn = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(1,11,1)}

#cv =5
grid = GridSearchCV(knn, parameters, cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[47]:


knnclf_accuracy_train = grid.best_estimator_.score(X_train, y_train)
knnclf_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('KNN Classifier - Train Accuracy: %.4f'%knnclf_accuracy_train)
print('KNN Classifier - Test Accuracy: %.4f '%knnclf_accuracy_test)


# In[48]:


knnclf_auc_train = roc_auc_score(y_train, grid.predict(X_train))
knnclf_auc_test = roc_auc_score(y_test, grid.predict(X_test))

print('KNN Classifier - Train ROC AUC Score: %.4f'%knnclf_auc_train)
print('KNN Classifier - Test ROC AUC Score: %.4f '%knnclf_auc_test)


# In[49]:


pred = grid.predict(X_test)
print(classification_report(y_test, pred))


# In[74]:


Classification_table =[['KNN Classification', 'n=10', (knnclf_accuracy_train), (knnclf_accuracy_test)]]


# # Classification Model (2) - Logistic Regression

# In[51]:


model = LogisticRegression(random_state=101)
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

#cv =5
grid = GridSearchCV(model, parameters, cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[52]:


logistic_accuracy_train = grid.best_estimator_.score(X_train, y_train)
logistic_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Logistic Regression Classifier - Train Accuracy: %.4f'%logistic_accuracy_train)
print('Logistic Regression Classifier - Test Accuracy: %.4f '%logistic_accuracy_test)


# In[53]:


logistic_auc_train = roc_auc_score(y_train, grid.predict(X_train))
logistic_auc_test = roc_auc_score(y_test, grid.predict(X_test))

print('Logstic Regression - Train ROC AUC Score: %.4f'%logistic_auc_train)
print('Logistic Regression - Test ROC AUC Score: %.4f '%logistic_auc_test)


# In[54]:


pred = grid.predict(X_test)
print(classification_report(y_test, pred))


# In[75]:


Classification_table =Classification_table + [['Logistic Regression', 'C=10,penalty=l1', logistic_accuracy_train, logistic_accuracy_test]]


# # Classification Model (3) - Linear Support Vector Machine

# In[56]:


model = LinearSVC()
parameters = [{'C':[0.001, 0.01, 0.1, 1, 10, 100], 'penalty':['l2'],'loss':['hinge','squared_hinge']},
              {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'penalty':['l1'], 'dual':[False], 'loss':['squared_hinge']}]

#cv =5
grid = GridSearchCV(model, parameters, cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[57]:


lsvc_accuracy_train = grid.best_estimator_.score(X_train, y_train)
lsvc_accuracy_test = grid.best_estimator_.score(X_test, y_test) 

print('Linear SVC - Train Accuracy: %.4f'%lsvc_accuracy_train)
print('Linear SVC - Test Accuracy: %.4f '%lsvc_accuracy_test)


# In[58]:


lsvc_auc_train = roc_auc_score(y_train, grid.predict(X_train))
lsvc_auc_test = roc_auc_score(y_test, grid.predict(X_test))

print('Linear SVC - Train ROC AUC Score: %.4f'%lsvc_auc_train)
print('Linear SVC - Test ROC AUC Score: %.4f '%lsvc_auc_test)


# In[59]:


pred = grid.predict(X_test)
print(classification_report(y_test, pred))


# In[81]:


Classification_table  = Classification_table +[['Linear Support Vector Machine', 'C=10,penalty=l1', lsvc_accuracy_train, lsvc_accuracy_test]]


# # Classification Model (4) - rbf Kernelized Support Vector Machine

# In[61]:


model = SVC(kernel='rbf')
parameters = [{'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001,0.01,0.1,1,10,100]}]

#cv =5
grid = GridSearchCV(model, parameters, scoring = 'roc_auc',cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[62]:


rbf_svc_accuracy_train = grid.best_estimator_.score(X_train, y_train)
rbf_svc_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('RBF SVC - Train Accuracy: %.4f'%rbf_svc_accuracy_train)
print('RBF SVC - Test Accuracy: %.4f '%rbf_svc_accuracy_test)


# In[63]:


rbf_svc_auc_train = roc_auc_score(y_train, grid.predict(X_train))
rbf_svc_auc_test = roc_auc_score(y_test, grid.predict(X_test))

print('RBF Kernel SVC - Train ROC AUC Score: %.4f'%rbf_svc_auc_train)
print('RBF Kernel SVC - Test ROC AUC Score: %.4f '%rbf_svc_auc_test)


# In[64]:


pred = grid.predict(X_test)
print(classification_report(y_test, pred))


# In[82]:


Classification_table  = Classification_table +[['rbf Kernelized Support Vector Machine', 'C=100,gamma=0.001', rbf_svc_accuracy_train, rbf_svc_accuracy_test]]


# # Classfication Model (5) - Decision Tree Classification

# In[66]:


model = DecisionTreeClassifier()
parameters = [{'max_depth':[1,2,3,4,5,6,7,8,9,10], 'max_features':[1,2,3,4,5,6,7,8,9,10]}]

#cv =5
grid = GridSearchCV(model, parameters, scoring='roc_auc', cv=5, n_jobs = -1)
grid.fit(X_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[67]:


dtree_accuracy_train = grid.best_estimator_.score(X_train, y_train)
dtree_accuracy_test = grid.best_estimator_.score(X_test, y_test)

print('Decision Tree Classofier - Train Accuracy: %.4f'%dtree_accuracy_train)
print('Decision Tree Classifier - Test Accuracy: %.4f '%dtree_accuracy_test)


# In[68]:


dtree_auc_train = roc_auc_score(y_train, grid.predict(X_train))
dtree_auc_test = roc_auc_score(y_test, grid.predict(X_test))

print('Decision Tree Classifier - Train ROC AUC Score: %.4f'%dtree_auc_train)
print('Decision Tree Classifier - Test ROC AUC Score: %.4f '%dtree_auc_test)


# In[69]:


pred = grid.predict(X_test)
print(classification_report(y_test, pred))


# In[83]:


Classification_table  = Classification_table +[['Decision Tree Classification', 'max_depth=6,max_features=9', dtree_accuracy_train, dtree_accuracy_test]]


# In[84]:


Classification_report = pd.DataFrame(Classification_table,columns = ['Model name', 'Model parameter', 'Train accuracy', 'Test accuracy'])
Classification_report


# In[89]:


sns.barplot(x =Classification_report.index, data = Classification_report)


# # The Best Models

# - Best Regression Model: Linear Regression

# - Best Classification Model: Decision Tree
