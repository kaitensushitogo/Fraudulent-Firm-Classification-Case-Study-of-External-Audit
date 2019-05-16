
# coding: utf-8

# ### <font color='red'> Project 2
# 
# Project Description:
# - Use same datasets as Project 1.
# - Preprocess data: Explore data and apply data scaling.
# 
# Regression Task:
# - Apply any two models with bagging and any two models with pasting.
# - Apply any two models with adaboost boosting
# - Apply one model with gradient boosting
# - Apply PCA on data and then apply all the models in project 1 again on data you get from PCA. Compare your results with results in project 2. You don't need to apply all the models twice. Just copy the result table from project 1, prepare similar table for all the models after PCA and compare both tables. Does PCA help in getting better results?
# - Apply deep learning models covered in class
# 
# Classification Task:
# - Apply two voting classifiers - one with hard voting and one with soft voting
# - Apply any two models with bagging and any two models with pasting.
# - Apply any two models with adaboost boosting
# - Apply one model with gradient boosting
# - Apply PCA on data and then apply all the models in project 1 again on data you get from PCA. Compare your results with results in project 1. You don't need to apply all the models twice. Just copy the result table from project 1, prepare similar table for all the models after PCA and compare both tables. Does PCA help in getting better results?
# - Apply deep learning models covered in class
# 
# Deliverables:
# - Use markdown to provide inline comments for this project.
# - Your outputs should be clearly executed in the notebook i.e. we should not need to rerun the code to obtain the outputs.
# - Visualization encouraged.
# - If you are submitting two different files, then please only one group member submit both the files. If you submit two files separately from different accounts, it will be submitted as two different attempts.
# - If you are submitting two different files, then please follow below naming convetion:
#     Project2_Regression_GroupXX_Firstname1_Firstname2.ipynb
#     Project2_Classification_GroupXX_Firstname1_Firstname2.ipynb
# - If you are submitting single file, then please follow below naming convetion:
#     Project2_Both_GroupXX_Firstname1_Firstname2.ipynb
# 
# Questions regarding the project:
# - We have created a discussion board under Projects folder on e-learning. Create threads over there and post your queries related to project there.
# - We will also answer queries there. We will not be answering any project related queries through the mail.

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


# In[4]:


audit_risk.info() 
#one missing values in Money_Value, and Location_ID dtype is object; need to be fixed


# In[5]:


audit_risk.describe()


# In[6]:


trial.info() #No missing values but Location_ID is object; need to be fixed


# In[7]:


trial.describe()


# In[8]:


# Remove categorical values in the column of LOCATION_ID
# Did not assign new values because they could be already one of the LOCATION_ID
audit_risk = audit_risk[audit_risk.LOCATION_ID != 'LOHARU']
audit_risk = audit_risk[audit_risk.LOCATION_ID != 'NUH']
audit_risk = audit_risk[audit_risk.LOCATION_ID != 'SAFIDON']

trial = trial[trial.LOCATION_ID != 'LOHARU']
trial = trial[trial.LOCATION_ID != 'NUH']
trial = trial[trial.LOCATION_ID != 'SAFIDON']


# In[9]:


# Changing dtype(object to integer)
audit_risk['LOCATION_ID'] = pd.to_numeric(audit_risk['LOCATION_ID'])
trial['LOCATION_ID'] = pd.to_numeric(trial['LOCATION_ID'])


# In[10]:


# Removing duplicated columns
# We removed duplicated parameters with different scales in advance 
trial.drop(['Sector_score','LOCATION_ID','PARA_A','SCORE_A',
            'PARA_B','SCORE_B','TOTAL','numbers','District','Money_Value','History','Score','Risk'], axis=1, inplace = True)


# In[11]:


# Dealing with missing values
print(audit_risk.isnull().sum().sort_values(ascending = False))
audit_risk['Money_Value'] = audit_risk.fillna(audit_risk['Money_Value'].mean())


# In[12]:


trial.isnull().sum().sort_values(ascending = False)


# In[13]:


# Checking the shape again 
print(audit_risk.shape)
print(trial.shape)


# In[14]:


# Merging datasets
df = pd.concat([audit_risk, trial], axis=1)
df.shape


# In[15]:


df.info()


# ## Exploring & Scaling dataset

# In[16]:


#Exploring dataset - (1) Correlation within variables
plt.figure(figsize=(12,8))
sns.heatmap(df.drop(['Detection_Risk'], axis=1).corr(), cmap = 'coolwarm')


# In[17]:


# Correlation with target variables
cor = pd.DataFrame(df.drop("Audit_Risk", axis=1).apply(lambda x: x.corr(df.Audit_Risk)).sort_values(ascending = False)).rename(columns = {0:'Correlation'})
cor.dropna()


# In[18]:


from sklearn.preprocessing import MinMaxScaler # Used MinMaxScaler
reg_data = df.drop(['Audit_Risk','Risk'], axis=1)
reg_target = df['Audit_Risk']


# In[19]:


# Splitting training and test data
from sklearn.model_selection import train_test_split
X_train_org, X_test_org, y_train, y_test = train_test_split(reg_data, reg_target, random_state=0)

# Scaling data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# # Regression Task 

# ## (1) Two models with bagging and pasting

# In[20]:


#importing models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import BaggingRegressor


# In[21]:


#Linear Regression
print('(1) Linear Regression')
print()

lreg = LinearRegression()

lreg_bag = BaggingRegressor(lreg, bootstrap=True, n_jobs=-1, random_state=0)
lreg_bag.fit(X_train, y_train)
print("Bagging")
print("Train Score: {:4f}".format(lreg_bag.score(X_train, y_train)))
print("Test Score: {:4f}".format(lreg_bag.score(X_test, y_test)))
print()

lreg_pas = BaggingRegressor(lreg, bootstrap=False, n_jobs=-1, random_state=0)
lreg_pas.fit(X_train, y_train)
print("Pasting")
print("Train Score: {:4f}".format(lreg_pas.score(X_train, y_train)))
print("Test Score: {:4f}".format(lreg_pas.score(X_test, y_test)))


# In[22]:


#LASSO Regression
print('(2) Lasso Regression')
print()

lasso = Lasso(alpha=1.0) #the best parameter alpha=1.0
lasso_bag = BaggingRegressor(lasso, bootstrap=True, n_jobs=-1, random_state=0)
lasso_bag.fit(X_train, y_train)
print("Bagging")
print("Train Score: {:4f}".format(lasso_bag.score(X_train, y_train)))
print("Test Score: {:4f}".format(lasso_bag.score(X_test, y_test)))
print()

print("Pasting")
lasso_pas = BaggingRegressor(lasso, bootstrap=False, n_jobs=-1, random_state=0)
lasso_pas.fit(X_train, y_train)
print("Train Score: {:4f}".format(lasso_pas.score(X_train, y_train)))
print("Test Score: {:4f}".format(lasso_pas.score(X_test, y_test)))


# In[23]:


#KNN Regression
print('(3) KNN Regression')
print()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

knn_reg = KNeighborsRegressor(3)  #the best parameter n=3
knn_reg.fit(X_train, y_train)
knn_bag = BaggingRegressor(knn_reg, bootstrap=True, n_jobs=-1, random_state=0)
lasso_bag.fit(X_train, y_train)
print("Bagging")
print("Train Score: {:4f}".format(lasso_bag.score(X_train, y_train)))
print("Test Score: {:4f}".format(lasso_bag.score(X_test, y_test)))
print()

print("Pasting")
lasso_pas = BaggingRegressor(knn_reg, bootstrap=False, n_jobs=-1, random_state=0)
lasso_pas.fit(X_train, y_train)
print("Train Score: {:4f}".format(lasso_pas.score(X_train, y_train)))
print("Test Score: {:4f}".format(lasso_pas.score(X_test, y_test)))


# ## (2) Two models with Adaboost boosting

# In[24]:


from sklearn.ensemble import AdaBoostRegressor

#Linear Regression
print('(1) Linear Regression')
print()

param_grid = { 'n_estimators': [10, 50, 100],
              'learning_rate' : [0.1,0.5,1] }
ada_grid = GridSearchCV(AdaBoostRegressor(LinearRegression()), 
                        param_grid,n_jobs=-1)
ada_grid.fit(X_train, y_train)
print('Best Parameter: {}'.format(ada_grid.best_params_))

ada_reg = AdaBoostRegressor(LinearRegression(), learning_rate=0.1, n_estimators=10)
ada_reg.fit(X_train, y_train)
print("Train Score: {:4f}".format(ada_reg.score(X_train, y_train)))
print("Test Score: {:4f}".format(ada_reg.score(X_test, y_test)))
print()


# In[25]:


from sklearn.ensemble import AdaBoostRegressor

#Linear Regression
print('(2) LASSO Regression')
print()

param_grid = { 'n_estimators': [10, 50, 100],
              'learning_rate' : [0.1,0.5,1] }
ada_grid = GridSearchCV(AdaBoostRegressor(Lasso(alpha=1.0)), 
                        param_grid,n_jobs=-1)
ada_grid.fit(X_train, y_train)
print('Best Parameter: {}'.format(ada_grid.best_params_))

ada_reg = AdaBoostRegressor(Lasso(alpha=1.0), learning_rate=0.1, n_estimators=50)
ada_reg.fit(X_train, y_train)
print("Train Score: {:4f}".format(ada_reg.score(X_train, y_train)))
print("Test Score: {:4f}".format(ada_reg.score(X_test, y_test)))
print()


# ## (3) One model with Gradient boosting
# 

# In[26]:


from sklearn.ensemble import GradientBoostingRegressor

#Linear Regression
print('(1) Gradient Boosted Regression Trees')
print()

param_grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.5],
    "max_depth":[3,5,8, 10],
    "subsample":[0.5, 0.75, 1.0],
    "n_estimators":[3, 5, 8, 10]
    }

gbr_grid = GridSearchCV(GradientBoostingRegressor(),  
                        param_grid,n_jobs=-1)
gbr_grid.fit(X_train, y_train)
print('Best Parameter: {}'.format(gbr_grid.best_params_))


gbr = GradientBoostingRegressor(learning_rate=0.5, max_depth=8, n_estimators=10, subsample=0.75, random_state=0)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Train Score: {:4f}".format(gbr.score(X_train, y_train)))
print("Test Score: {:4f}".format(gbr.score(X_test, y_test)))
print("Mean Squared Error: {:4f}".format(mse))


# ### (4) PCA 

# In[27]:


# Splitting training and test data
from sklearn.model_selection import train_test_split
X_train_org, X_test_org, y_train, y_test = train_test_split(reg_data, reg_target, random_state=0)

# Scaling data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)

# Dimension Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components= 0.95)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test = pca.transform(X_test)


# In[28]:


print('Number of dimensions:', pca.n_components_) 
print('Explained Variance Ratio:',1- pca.explained_variance_ratio_.sum())


# In[29]:


#(1) Linear Regression

lreg = LinearRegression()
param_grid = {'fit_intercept':[True,False], 'normalize':[True,False]}

#cv =5
grid = GridSearchCV(lreg, param_grid, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[30]:


lreg_accuracy_train = grid.best_estimator_.score(X_reduced_test, y_test)
lreg_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Linear Regression - Train Score: %.2f'%lreg_accuracy_train)
print('Linear Regression - Test Score: %.2f '%lreg_accuracy_test)


# In[31]:


report_table = [['Linear Regression', '',
                 grid.best_estimator_.score(X_reduced_train, y_train), 
                 grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[32]:


#(2) Ridge Regression
from sklearn.linear_model import Ridge
param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# create and fit a ridge regression model, testing each alpha
# cv = 5
model = Ridge(random_state=0)
grid = GridSearchCV(model, param_grid, cv = 5)
grid.fit(X_reduced_train, y_train)
# summarize the results of the grid search
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[33]:


ridge_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
ridge_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Ridge Regression - Train Score: %.2f'%ridge_accuracy_train)
print('Ridge Regression - Test Score: %.2f '%ridge_accuracy_test)


# In[34]:


report_table = report_table + [['Ridge', 'alpha = 100', 
                                grid.best_estimator_.score(X_reduced_train, y_train), 
                                grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[35]:


#(3) LASSO
from sklearn.linear_model import Lasso
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# create and fit a ridge regression model, testing each alpha
# cv = 5
model = Lasso(random_state=0)
grid = GridSearchCV(model, param_grid, cv = 5)
grid.fit(X_reduced_train, y_train)
# summarize the results of the grid search
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[36]:


lasso_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
lasso_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Lasso Regression - Train Score: %.2f'%lasso_accuracy_train)
print('Lasso Regression - Test Score: %.2f '%lasso_accuracy_test)


# In[37]:


report_table = report_table + [['Lasso', 'alpha = 1.0', 
                                grid.best_estimator_.score(X_reduced_train, y_train), 
                                grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[38]:


#(4) KNN
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
param_grid = {'n_neighbors':np.arange(1,11,1)}

#cv =5
grid = GridSearchCV(knn, param_grid, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[39]:


knnreg_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
knnreg_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('KNN Regression - Train Score: %.2f'%knnreg_accuracy_train)
print('KNN Regression - Test Score: %.2f '%knnreg_accuracy_test)


# In[40]:


report_table =report_table +  [['KNN', 'n = 3', 
                                grid.best_estimator_.score(X_reduced_train, y_train), 
                                grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[41]:


#(5) Polynomial Regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

param_grid = {'polynomialfeatures__degree': np.arange(4), 
        'linearregression__fit_intercept': [True, False], 
        'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=5, n_jobs=-1)

grid.fit(X_reduced_train, y_train)
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))


# In[42]:


poly_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
poly_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Polynomial Regression - Train Score: %.2f'%poly_accuracy_train)
print('Polynomial Regression - Test Score: %.2f '%poly_accuracy_test)


# In[43]:


report_table =report_table +  [['Polynomial', 'degree=1', 
                                grid.best_estimator_.score(X_reduced_train, y_train), 
                                grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[44]:


#(6) Linear SVR
from sklearn.svm import LinearSVR
model = LinearSVR()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10], 'epsilon':[0.001,0.01,0.1,1,10]}

#cv = 5
grid = GridSearchCV(model, parameters, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[45]:


lsvr_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
lsvr_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Linear SVR - Train Score: %.2f'%lsvr_accuracy_train)
print('Linear SVR - Test Score: %.2f '%lsvr_accuracy_test)


# In[46]:


report_table =report_table +  [['SVR', 'C=1, epsilon=10', 
                                grid.best_estimator_.score(X_reduced_train, y_train), 
                                grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[47]:


#(7) RBF kernel SVR
from sklearn.svm import SVR
model = SVR(kernel='rbf')
parameters = {'C':[0.001, 0.01, 0.1, 1,10],'gamma':[0.0001,0.001, 0.01, 0.1, 1],'epsilon':[0.01, 0.1, 1]}

#cv =5
grid = GridSearchCV(model, parameters, cv=5, return_train_score=True, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print()


# In[48]:


rbf_svr_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
rbf_svr_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('RBF SVR - Train Score: %.2f'%rbf_svr_accuracy_train)
print('RBF SVR - Test Score: %.2f '%rbf_svr_accuracy_test)


# In[49]:


report_table = report_table + [['RBF Kernel SVR', 'C=10, epsilon=1, gamma=0.1', 
                                grid.best_estimator_.score(X_reduced_train, y_train), 
                                grid.best_estimator_.score(X_reduced_test, y_test)]]


# In[50]:


report = pd.DataFrame(report_table,columns = ['Model Name', 'Model Parameter', 'Train Score', 'Test Score'])
report.index = report['Model Name']
report.drop(['Model Name'],axis=1,inplace=True)
report= report.drop(['Polynomial'])# as polynomial regression with the degree of 1 is exatly same to linear Regression
report.sort_values(by='Test Score', ascending = False)


# In[51]:


sns.barplot(y =report.index, x = 'Test Score',data = report.sort_values(by='Test Score', ascending=False))


# Result from Project 1
# <img src="Result Table.png">
# <img src="Result 1.png">

# PCA result is not bettern than the one from Project 1 on Regression Task

# ### (5) Deep Learning Task

# In[103]:


#Loading packages
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# In[101]:


def create_model():
    # Defining model
    model = Sequential()
    model.add(Dense(12, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compiling model
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
    return model


# In[102]:


seed = 10
np.random.seed(10)


# In[104]:


model = KerasRegressor(build_fn= create_model, verbose=0)
param_grid = {'batch_size':[5,10,25,50], 'epochs':[10, 50,100]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)


# In[105]:


# Fitting model
grid_search_result = grid_search.fit(X_train, y_train)


# In[120]:


print("Best parameters: {}".format(grid_search_result.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_result.best_score_))
print()


# In[119]:


grid_train_pred = grid_search_result.predict(X_train)
grid_test_pred = grid_search_result.predict(X_test)


# In[123]:


dl_train_score = grid_search_result.best_estimator_.score(X_train, y_train)
dl_test_score = grid_search_result.best_estimator_.score(X_test, y_test)

print('Deep Learning - Train Score: %.2f'%dl_train_score)
print('Deep Learning - Test Score: %.2f '%dl_test_score)


# In[124]:


from sklearn.metrics import r2_score, recall_score, precision_score

print('Train Score {:.2f}'.format(r2_score(y_train, grid_train_pred)))
print('Test Score {:.2f}'.format(r2_score(y_test, grid_test_pred)))


# # Classification Task

# In[52]:


# Creating dataset for classification task
clf_data = df.drop(['Audit_Risk','Risk'], axis=1)
clf_target = df['Risk']

# Splitting training and test data
from sklearn.model_selection import train_test_split
X_train_org, X_test_org, y_train, y_test = train_test_split(clf_data, clf_target, random_state=0)

# Scaling data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# In[53]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ## (1) Two voting classifiers - one with hard voting and one with soft voting

# In[54]:


log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
knn_clf = KNeighborsClassifier(7)
knn_clf.fit(X_train, y_train)
svm_clf = SVC(C = 10, probability = True)
svm_clf.fit(X_train, y_train)

voting_hard_clf = VotingClassifier(estimators=[('lr', log_clf), ('knn', knn_clf), ('svc', svm_clf)], voting='hard')
voting_hard_clf.fit(X_train, y_train)

print("Hard Voting")
from sklearn.metrics import accuracy_score
for clf in (log_clf, knn_clf, svm_clf, voting_hard_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, 'Test Accuracy Score {:.4f}'.format(accuracy_score(y_test, y_pred)))
print()

("Soft Voting")
voting_soft_clf = VotingClassifier(estimators=[('lr', log_clf), ('knn', knn_clf), ('svc', svm_clf)], voting='soft')
voting_soft_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, knn_clf, svm_clf, voting_soft_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, 'Test Accuracy Score {:.4f}'.format(accuracy_score(y_test, y_pred)))


# ## (2) Two models with bagging and any two models with pasting

# In[55]:


#importing models
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, classification_report


# In[56]:


#Logistic Regression
print('(1) Logistic Regression')
print()

log = LogisticRegression(C=10, penalty='l1')

log_bag = BaggingClassifier(log, bootstrap=True, n_jobs=-1, random_state=0)
log_bag.fit(X_train, y_train)
y_pred_bag = log_bag.predict(X_test)

print("Bagging")
print("- Accuracy Score")
print("Train Accuracy Score: {:4f}".format(log_bag.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(log_bag.score(X_test, y_test)))
print()

#Classficiation Report
print(classification_report(y_test, y_pred_bag))

#ROC AUC Score
print("- ROC AUC Score")
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_bag)))
print()
print()


log_pas = BaggingClassifier(log, bootstrap=False, n_jobs=-1, random_state=0)
log_pas.fit(X_train, y_train)
y_pred_pas = log_pas.predict(X_test)
print("Pasting")
print("Train Accuracy Score: {:4f}".format(log_pas.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(log_pas.score(X_test, y_test)))
      
#Classficiation Report
print(classification_report(y_test, y_pred_pas))
print()

#ROC AUC Score
print('- ROC AUC Score')
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_pas)))
print()


# In[57]:


#Linear SVC
from sklearn.svm import LinearSVC
print('(1) Linear SVC')
print()

lsvc = LinearSVC(C= 10, penalty = 'l1', dual=False, loss='squared_hinge')

lsvc_bag = BaggingClassifier(lsvc, bootstrap=True, n_jobs=-1, random_state=0)
lsvc_bag.fit(X_train, y_train)
y_pred_bag = lsvc_bag.predict(X_test)

print("Bagging")
print("- Accuracy Score")
print("Train Accuracy Score: {:4f}".format(lsvc_bag.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(lsvc_bag.score(X_test, y_test)))
print()

#Classficiation Report
print(classification_report(y_test, y_pred_bag))

#ROC AUC Score
print("- ROC AUC Score")
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_bag)))
print()
print()


lsvc_pas = BaggingClassifier(lsvc, bootstrap=False, n_jobs=-1, random_state=0)
lsvc_pas.fit(X_train, y_train)
y_pred_pas = log_pas.predict(X_test)
print("Pasting")
print("Train Accuracy Score: {:4f}".format(lsvc_pas.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(lsvc_pas.score(X_test, y_test)))
      
#Classficiation Report
print(classification_report(y_test, y_pred_pas))
print()

#ROC AUC Score
print('- ROC AUC Score')
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_pas)))
print()


# ##  (3) Two models with adaboost boosting

# In[58]:


from sklearn.ensemble import AdaBoostClassifier

#Logistic Regression
print('(1) Logistic Regression')
print()

param_grid = { 'n_estimators': [3, 5, 10, 50, 75, 100],
              'learning_rate' : [0.01, 0.05, 0.1, 0.5,1] }
ada_grid = GridSearchCV(AdaBoostClassifier(LogisticRegression(C=10, penalty='l1')), 
                        param_grid,n_jobs=-1)
ada_grid.fit(X_train, y_train)
print('Best Parameter: {}'.format(ada_grid.best_params_))
print()

ada_clf = AdaBoostClassifier(LogisticRegression(), learning_rate=0.05, n_estimators=50)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

print("- Accuracy Score")
print("Train Accuracy Score: {:4f}".format(ada_clf.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(ada_clf.score(X_test, y_test)))
print()

#Classficiation Report
print(classification_report(y_test, y_pred_ada))

#ROC AUC Score
print("- ROC AUC Score")
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_ada)))
print()
print()


# In[59]:


from sklearn.svm import LinearSVC
#Linear SVC
print('(2) Linear SVC')
print()

param_grid = { 'n_estimators': [3, 5, 10, 50, 75, 100],
              'learning_rate' : [0.01, 0.05, 0.1, 0.5,1] }
ada_grid = GridSearchCV(AdaBoostClassifier(LinearSVC(C= 10, penalty = 'l1', dual=False, loss='squared_hinge'),
                                           algorithm='SAMME'), param_grid,n_jobs=-1)
ada_grid.fit(X_train, y_train)
print('Best Parameter: {}'.format(ada_grid.best_params_))
print()

ada_clf = AdaBoostClassifier(LinearSVC(C= 10, penalty = 'l1', dual=False, loss='squared_hinge'), 
                             algorithm='SAMME', learning_rate=0.01, n_estimators=3)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

print("- Accuracy Score")
print("Train Accuracy Score: {:4f}".format(ada_clf.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(ada_clf.score(X_test, y_test)))
print()

#Classficiation Report
print(classification_report(y_test, y_pred_ada))

#ROC AUC Score
print("- ROC AUC Score")
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_ada)))


# ## (4) One model with gradient boosting 

# In[60]:


from sklearn.ensemble import GradientBoostingClassifier

print('(1) Gradient Boosted Classification Trees')
print()

param_grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.5],
    "max_depth":[3,5,8, 10],
    "subsample":[0.5, 0.75, 1.0],
    "n_estimators":[3, 5, 8, 10]
    }

gbr_grid = GridSearchCV(GradientBoostingClassifier(random_state=1),  
                        param_grid,n_jobs=-1)
gbr_grid.fit(X_train, y_train)
print('Best Parameter: {}'.format(gbr_grid.best_params_))


# In[61]:


gbr_clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=58, subsample =0.75, random_state=0)
gbr_clf.fit(X_train, y_train)
y_pred_gbr = gbr_clf.predict(X_test)

print("- Accuracy Score")
print("Train Accuracy Score: {:4f}".format(gbr_clf.score(X_train, y_train)))
print("Test Accuracy Score: {:4f}".format(gbr_clf.score(X_test, y_test)))
print()

#Classficiation Report
print(classification_report(y_test, y_pred_gbr))

#ROC AUC Score
print("- ROC AUC Score")
print("Test AUC Score: {:4f}".format(roc_auc_score(y_test, y_pred_gbr)))


# ### (5) PCA

# In[62]:


# Dimension Reduction
pca = PCA(0.95)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test = pca.transform(X_test)

print('Number of dimensions:', pca.n_components_) 
print('Explained Variance Ratio:',1- pca.explained_variance_ratio_.sum())


# In[63]:


#(1) KNN 
knn = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(1,11,1)}

#cv =5
grid = GridSearchCV(knn, parameters, scoring='roc_auc',cv=5, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[64]:


knnclf_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
knnclf_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('KNN Classifier - Train Accuracy: %.4f'%knnclf_accuracy_train)
print('KNN Classifier - Test Accuracy: %.4f '%knnclf_accuracy_test)


# In[65]:


knnclf_auc_train = roc_auc_score(y_train, grid.predict(X_reduced_train))
knnclf_auc_test = roc_auc_score(y_test, grid.predict(X_reduced_test))

print('KNN Classifier - Train ROC AUC Score: %.4f'%knnclf_auc_train)
print('KNN Classifier - Test ROC AUC Score: %.4f '%knnclf_auc_test)


# In[66]:


pred = grid.predict(X_reduced_test)
print(classification_report(y_test, pred))


# In[67]:


report_table = [['KNN', 'n=7',knnclf_accuracy_train, knnclf_accuracy_test, knnclf_auc_train,knnclf_auc_test]]


# In[68]:


#(2) Logistic Regression
model = LogisticRegression(random_state=0)
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

#cv =5
grid = GridSearchCV(model, parameters, scoring='roc_auc', cv=5, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[69]:


logistic_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
logistic_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Logistic Regression Classifier - Train Accuracy: %.4f'%logistic_accuracy_train)
print('Logistic Regression Classifier - Test Accuracy: %.4f '%logistic_accuracy_test)


# In[70]:


logistic_auc_train = roc_auc_score(y_train, grid.predict(X_reduced_train))
logistic_auc_test = roc_auc_score(y_test, grid.predict(X_reduced_test))

print('Logstic Regression - Train ROC AUC Score: %.4f'%logistic_auc_train)
print('Logistic Regression - Test ROC AUC Score: %.4f '%logistic_auc_test)


# In[71]:


pred = grid.predict(X_reduced_test)
print(classification_report(y_test, pred))


# In[72]:


report_table = report_table + [['Logistic Regression', 'C=100, penalty=l2', 
                                logistic_accuracy_train, logistic_accuracy_test, logistic_auc_train,logistic_auc_test]]


# In[73]:


#(3) Linear SVC
model = LinearSVC(random_state=0)
parameters = [{'C':[0.001, 0.01, 0.1, 1, 10, 100], 'penalty':['l2'],'loss':['hinge','squared_hinge']},
              {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'penalty':['l1'], 'dual':[False], 'loss':['squared_hinge']}]

#cv =5
grid = GridSearchCV(model, parameters, scoring= 'roc_auc', cv=5, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[74]:


lsvc_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
lsvc_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test) 

print('Linear SVC - Train Accuracy: %.4f'%lsvc_accuracy_train)
print('Linear SVC - Test Accuracy: %.4f '%lsvc_accuracy_test)


# In[75]:


lsvc_auc_train = roc_auc_score(y_train, grid.predict(X_reduced_train))
lsvc_auc_test = roc_auc_score(y_test, grid.predict(X_reduced_test))

print('Linear SVC - Train ROC AUC Score: %.4f'%lsvc_auc_train)
print('Linear SVC - Test ROC AUC Score: %.4f '%lsvc_auc_test)


# In[76]:


pred = grid.predict(X_reduced_test)
print(classification_report(y_test, pred))


# In[77]:


report_table  = report_table +[['Linear SVC', 'C=100, penalty=l2', 
                                lsvc_accuracy_train, lsvc_accuracy_test,lsvc_auc_train,lsvc_auc_test]]


# In[78]:


#(4) RBF Kernel SVC
model = SVC(kernel='rbf', random_state=0)
parameters = [{'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001,0.01,0.1,1,10,100]}]

#cv =5
grid = GridSearchCV(model, parameters, scoring = 'roc_auc',cv=5, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[79]:


rbf_svc_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
rbf_svc_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('RBF SVC - Train Accuracy: %.4f'%rbf_svc_accuracy_train)
print('RBF SVC - Test Accuracy: %.4f '%rbf_svc_accuracy_test)


# In[80]:


rbf_svc_auc_train = roc_auc_score(y_train, grid.predict(X_reduced_train))
rbf_svc_auc_test = roc_auc_score(y_test, grid.predict(X_reduced_test))

print('RBF Kernel SVC - Train ROC AUC Score: %.4f'%rbf_svc_auc_train)
print('RBF Kernel SVC - Test ROC AUC Score: %.4f '%rbf_svc_auc_test)


# In[81]:


pred = grid.predict(X_reduced_test)
print(classification_report(y_test, pred))


# In[82]:


report_table = report_table +[['rbf Kernelized Support Vector Machine', 'C=0.001,gamma=1', 
                               rbf_svc_accuracy_train, rbf_svc_accuracy_test,rbf_svc_auc_train, rbf_svc_auc_test]]


# In[83]:


#(5) Decission Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
parameters = [{'max_depth':[1,2,3,4,5,6,7,8,9,10], 'max_features':[1,2,3,4,5,6,7,8]}]

#cv =5
grid = GridSearchCV(model, parameters, scoring='roc_auc', cv=5, n_jobs = -1)
grid.fit(X_reduced_train, y_train)
print('cv=5')
print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.4f}".format(grid.best_score_))
print()


# In[84]:


dtree_accuracy_train = grid.best_estimator_.score(X_reduced_train, y_train)
dtree_accuracy_test = grid.best_estimator_.score(X_reduced_test, y_test)

print('Decision Tree Classofier - Train Accuracy: %.4f'%dtree_accuracy_train)
print('Decision Tree Classifier - Test Accuracy: %.4f '%dtree_accuracy_test)


# In[85]:


dtree_auc_train = roc_auc_score(y_train, grid.predict(X_reduced_train))
dtree_auc_test = roc_auc_score(y_test, grid.predict(X_reduced_test))

print('Decision Tree Classifier - Train ROC AUC Score: %.4f'%dtree_auc_train)
print('Decision Tree Classifier - Test ROC AUC Score: %.4f '%dtree_auc_test)


# In[86]:


pred = grid.predict(X_reduced_test)
print(classification_report(y_test, pred))


# In[87]:


report_table = report_table +[['Decision Tree Classification', 'max_depth=4, max_features=3',
                               dtree_accuracy_train, dtree_accuracy_test, dtree_auc_train, dtree_auc_test]]


# In[88]:


report_table = pd.DataFrame(report_table, columns = ['Model Name', 'Model Parameter', 'Train Accuracy', 'Test Accuracy', 'Train AUC Score', 'Test AUC Score'])
report_table.index = report_table['Model Name']
report_table.drop(['Model Name'],axis=1,inplace=True)


# In[89]:


report_table.sort_values(by=['Test Accuracy','Test AUC Score'], ascending = False)


# In[90]:


sns.barplot(y =report_table.index, x = 'Test AUC Score', data= report_table.sort_values(by='Test AUC Score', ascending=False))


# Result from Project 1
# <img src="Result1.png">

# PCA result is not better than the one from Project 1 on Classification Task

# ### (6) Deep Learning Task

# In[125]:


from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    # Defining model
    model = Sequential()
    model.add(Dense(12, input_dim=30,activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compiling model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[126]:


model = KerasClassifier(build_fn= create_model, verbose=0)
param_grid = {'batch_size':[5,10,25,50], 'epochs':[10, 50,100]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)


# In[127]:


# Fitting model
grid_search_result = grid_search.fit(X_train, y_train)


# In[132]:


print("Best parameters: {}".format(grid_search_result.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_result.best_score_))
print()


# In[134]:


dl_accuracy_train = grid_search_result.best_estimator_.score(X_train, y_train)
dl_accuracy_test = grid_search_result.best_estimator_.score(X_test, y_test)

print('Deep Learning - Train Accuracy: %.4f'%dl_accuracy_train)
print('Deep Learning - Test Accuracy: %.4f '%dl_accuracy_test)


# dl_train_score = grid_search_result.best_estimator_.score(X_train, y_train)
# dl_test_score = grid_search_result.best_estimator_.score(X_test, y_test)
# 
# print('Deep Learning - Train Score: %.2f'%dl_train_score)
# print('Deep Learning - Test Score: %.2f '%dl_test_score)

# In[138]:


grid_train_pred = grid_search_result.predict(X_train)
grid_test_pred = grid_search_result.predict(X_test)

print('Deep Learning - ROC AUC Score {:.4f}'.format(roc_auc_score(y_train, grid_train_pred)))
print('Deep Learning - ROC AUC Score {:.4f}'.format(roc_auc_score(y_test, grid_test_pred)))

