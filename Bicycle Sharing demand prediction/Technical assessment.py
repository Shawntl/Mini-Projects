
# coding: utf-8

# # Bicycle sharing demand Prediction

# This is a city bicycle rented system，I'm provided Washington DC bicycle rented records per hour in two years, train datasets include every month first 19 days and test datasets consist of last 10 days(we need to predict this part of time period.

# # Data load and Analysis

# **，we are going to use pandas in python to do data analysis**<br>
# **  numpy is also indispensable**

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[56]:


Folder_Path = '/Users/songzhewei/Desktop/technical assessment'
SaveFile_Name = 'DC2011-2012.csv'  
def read(path, newFileName):
    files = os.listdir(path)
    with open(path + "/" + newFileName, "w") as f:
        for file in files:
            if file != newFileName:
                with open(path + "/" + file) as f1:
                    while 1:
                        line = f1.readline()
                        if not line:
                            break
                        f.write(line)
                f.write("\n")


# In[ ]:


df=read(Folder_Path, SaveFile_Name)


# In[58]:


df1 = pd.read_csv("2011.csv")
df2 = pd.read_csv("2012.csv")


# In[59]:


df=df1.append(df2)


# **<font color=red>load data into cache，show it first，let's see first 10 rows</font>**

# In[60]:


df


# **Then we let pandas to tell us some information**<br>
# **<font color=red>we have to know features name and type at the beginning</font>**

# In[61]:


df.dtypes


# **<font color=red>then we should know how large the dataset is</font>**

# In[62]:


df.shape


# **<font color=red>In conclusion，we have 10886 rows，each row has 12 different features</font>**<br>
# **<font color=red>Also there might be some noise data to deal with，so let's see if there are some missing values</font>**<br>

# In[63]:


df.count()


# **<font color=red>we can see that there is no missing values</font>**

# In[65]:


type(df.datetime)


# **<font color=red>Let's process time feature, since it has much more information and target value always change with time</font>**

# In[66]:


df['month'] = pd.DatetimeIndex(df.datetime).month
df['day'] = pd.DatetimeIndex(df.datetime).dayofweek
df['hour'] = pd.DatetimeIndex(df.datetime).hour


# In[67]:


df.head(10)


# **<font color=red>After preprocessing time series feature, we can drop original time features</font>**<br>
# **<font color=red>And in this baseline version, we don't use registered feature as well</font>**

# In[68]:


df_origin = df
df = df.drop(['datetime','casual','registered'], axis = 1)


# In[69]:


df.head(5)


# **<font color=red>Well, that seems more clear</font>**

# In[70]:


df.shape


# **<font color=red>separate dataset into two:</font>**<br>
# **<font color=red>1. df_target：goal，count feature</font>**<br>
# **<font color=red>2. df_data：data</font>**

# In[73]:


df_target = df['count'].values
df_data = df.drop(['count'],axis = 1).values
print('df_data shape is ', df_data.shape)
print ('df_target shape is ', df_target.shape)


# # Machine Learning Algorithms

# **<font color=red>the process below shows that we might spend lots of time on parameter modification，different parameters would lead to different results</font>**

# In[74]:


from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score


# **<font color=red>data size is small, we are going to try different algorithms</font>**<br><br>
# **<font color=red>We would use cross validation（validation data is 20%）to see model's performance，we would try Suport Vector Regression, Ridge Regression and Random Forest Regressor</font>**<br><br>
# 

# In[84]:


cv = cross_validation.ShuffleSplit(len(df_data), n_iter=3, test_size=0.2,
    random_state=0)

print("ridge")    
for train, test in cv:    
    svc = linear_model.Ridge().fit(df_data[train], df_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_data[train], df_target[train]), svc.score(df_data[test], df_target[test])))
    
print ("SVR(kernel='rbf',C=10,gamma=.001)")
for train, test in cv:
    
    svc = svm.SVR(kernel ='rbf', C = 10, gamma = .001).fit(df_data[train], df_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_data[train], df_target[train]), svc.score(df_data[test], df_target[test])))
    
print ("Random Forest(n_estimators = 100)")    
for train, test in cv:    
    svc = RandomForestRegressor(n_estimators = 100).fit(df_data[train], df_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_data[train], df_target[train]), svc.score(df_data[test], df_target[test])))


# **<font color=red>Random forest has best performance</font>**<br><br>
# **<font color=red>Next step we are going to do parameter modification<font>**<br><br>
# **<font color=red>There is tools call grid search which can help us find optimal parameter</font>**<br><br>

# In[88]:


X = df_data
y = df_target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=0)

tuned_parameters = [{'n_estimators':[10,100,500]}]   
    
scores = ['r2']

for score in scores:
    
    print(score)
    
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("we found optimal parameter")
    print( "")
    #best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
    print ("")
    print("score is:")
    print ("")

    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print( "")


# **<font color=red>We can see，Grid Search is helpful，we use these parameter on our model。</font>**<br>
# **<font color=red>we also need to check whether our model is overfitting</font>**<br>
# **<font color=red>plot learning curve</font>**

# In[89]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves (Random Forest, n_estimators = 100)"
cv = cross_validation.ShuffleSplit(df_data.shape[0], n_iter=10,test_size=0.2, random_state=0)
estimator = RandomForestRegressor(n_estimators = 100)
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)

plt.show()


# **<font color=red>There is a big gap between training curve and test curve, overfiting occured</font>**<br>

# In[25]:


# migitate overfitting
print "Random Forest(n_estimators=200, max_features=0.6, max_depth=15)"
for train, test in cv: 
    svc = RandomForestRegressor(n_estimators = 200, max_features=0.6, max_depth=15).fit(df_data[train], df_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_data[train], df_target[train]), svc.score(df_data[test], df_target[test])))


# **<font color=red>we can use registered feature to do prediction</font>**<br>
# **<font color=red>separate dataset into two</font>**<br>

# In[26]:


df_registered = df_origin.drop(['datetime','casual','count'], axis = 1)
df_casual = df_origin.drop(['datetime','count','registered'], axis = 1)


# In[27]:


df_train_registered.head()


# In[29]:


df_train_casual.head()


# #### <font color=red>Data analysis and visulization</font>

# In[40]:


# windspeed
df_origin.groupby('windspeed').mean().plot(y='count', marker='o')
plt.show()


# In[41]:


# humidity
df_origin.groupby('humidity').mean().plot(y='count', marker='o')
plt.show()


# In[42]:


# temperature
df_origin.groupby('temp').mean().plot(y='count', marker='o')
plt.show()


# In[46]:


#temp humidity changing
df_train_origin.plot(x='temp', y='humidity', kind='scatter')
plt.show()


# In[35]:


# scatter different dimentions distribution
fig, axs = plt.subplots(2, 3, sharey=True)
df_origin.plot(kind='scatter', x='temp', y='count', ax=axs[0, 0], figsize=(16, 8), color='magenta')
df_origin.plot(kind='scatter', x='atemp', y='count', ax=axs[0, 1], color='cyan')
df_origin.plot(kind='scatter', x='humidity', y='count', ax=axs[0, 2], color='red')
df_origin.plot(kind='scatter', x='windspeed', y='count', ax=axs[1, 0], color='yellow')
df_origin.plot(kind='scatter', x='month', y='count', ax=axs[1, 1], color='blue')
df_origin.plot(kind='scatter', x='hour', y='count', ax=axs[1, 2], color='green')


# In[37]:


sns.pairplot(df_origin[["temp", "month", "humidity", "count"]], hue="count")


# In[48]:


# correlation analysis
corr = df_origin[['temp','weather','windspeed','day', 'month', 'hour','count']].corr()
corr


# In[52]:



plt.figure()
plt.matshow(corr)
plt.colorbar()
plt.show()

