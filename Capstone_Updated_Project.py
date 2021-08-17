#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[173]:


import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 
import seaborn as sns
import scipy.stats
import statsmodels.api as sm


# In[174]:


import warnings
warnings.filterwarnings('ignore')


# # Loading the datasets

# In[175]:


mat = pd.read_csv("Desktop\student-mat.csv", sep=';')

por = pd.read_csv("Desktop\student-por.csv", sep=';')
data=pd.concat([mat,por])
data.G3.unique()


# In[ ]:





# # Checking through the data to see if there are missing values missing values 

# In[177]:



data.isnull().sum()


# In[178]:


data.columns


# # Checking the shape of the data 

# In[179]:



np.shape(data)


# # Final distribution of the final grade

# In[180]:


plt.figure(figsize=(15,5))
sns.countplot(x=data['G3'], palette='Set1')
plt.title('Final Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Students', fontsize=16)
plt.show()


# # Correlation between variables using heatmap

# In[181]:


corr = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap="RdBu")
plt.title('Correlation Heatmap', fontsize=2)


# In[ ]:





# # Preparing our data for model training

# In[182]:


dfd = data.copy()


# In[183]:


# label encode final_grade
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dfd.G3 = le.fit_transform(dfd.G3)


# In[184]:


X = dfd.drop('G3',axis=1)
y = dfd.G3

y.unique()


# In[185]:


X = X.apply(pd.to_numeric, errors='coerce')
y= y.apply(pd.to_numeric, errors='coerce')


X.fillna(0, inplace=True)
y.fillna(0, inplace=True)


# # Computing and Sketching feature importance 

# In[186]:


from sklearn.linear_model import LinearRegression
plt.figure(figsize=(15,5))
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Feature Importance")
plt.show()


# # Most of our features are not significat so we drop them 

# In[187]:


new_data=data.drop([ 'school', 'sex', 'age', 'address', 'famsize', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'guardian', 'traveltime',
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'romantic', 'famrel', 'goout', 'Dalc',
       'Walc', 'health', 'absences' ] , axis=1)


# In[188]:


new_data.studytime.unique()


# # Lable Encoding 

# In[189]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le_1 = preprocessing.LabelEncoder()
le_2 = preprocessing.LabelEncoder()



new_data.Pstatus= le.fit_transform(new_data.Pstatus)

new_data.reason= le_1.fit_transform(new_data.reason)

new_data.internet= le_2.fit_transform(new_data.internet)

new_data.internet.unique()


# In[190]:


import pickle
pickle.dump(le, open("new_data.Pstatus.pkl","wb"))
pickle.dump(le_1, open("new_data.reason.pkl","wb"))
pickle.dump(le_2, open("new_data.internet.pkl","wb"))


# In[191]:



dfd = new_data.copy()


# In[192]:


dfd['G3'] = dfd['G3'].apply(lambda el: "Good" if el >10 else "Bad")


# In[ ]:





# # Splitting and Normalizing our data

# In[194]:


# dataset train_test_split
from sklearn.model_selection import train_test_split
X = dfd.drop('G3',axis=1)
y = dfd.G3
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[195]:


# from sklearn.preprocessing import MinMaxScaler


X_train_scaled = X_train
X_test_scaled = X_test


# In[ ]:





# In[219]:


# find the optimal # of minimum samples leaf
from sklearn.tree import DecisionTreeClassifier

msl=[]
for i in range(1,58):
    tree = DecisionTreeClassifier(min_samples_leaf=i)
    t= tree.fit(X_train_scaled, y_train)
    ts=t.score(X_test_scaled, y_test)
    msl.append(ts)
msl = pd.Series(msl)
#msl.where(msl==msl.max()).dropna()


# In[198]:



dctm = DecisionTreeClassifier(min_samples_leaf=10)
t= dctm.fit(X_train_scaled , y_train)
print("Decisioin Tree Model Score" , ":" + str( t.score(X_train_scaled, y_train)) , "," , 
      "Cross Validation Score" ,":" + str(t.score(X_test_scaled, y_test)))


# # Plotting ROC 

# In[199]:


from sklearn import  metrics, model_selection, svm

metrics.plot_roc_curve(t, X_test_scaled, y_test)  
plt.show()


# In[ ]:





# In[200]:


from sklearn.svm import SVC
svc= SVC()
s_model = svc.fit(X_train_scaled, y_train)
print("SVC Model Score" , ":" , s_model.score(X_train_scaled, y_train) , "," ,
      "Cross Validation Score" ,":" , s_model.score(X_test_scaled, y_test))


# In[201]:


from sklearn import  metrics, model_selection, svm

metrics.plot_roc_curve(s_model, X_test_scaled, y_test)  
plt.show()


# In[ ]:





# In[202]:


# find a good # of estimators
from sklearn.ensemble import RandomForestClassifier

rfc_1=[]
for i in range(1,58):
    forest = RandomForestClassifier()
    f = forest.fit(X_train_scaled, y_train)
    fs = f.score(X_test_scaled, y_test)
    rfc_1.append(fs)
rfc_1 = pd.Series(rfc_1)
rfc_1.where(rfc_1==rfc_1.max()).dropna()


# #  final model

# In[203]:



rfc_2 = RandomForestClassifier(n_estimators=32, min_samples_leaf=2)
f = rfc_2.fit(X_train_scaled, y_train)
print("Raondom Forest Model Score" , ":"  + str(f.score(X_train_scaled, y_train)) , "," ,
      "Cross Validation Score" ,":" +str( f.score(X_test_scaled, y_test)))


# In[204]:


from sklearn import  metrics, model_selection, svm

metrics.plot_roc_curve(f, X_test_scaled, y_test)  
plt.show()


# In[ ]:





# In[205]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=2)
af = ada.fit(X_train_scaled, y_train)
print("Ada Boost Model Score" , ":" , af.score(X_train_scaled, y_train) , "," ,
      "Cross Validation Score" ,":" , af.score(X_test_scaled, y_test))


# In[206]:


metrics.plot_roc_curve(af, X_test_scaled, y_test)  
plt.show()


# In[208]:


pickle.dump(rfc_2, open("rfc_2.sav","wb"))


# In[ ]:





# In[217]:


model_pkl = pickle.load(open("rfc_2.sav",'rb'))


# In[218]:


model_pkl.predict([[1,1,2,0,0,3,6,18]])


# In[213]:


rfc_2.predict([[1,1,2,0,0,3,6,18]])


# In[ ]:




