# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:57:21 2018

@author: nibir.nath
"""
#Project Big Mart Sales Prediction
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import seaborn as sns
color=sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

bm_traindata = pd.read_csv('Train.csv')
bm_testdata = pd.read_csv('Test.csv')

#Check the number of samples and features
print("The train data size before dropping ID feature is:{}".format(bm_traindata.shape))
#Save the Item Identifier & Outlet Identifier Column:
train_Item_Identifier = bm_traindata['Item_Identifier']
train_outlet_Identifier = bm_traindata['Outlet_Identifier']

#statistical Analysis of Target Variable
sns.distplot(bm_traindata['Item_Outlet_Sales'],fit = norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(bm_traindata['Item_Outlet_Sales'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc = 'best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(bm_traindata['Item_Outlet_Sales'], plot=plt)
plt.show()

#Detecting OUtliers
fig,ax = plt.subplots()
ax.scatter(x = bm_traindata['Item_Weight'],y = bm_traindata['Item_Outlet_Sales'])
plt.ylabel('Outlet sale',fontsize = 12)
plt.xlabel('weight',fontsize=12)
plt.show()

fig,ax = plt.subplots()
ax.scatter(x = bm_traindata['Item_Visibility'],y = bm_traindata['Item_Outlet_Sales'])
plt.ylabel('outlet sales',fontsize = 12)
plt.xlabel('visibility',fontsize=12)
plt.show()

fig,ax = plt.subplots()
ax.scatter(x = bm_traindata['Item_MRP'],y = bm_traindata['Item_Outlet_Sales'])
plt.ylabel('outlet sales',fontsize = 12)
plt.xlabel('MRP',fontsize=12)
plt.show()

plt.boxplot(bm_traindata['Item_Outlet_Sales'])


ntrain=bm_traindata.shape[0]
ntest=bm_testdata.shape[0]


all_data = pd.concat((bm_traindata,bm_testdata)).reset_index(drop = True)
#all_data.drop(['Item_Identifier','Outlet_Identifier'],axis = 1,inplace = True)
print("all data size is : {}",format(all_data.shape))
type(all_data)

#Treating Missing Values
all_data_na=(all_data.isnull().sum()/len(all_data))*100

missing_data=pd.DataFrame({'Missing Data':all_data_na})
missing_data.head(20)

#Imputing Missing values
all_data["Item_Weight"] = all_data["Item_Weight"].transform(lambda x:x.fillna(x.median()))
all_data["Outlet_Size"] = all_data["Outlet_Size"].fillna('Small')
all_data["Item_Outlet_Sales"] = all_data["Item_Outlet_Sales"].transform(lambda x:x.fillna(x.median()))
#Percentage Missing values after Imputation
all_data_na_modified=(all_data.isnull().sum()/len(all_data))*100

missing_data_new=pd.DataFrame({'Missing Data':all_data_na_modified})
missing_data_new.head(20)

#Features Engineering
all_data['Item_Type_combined']=all_data['Item_Identifier'].apply(lambda x:x[0:2]) 
#Rename them to more meaningful category
all_data['Item_Type_combined']=all_data['Item_Type_combined'].map({'FD':'Food','DR':'Drinks','NC':'Non-consumable'}) 
all_data['Item_Type_combined'].value_counts()

all_data['outlet_years'] = 2013-all_data['Outlet_Establishment_Year']

#Renaming some categorical variables
all_data['Item_Fat_Content']=all_data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
all_data['Item_Fat_Content'].value_counts()

all_data.loc[all_data['Item_Type_combined']=='Non-consumable','Item_Fat_Content']='Non-edible'

#Transforming Categorical Variable to Numerical
# Encoding categorical data
# Encoding the Independent Variable
type(all_data['Item_Fat_Content'])

cleanup_nums={'Item_Fat_Content':{'Non-edible':1,'Regular':2,'Low Fat':3},
              'Outlet_Location_Type':{'Tier 1':1,'Tier 2':2,'Tier 3':3},
              'Outlet_Size':{'High':1,'Medium':2,'Small':3},
              'Outlet_Type':{'Grocery Store':1,'Supermarket Type2':2,'Supermarket Type3':3,'Supermarket Type1':4}}

all_data.replace(cleanup_nums,inplace=True)
from sklearn.preprocessing import LabelEncoder

lbl=LabelEncoder()
lbl.fit(all_data['Item_Type_combined'].values)
all_data['Item_Type_combined']=lbl.transform(all_data['Item_Type_combined'].values)
    
print('The shape of data is: {}'.format(all_data.shape))
##I need to start from here##
#Seperation of variable
y_val = all_data['Item_Outlet_Sales'].values
all_data.drop(['Item_Outlet_Sales'],axis = 1,inplace = True)
all_data.drop(['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier','Item_Type'],axis = 1,inplace = True)

#Splitting into training and testing data
bm_traindata = all_data[:ntrain]
bm_testdata = all_data[ntrain:]
y_train = y_val[:ntrain]
y_test = y_val[ntrain:]

#Scaling of Variables
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#bm_traindata1 = sc_X.fit_transform(bm_traindata)
#sc_x = StandardScaler()
#bm_testdata1 = sc_x.fit_transform(bm_testdata)
#sc_Y = StandardScaler()
#y_train1 = sc_Y.fit_transform(y_train.reshape(-1,1))
#sc_y = StandardScaler()
#y_test1 = sc_y.fit_transform(y_test.reshape(-1,1))

# Fitting Multiple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(bm_traindata1, y_train1)
#regressor.coef_
#regressor.score(bm_traindata1, y_train1)
# Predicting the Test set results
#y_pred = regressor.predict(bm_testdata1)

#Fitting Decision tree model to the training set
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state = 0)
regressor1.fit(bm_traindata, y_train)
regressor1.score(bm_traindata, y_train)
y_pred1 = regressor1.predict(bm_testdata)

# Create a Pandas dataframe from the data.
df = pd.DataFrame(y_pred1)
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Bigmart_solution1.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

writer.save()

# Fitting Random Forest Regression to the dataset
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#regressor.fit(bm_traindata, y_train)
#regressor.score(bm_traindata, y_train)
#y_pred = regressor.predict(bm_testdata)
#from sklearn.preprocessing import StandardScaler
#scalery = StandardScaler().fit(y_pred.reshape(-1,1))
#y_new_inverse = scalery.inverse_transform(y_pred.reshape(-1,1))
#type(y_pred)
# Create a Pandas dataframe from the data.
#df = pd.DataFrame(y_pred)
# Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter('Bigmart_solution.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
#df.to_excel(writer, sheet_name='Sheet1')

#writer.save()



