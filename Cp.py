
-*- coding: utf-8 -*-
# Multiple Linear Regression
# Importing the librariesi
importinumpy as np
24 | P a g e
importimatplotlib.pyplot as plt
importipandas as pd
fromipandas import Series, DataFrame
# Importini the dataset
dataseti= pd.read_csv('rape.csv') 
X =idataset.iloc[:, [0,1,2,3]].values
y =idataset.iloc[:,[4,5,6,7,8,9,10]].values
# Encodingicategorical data
‘from sklearn.preprocessing import LabelEncoder, OneHotEncode’
labelencoder_X_0 = iLabelEncoder()
X[:, 0] =labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_1i=iLabelEncoder()
X[:, 2] =labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder =iOneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder =iOneHotEncoder(categorical_features = [2])
X =ionehotencoder.fit_transform(X).toarray()
#takingicare ofimissing data
'''fromi sklearn.preprocessingi import iImputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,[0,2]])
X[:,[0,2]]= iimputer.transform(X[:,[0,2]])
'''
#Avoidingithe DummyiVariableiTrap
X =X[:, 1:]
#iSplitting the dataset intoitheiTrainingiset and Testiset
from sklearn.cross_validation import train_test_split
“X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)”
# FeatureiScaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#iFittingiMultipleiLineariRegression to theiTraining set
fromisklearn.linearimodel impori LinearRegression
regressor =iLinearRegression()
regressor.fit(X_train, y_train) i
# Predicting the Test set results with multiple linear regression
y_pred_MultiplelinearRegression = regressor.predict(X_test)
#calculating mseifor linear regression
mse_linearRegression =inp.mean((y_pred_MultiplelinearRegression - y_test)**2)
#calculating r-square for multiple linear regression
regressor.score(X_test,y_test) i
#fitting ridgeiregression
“from sklearn.linear_model import Ridge
## training the model
ridgeReg =iRidge(alpha=0.05, normalize=True)
ridgeReg.fit(X_train,y_train)”
y_pred_RidgeRegression = ridgeReg.predict(X_test)
#calculatingimse for Ridge linear regression
mse_ridgeRegression = np.mean((y_pred_RidgeRegression - y_test)**2)
#calculating r-square for ridge regression
ridgeReg.score(X_test,y_test)
#fitting lassoiregression
”from sklearn.linear_model import Lasso
lassoReg =i Lasso(alpha=0.3, normalize=True)
ilassoReg.fit(X_train,y_train)
y_pred_LassoRegression =ilassoReg.predict(X_test)
#calculating mseifor Lasso regression
mse_lassoRegression =inp.mean((y_pred_LassoRegression - y_test)**2)”
#calculating r-square for lasso regression
lassoReg.score(X_test,y_test) i
#fitting elastic net regression
from sklearn.linear_model import ElasticNeti
iENreg =iElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
iENreg.fit(X_train,y_train)
y_pred_ElasticNet = ENreg.predict(X_test)
#calculating mse for ElasticNet linear regression
28 | P a g e
mse_ElasticNet = np.mean((y_pred_ElasticNet - y_test)**2)
#calculating r-square for elastic net regression
ENreg.score(X_test,y_test)
‘”# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor’
regressor_tree =iDecisionTreeRegressor(random_state = 0)
iregressor_tree.fit(X_train, y_train)”
#iPredicting ainewiresult
y_pred_Dtree =iregressor_tree.predict(X_test)
#calculating mse for DecisionTree regression
mse_Dtree = np.mean((y_pred_Dtree - y_test)**2)
#calculating r-square for decisiontree regression
regressor_tree.score(X_test,y_test) 
#randomiforest
“# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor”
“regressor_randomForest = RandomForestRegressor(n_estimators = 15, random_state = 0)
regressor_randomForest.fit(X_train, y_train)
# Predicting a new result”
y_pred_randomForest = regressor_randomForest.predict(X_test)
#calculating mse for RandomForest regression
mse_randomForest = np.mean((y_pred_randomForest - y_test)**2)
#calculating r-square for random forest regression
regressor_randomForest.score(X_test,y_test)
#for andhrapradesh
total_crime_ap = dataset[dataset['Subgroup'] == 'Total Rape Victims']
ap = total_crime_ap[total_crime_ap['Area_Name'] == 'Andhra Pradesh']
ap.set_index('Year')[['Victims_Above_50_Yrs','Victims_Between_10-14_Yrs', 'Victims_Between_14-
18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-
50_Yrs','Victims_Upto_10_Yrs']].plot(kind = 'line', figsize = (15,15))
plt.xlabel('Years')
plt.ylabel('No. of Cases in Andhra Pradesh')
plt.title('Crime against Women in Andhra Pradesh')
plt.show()
#for odisha
total_crime_ap = dataset[dataset['Subgroup'] == 'Total Rape Victims']
ap = total_crime_ap[total_crime_ap['Area_Name'] == 'Odisha']
ap.set_index('Year')[['Victims_Above_50_Yrs','Victims_Between_10-14_Yrs', 'Victims_Between_14-
18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-
50_Yrs','Victims_Upto_10_Yrs']].plot(kind = 'line', figsize = (15,15))
plt.xlabel('Years')
plt.ylabel('No. of Cases in odisha')
plt.title('Crime against Women in odisha')
#for year 2010 
data24 = dataset[(dataset['Subgroup'] == 'Total Rape Victims') & (dataset['Year'] == 2010)]
allstates24 = data24[['Victims_Above_50_Yrs','Victims_Between_10-14_Yrs', 'Victims_Between_14-
18_Yrs','Victims_Between_18-30_Yrs','Victims_Between_30-
50_Yrs','Victims_Upto_10_Yrs']].plot(kind = 'barh', figsize = (20,20), width = 1)
allstates24.set_xlabel('No. of Cases in 2010')
allstates24.set_yticklabels(data24['Area_Name'])
plt.show()
