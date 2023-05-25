#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from datetime import datetime 

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf


# In[2]:


df = pd.read_feather('../../../jmarchezi/research-projects/solar-wind-data-with-gaps/outputData/combined_ace_data_resampled.feather')


# In[3]:


pd.to_datetime(df['ACEepoch'], format='%Y-%m-%d %H:%M:%S')
df.set_index('ACEepoch', inplace=True, drop=True)
df.index = pd.to_datetime(df.index)

df.dropna(inplace=True)
df.drop(['YR', 'Month', 'Day', 'HR'], axis=1, inplace=True)


# In[4]:


time_diff = df.index.to_series().diff()
mask = (time_diff.shift(-1) == pd.Timedelta(minutes=1)) & (time_diff == pd.Timedelta(minutes=1))

print(mask.sum())

# Find consecutive True values in the mask
consecutive_true = mask.cumsum()
mask = mask & (consecutive_true % 2 == 1)

print(mask.sum())

num_true = mask.sum()
max_samples = int(len(df) * 0.2)  # 20% of the total number of rows

# If the number of True values exceeds the maximum samples, randomly sample from the True indices
if num_true > max_samples:
    sample_indices = mask[mask].sample(n=max_samples, random_state=42).index
    mask = mask.index.isin(sample_indices)

samples_to_nan = df[mask]


# In[5]:


test_vx_nans = df['Vx'].copy()
test_vx_nans.loc[test_vx_nans.index.isin(samples_to_nan.index)] = np.nan
print(df)


# In[6]:


# Check for consecutive rows with NaN values
consecutive_nan = test_vx_nans.isnull() & test_vx_nans.shift().isnull()

# Check if any consecutive NaN rows exist
if consecutive_nan.any():
    print("Consecutive NaN rows exist.")
else:
    print("No consecutive NaN rows.")


# In[7]:


linear_interp = test_vx_nans.interpolate(method='linear')
test_vx = pd.DataFrame({'real':df['Vx'].copy(),
						'nans':test_vx_nans,
						'linear_interp':linear_interp})


# In[8]:


test_vx = test_vx.loc[test_vx.index.isin(samples_to_nan.index)]
test_vx


# In[9]:


training_data = df[~df.index.isin(samples_to_nan.index)]
testing_data = df[df.index.isin(samples_to_nan.index)]


# In[10]:


X_train = np.array(training_data.drop('Vx', axis=1).reset_index(drop=True, inplace=False))
y_train = np.array(training_data['Vx'].reset_index(drop=True, inplace=False))
X_test = np.array(testing_data.drop('Vx', axis=1).reset_index(drop=True, inplace=False))
y_test = np.array(testing_data['Vx'].reset_index(drop=True, inplace=False))


# In[11]:


'''Model training and predicting section. These will be the SK learn based models such as linear regression, tree-based models, etc.'''


# In[12]:


# First is linear regression
linear_reg = LinearRegression().fit(X_train, y_train)
LR_test = linear_reg.predict(X_test)
test_vx['linear_regression'] = LR_test


# In[13]:


# Next is K-Nearest Neighbors. Here will do a single one and an ensamble
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
knn_test = knn.predict(X_test)
test_vx['knn'] = knn_test


# In[14]:


# the knn ensamble where we change the number of neighbors
knn2 = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn2.fit(X_train, y_train)
knn2_test = knn2.predict(X_test)

knn4 = KNeighborsRegressor(n_neighbors=4, weights='distance')
knn4.fit(X_train, y_train)
knn4_test = knn4.predict(X_test)

knn6 = KNeighborsRegressor(n_neighbors=6, weights='distance')
knn6.fit(X_train, y_train)
knn6_test = knn6.predict(X_test)

knn8 = KNeighborsRegressor(n_neighbors=8, weights='distance')
knn8.fit(X_train, y_train)
knn8_test = knn8.predict(X_test)

knn10 = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn10.fit(X_train, y_train)
knn10_test = knn10.predict(X_test)

knn_ensamble = np.mean([knn2_test, knn4_test, knn6_test, knn8_test, knn10_test], axis=0)

test_vx['knn_ensamble'] = knn_ensamble


# In[15]:


# Next is Decision Tree. Here will do a single one and an ensamble
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
tree_test = tree.predict(X_test)
test_vx['tree'] = tree_test


# In[16]:


# the decision tree ensamble where we change the random state
tree0 = DecisionTreeRegressor(random_state=1)
tree0.fit(X_train, y_train)
tree0_test = tree0.predict(X_test)

tree1 = DecisionTreeRegressor(random_state=10)
tree1.fit(X_train, y_train)
tree1_test = tree1.predict(X_test)

tree2 = DecisionTreeRegressor(random_state=100)
tree2.fit(X_train, y_train)
tree2_test = tree2.predict(X_test)

tree3 = DecisionTreeRegressor(random_state=1000)
tree3.fit(X_train, y_train)
tree3_test = tree3.predict(X_test)

tree4 = DecisionTreeRegressor(random_state=10000)
tree4.fit(X_train, y_train)
tree4_test = tree4.predict(X_test)

tree_ensamble = np.mean([tree0_test, tree1_test, tree2_test, tree3_test, tree4_test], axis=0)

test_vx['tree_ensamble'] = tree_ensamble


# In[17]:


test_vx


# In[31]:


# Random Forest time. Here will do a single one and an ensamble
forest = RandomForestRegressor(verbose=1, n_estimators=100, random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
forest_test = forest.predict(X_test)
test_vx['forest'] = forest_test


# In[70]:


ann_xtrain, ann_xval, ann_ytrain, ann_yval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(ann_xtrain)
ann_xtrain = scaler.transform(ann_xtrain)
ann_xval = scaler.transform(ann_xval)
ann_xtest = scaler.transform(X_test)


# In[71]:


model = Sequential()						# initalizing the model

model.add(Dense(100, activation='relu', input_shape=(ann_xtrain.shape[1],)))		# Adding dense layers with dropout in between
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)		# learning rate that actually started producing good results
model.compile(optimizer=opt, loss='mse')					# Ive read that cross entropy is good for this type of model
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)		# early stop process prevents overfitting

model.fit(ann_xtrain, ann_ytrain, validation_data=(ann_xval, ann_yval),
					verbose=1, shuffle=True, epochs=500, callbacks=[early_stop], batch_size=1024)

ann_test = model.predict(ann_xtest)

test_vx['ANN'] = ann_test


# In[19]:


# # the random forest ensamble where we change the random state and the number of estimators
# forest0 = RandomForestRegressor(verbose=1, n_estimators=10, random_state=1)
# forest0.fit(X_train, y_train)
# forest0_test = forest0.predict(X_test)

# forest1 = RandomForestRegressor(verbose=1, n_estimators=50, random_state=10)
# forest1.fit(X_train, y_train)
# forest1_test = forest1.predict(X_test)

# forest2 = RandomForestRegressor(verbose=1, n_estimators=100, random_state=100)
# forest2.fit(X_train, y_train)
# forest2_test = forest2.predict(X_test)

# forest3 = RandomForestRegressor(verbose=1, n_estimators=500, random_state=1000)
# forest3.fit(X_train, y_train)
# forest3_test = forest3.predict(X_test)

# forest4 = RandomForestRegressor(verbose=1, n_estimators=1000, random_state=10000)
# forest4.fit(X_train, y_train)
# forest4_test = forest4.predict(X_test)

# forest_ensamble = np.mean([forest0_test, forest1_test, forest2_test, forest3_test, forest4_test], axis=0)

# test_vx['forest_ensamble'] = forest_ensamble


# In[65]:


test_vx


# In[62]:


error_list = []
y_true = test_vx['real']
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['linear_interp'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['linear_regression'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['knn'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['knn_ensamble'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['tree'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['tree_ensamble'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['forest'])))
error_list.append(np.sqrt(mean_squared_error(y_true, test_vx['ANN'])))


# In[63]:


fig = plt.figure(figsize=(10,5))
x = [i for i in range(len(error_list))]
x_labels = ['Linear Interpolation', 'Linear Regression', 'KNN', 'KNN Ensemble', 'Decision Tree', 'DT Ensemble', 'Random Forest', 'ANN']
plt.scatter(x, error_list)
plt.xticks(x, x_labels)
plt.ylabel('RMSE')
plt.title('1 minute gaps in Vx')
plt.show()


# In[64]:


print(error_list)


# In[42]:


testing_data


# In[45]:


param = 'T'

error_df = pd.DataFrame()
error_df['real'] = testing_data[param]
error_df['linear_interp'] = y_true - test_vx['linear_interp']
error_df['linear_regression'] = y_true - test_vx['linear_regression']
error_df['knn'] = y_true - test_vx['knn']
error_df['knn_ensamble'] = y_true - test_vx['knn_ensamble']
error_df['tree'] = y_true - test_vx['tree']
error_df['tree_ensamble'] = y_true - test_vx['tree_ensamble']
error_df['forest'] = y_true - test_vx['forest']


# In[46]:


fig = plt.figure(figsize=(20,15))

ax0 = plt.subplot(341)
plt.scatter(error_df['real'], error_df['linear_interp'], color='blue')
plt.title('Linear Interp.')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

ax1 = plt.subplot(342)
plt.scatter(error_df['real'], error_df['linear_regression'], color='orange')
plt.title('Linear Regression')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

ax2 = plt.subplot(343)
plt.scatter(error_df['real'], error_df['knn'], color='green')
plt.title('KNN')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

ax3 = plt.subplot(344)
plt.scatter(error_df['real'], error_df['knn_ensamble'], color='black')
plt.title('KNN Ensemble')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

ax4 = plt.subplot(345)
plt.scatter(error_df['real'], error_df['tree'], color='purple')
plt.title('Decision Tree')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

ax5 = plt.subplot(346)
plt.scatter(error_df['real'], error_df['tree_ensamble'], color='red')
plt.title('DT Ensemble')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

ax6 = plt.subplot(347)
plt.scatter(error_df['real'], error_df['forest'], color='brown')
plt.title('Random Forest')
plt.axhline(0, linestyle='--')
plt.ylabel('Difference')
plt.xlabel(param)

plt.show()


# In[ ]:




