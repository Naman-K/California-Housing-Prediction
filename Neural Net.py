# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:23:47 2019

@author: NamanK
"""
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
import folium
from keras.layers import Dense
from keras.models import Sequential
df=pd.read_csv('housing1.csv')

from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
df['ocean_proximity']=number.fit_transform(df['ocean_proximity'].astype('str'))

df['num_rooms'] = df['total_rooms'] / df['households']

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x.fillna(x.mean(), inplace=True)
x.info()
x.describe()

from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.15)

x_norm = (x-x.mean())/x.std()
n_cols = x_norm.shape[1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_norm,y,test_size=0.17)

def my_model():
    model = Sequential()
    model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

model = my_model()
model.fit(x_train,y_train,validation_split=0.17,verbose=2,epochs=49)
    
y_pred= model.predict(x_test)
print("Root Mean Squared Error is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))

# calculating the mean of the latitudes 
# and longitudes of the locations of houses
latmean=df['latitude'].mean() 
lonmean=df['longitude'].mean() 
  
# Creating a map object using Map() function. 
# Location parameter takes latitudes and 
# longitudes as starting location. 
# (Map will be centered at those co-ordinates)  
map5 = folium.Map(location=[latmean,lonmean], 
        zoom_start=6,tiles = 'Mapbox bright') 
          
# Function to change the marker color  
# according to the elevation of volcano 
def color(value): 
    if value in range(0,149999): 
        col = 'green'
    elif value in range(150000,249999): 
        col = 'yellow'
    elif value in range(250000,349999): 
        col = 'orange'
    else: 
        col='red'
    return col 
      
# Iterating over longitude and latitude with median_house_value
for lat,lan,value in zip(df_test['latitude'],df_test['longitude'],df_test['median_house_value']): 
    # Marker() takes location coordinates  
    # as a list as an argument 
    folium.Marker(location=[lat,lan], 
                  icon= folium.Icon(color=color(value), 
                  icon_color='black',icon = 'home')).add_to(map5) 
                    
# Save the file created above 
print(map5.save('finalnn.html'))