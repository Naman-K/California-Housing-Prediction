# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:52:45 2019

@author: NamanK
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
import folium

df=pd.read_csv('housing1.csv')

from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
df['ocean_proximity']=number.fit_transform(df['ocean_proximity'].astype('str'))

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x.fillna(x.mean(), inplace=True)
x.info()

from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.15)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)

x_trains = preprocessing.scale(x_train)
x_tests = preprocessing.scale(x_test)
y_trains = preprocessing.scale(y_train)
y_tests = preprocessing.scale(y_test)

from sklearn.svm import SVR
model_svr = SVR(kernel="rbf")
model_svr.fit(x_trains,y_trains.ravel())

y_pred= model_svr.predict(x_tests)
print("Root Mean Squared Error is {}".format(np.sqrt(metrics.mean_squared_error(y_tests,y_pred))))

# calculating the mean of the latitudes 
# and longitudes of the locations of volcanoes 
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
print(map5.save('finally.html'))