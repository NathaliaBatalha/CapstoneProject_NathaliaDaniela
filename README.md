# CapstoneProject_NathaliaDaniela
Repository for DS Capstone Project of Nathalia and Daniela

The present project aims to predict the number of bikes using Bicing real data.

The final result is the percentage of free docks for each of the proposed stations given historical data.

The repository is composed by 5 notebooks. Each notebook contains a different step to complete the present study, as follows:

# 01_DF_creations_Station&Bicing_Kaggle_info

All extraction and tranformation data are done in this notebook. 

Required libraries:

	import os
	import json
	import urllib
	import pandas as pd
	import numpy as np
	from pandas import Timestamp, Series, date_range
	from datetime import datetime
	import lzma

The Bicing station information was extracted from: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/informacio-estacions-bicing

The Bicing station historical was extracted from: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/estat-estacions-bicing/resource/84c0d6e5-9011-40c2-80a7-65d9af4f671f?inner_span=True

## Station info

- The data is imported as JSON file and it requires to be accessed by selecting the df_json['data']['stations'] dictionary. 

- From the full data frame, the team decided to keep the following columns for data analysis: ['station_id', 'name', 'lat', 'lon', 'altitude', 'address', 'post_code', 'capacity']

-  It is noted that 'address' and 'name' are basically the same data, only formatting distinguished one from another. Thus, the team decided to keep only 'name' column and drop 'address' column. 

- The'post_code' column had to be treated to maintain a unique formatting. 

Comments: this first data set presented little challenges to handle. Although the columns selection required some time to analyze and discuss within the team. 


## Station historical challenges:

- Historical data from 2019-2022 are downloaded from website. A couple of files are ranemed, once they do not follow the name pattern.

- After data analysis, the team decided to keep the following columns for data set: ['station_id','num_docks_available','status','last_reported']

- All data are treated in a loop over months and years, and concatenated in one fine and large data frame. The final data frame is mainly done by: 

		- filtering the variables of interest;
		- computing only 'in service' data;
		- merging station info data to each row of bicing historical data;
		- transforming data/time columns to splitted to day, hour, month, and year columns;  
		- computing and normalizing dock availability based on data;
		- creating contexts columns: dock availability in a sequency of every 4 hours. 

## Data from Kaggle

The data set to be predicted is also treated in this notebook. Station data inspection, and formatting adjustments are made in the step.

## Weather data

Wind, precipitation and teperature historicals were obtained and explored in this part. The data will be available in case is useful for machine learning. 

After processing, all data frames are saved in a shared drive to be accesed by the team and from other notebooks. 


# 02_Visualization_Data_partI

Required libraries:

	import os
	import pandas as pd
	import numpy as np
	import matplotlib 
	from matplotlib.dates import MonthLocator
	import seaborn as sns


The notebook, being the first visualization part, presents line plots from data set. The following graphs can be accessed: 

- Timestamp - Docks available by year

- Percentage of docks available by month

- Percentage of docks available by hour 

- Percentage of docks available by day of the week

- Percentage of docks available by day of the week and hour

- Percentage of docks available at different times of the day: by month and day of the week

These plots showed us that the 2019 and 2020 data are very different from the current ones.
2019 was the year of implementation of the new Bicing system and 2020 the year of the most intense lockdowns due to the pandemic, so we decided to disregard this data in our training.

We can also see in these plots:
- difference in usage on weekdays and weekends
- schedules with greater and lesser use of bicycles

All these insights were useful in building our machine learning models.


# 02_Visualization_Data_partII

Intallation:

	!pip install geopandas 
	!pip install contextily 


Required libraries:

	import os
	import pandas as pd
	import numpy as np
	import matplotlib 
	from matplotlib.dates import MonthLocator
	import seaborn as sns
	import geopandas as gpd
	import contextily as cx


The second part of the visualization exploration refers to geographic plots, using GeoPandas library. 

The plots present useful information regarding station location x altitudes throughout the city. Also, the hourly dock availability in each station may also be explored in this notebook. 


# 03_Models_v6

Required libraries:

	import os
	import pandas as pd
	import numpy as np

	#sklearn
	from sklearn.impute import SimpleImputer
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import FunctionTransformer
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.model_selection import train_test_split
	from sklearn.compose import ColumnTransformer

	#models
	from sklearn.linear_model import LinearRegression
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
	from sklearn import neighbors
	from sklearn.neural_network import MLPRegressor

	from sklearn import set_config
	from sklearn.metrics import mean_squared_error
	from sklearn.model_selection import GridSearchCV

	import matplotlib as mpl
	import matplotlib.pyplot as plt

	import seaborn as sns



This notebook is dedicated to implement Regression Models for Machine Learning. The following models were explored, evaluated, improved, and analyzed based on data correlation and results. 

- LinearRegression()

- DecisionTreeRegressor()

- RandomForestRegressor()

- MLPRegressor()

- GradientBoostingRegressor()

- KNeighborsRegressor()


Training and tests results (diverse variables and parameters were considered) presented that the best Regression Model for the present study is RandomForestRegressor. 

From all the training process, it was learnt:

- Contexts present higher correlation, and are the essential variables 
- Categorical variables do not present significant impact on learning improvement, considering one hot encodding for them.
- Grid Search Estimator to find the best parameters was used, which provided supperior results.  


# 04_Clusters_stations

Required libraries:

	import os
	import pandas as pd
	import numpy as np
	import folium

	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import matplotlib.dates as md

	import seaborn as sns

	import sklearn
	from sklearn.cluster import KMeans
	from sklearn import metrics

This notebook was inspired by this "Dublin Bikes cluster analysis" notebook:
https://github.com/jameslawlor/dublin-bikes-timeseries-analysis/blob/master/dublin-bikes-time-series-clustering-and-mapping.ipynb

We used the k-means algorithm to divide stations with similar daily behaviors. Our intention was to train a model using the clusters instead of all stations.

We also plotted a map with Folium with stations divided by color, representing each cluster.



# Embedding_03_Models_tensorflow_2021_2022

Required libraries:

to complete


Once categorical variables do not present significant effect on learning process, the group decided to try Deep Learning model to verify embedding impact on categorical variables. 

Quick note: Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. 

The Keras functional API was used to build the model, because it is more flexible than the keras.Sequential API. The main reason is the flexibility of using multiple inputs on the model. 

Categorical variables need to be selected, quantified, states dimensions of its tensors and apply Embedding layer. 
