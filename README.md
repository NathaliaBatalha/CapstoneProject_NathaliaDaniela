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

To be completed!

After processing, all data frames are saved in a shared drive to be accesed by the team and from other notebooks. 


# 02_Visualization_Data_partI



# 02_Visualization_Data_partII
