# CapstoneProject_NathaliaDaniela
Repository for DS Capstone Project of Nathalia and Daniela

The present project aims to predict the number of bikes using Bicing real data.

The final result is the percentage of free docks for each of the proposed stations given historical data.

The repository is composed by 5 notebooks. Each notebook contains a different step to complete the present study, as follows:

# 01_DF_creations_Station&Bicing_Kaggle_info

All extraction and tranformation data were made in this first notebook. 

The Bicing station information was extracted from: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/informacio-estacions-bicing
The Bicing station historical was extracted from: https://opendata-ajuntament.barcelona.cat/data/ca/dataset/estat-estacions-bicing/resource/84c0d6e5-9011-40c2-80a7-65d9af4f671f?inner_span=True

* Station info challenges:

- The data was imported as JSON file and it required to be accessed by selecting the df_json['data']['stations'] dictionary. 

- From the full data frame, the team decided to keep the following columns for data analysis: ['station_id', 'name', 'lat', 'lon', 'altitude', 'address', 'post_code', 'capacity']

-  It was noted that 'address' and 'name' were basically the same data, only formatting distinguished one from another. Thus, the team decided to keep only 'name' column and drop 'address'column. 

- The'post_code' column had to be treated to maintain a unique formatting. 

Comments: this first data set presented little challenges to handle, it was interested to learn how to import JSON files from web and access their dictionaries. Although, the columns selection required some time to analyze and discuss within the team. 


* Station historical challenges:
