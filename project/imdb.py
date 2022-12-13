#!/usr/bin/env python
# coding: utf-8

# ## Loading the data

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark import SparkConf
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os


print(f"Here are the keys I found: {os.environ.get('AWS_SECRET_ACCESS_KEY')}")
# Spark setup
# Check if there are keys in the environment, to verify where to container is being run

if len(os.environ.get('AWS_SECRET_ACCESS_KEY')) < 1:

    config = {
        "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
        "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.InstanceProfileCredentialsProvider"
    }
else:
    config = {
        "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1"
    }

conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark = SparkSession.builder.getOrCreate()

BUCKET="dmacademy-course-assets"
KEY_PRE="vlerick/pre_release.csv"
KEY_AFTER="vlerick/after_release.csv"

# Read the files
before = spark.read.csv(f"s3a://{BUCKET}/{KEY_PRE}", header=True).toPandas()
after = spark.read.csv(f"s3a://{BUCKET}/{KEY_AFTER}", header=True).toPandas()

# before = pd.read_csv('/repo/data/pre_release.csv')
# after = pd.read_csv('/repo/data/after_release.csv')


# ## Cleaning the before dataset
# Merge first to remove useless values
df = before.merge(after, how='inner', on='movie_title')
#Also remove duplicates before doing anything else
df.drop_duplicates(inplace=True)
# Only 1048 records remain

to_predict = 'imdb_score'
# Here we will remove the columns we will not use, so we don't remove too much rows
df.drop(['director_name', 'actor_1_name','actor_2_name', 'actor_3_name', 'movie_title'], axis=1, inplace=True)
list_to_remove = list(['num_critic_for_reviews','num_voted_users', 'imdb_score', 'gross', 'movie_facebook_likes', 'num_user_for_reviews'])
list_to_remove.remove(to_predict)
df.drop(list_to_remove, axis=1, inplace=True)

# We will start by eliminating all the null (nan) values
# Replace nan values with 0 for all likes columns
df['director_facebook_likes'].replace(float('nan'), 0, inplace = True)
df['actor_3_facebook_likes'].replace(float('nan'), 0, inplace = True)
df['actor_1_facebook_likes'].replace(float('nan'), 0, inplace = True)
df['cast_total_facebook_likes'].replace(float('nan'), 0, inplace = True)
df['actor_2_facebook_likes'].replace(float('nan'), 0, inplace = True)

df[to_predict].replace(float('nan'), 0, inplace = True)

# Replace content_rating NaN values with 'Not Rated' since NaN probably means the movie doesn't have a content rating
df['content_rating'].replace(float('nan'), "Not Rated", inplace = True)

# A lot of movies have a NaN Language, we will replace this by the most spoken language in the country
df.loc[(df['country'] == "USA") & (df['language'].isnull() == True), 'language'] = "English"

# I want to replace the genre column with a bunch of columns, one for each genre
# Thi column will have a 1 if the movie is of thet certain genre
# First we need to figure out how many genres there are.
genres = dict()
uniques = set()
for full in df['genres']:
    for g in full.split('|'):
        result = genres.get(g)
        uniques.add(g)
        if result is None:
            genres[g] = 1 #First time genre is found
        else:
            genres[g] += 1 #Add an occurence

# We have 23 distinct genres which is a lot. We will combine all genres with less than 100 occurences in a 'Other_genres' column
genres_main = set()

for g in uniques:
    if genres[g] > 250:
        genres_main.add(g)

for g in genres_main:
    df[g] = 0
    df[g] = df[g].astype('int')

df['Other_genre'] = 0
df['Other_genre'] = df['Other_genre'].astype('int')

# Next, for each movie we have to set the value in the specific genre column to 1 for each movie
for i, row in df.iterrows():
    for g in row['genres'].split('|'):
        if g in genres_main:
            df.loc[i, g] = 1
        else:
            df.loc[i, 'Other_genre'] = 1

# Finally we can remove the genre column
df.drop('genres', axis=1, inplace=True)

# Next we apply one-hot encoding to 'country' and 'language'
# But we will put all countries & languages with less than 10 occurences together in an 'other' category

# Get all the countries to replace and to not replace
country_df = df['country'].value_counts().to_frame()
countries_other = set()
countries_main = set()
for i, row in country_df.iterrows():
    if row['country'] < 100:
        countries_other.add(i)
    else:
        countries_main.add(i)
        
#Now we replace all these names in the original dataset with 'Other'
for c in countries_other:
    df['country'].replace(c, 'Other_country', inplace = True)
    
#Add 'Other' to the set of unique countries
countries_main.add('Other_country')

# Create a new column for each unique value
for c in countries_main:
    df[c] = 0
    df[c] = df[c].astype('int')

    # Set the corresponding columns to 1
for i, row in df.iterrows():
    df.loc[i, row['country']] = 1

# Drop the original column
df.drop('country', axis=1, inplace=True)

# Get all the languages to replace and to not replace

languages_df = df['language'].value_counts().to_frame()
languages_other = set()
languages_main = set()
for i, row in languages_df.iterrows():
    if row['language'] < 10:
        languages_other.add(i)
    else:
        languages_main.add(i)

#Now we replace all these names in the original dataset with 'Other'
for l in languages_other:
    df['language'].replace(l, 'Other_language', inplace = True)
    
#Add 'Other' to the set of unique languages
languages_main.add('Other_language')

# Create a new column for each unique value
for l in languages_main:
    df[l] = 0
    df[l] = df[l].astype('int')

# Set the corresponding columns to 1
for i, row in df.iterrows():
    df.loc[i, row['language']] = 1

# Drop the original column
df.drop('language', axis=1, inplace=True)

# Next we will also encode the content rating By lerging them in 4 categories as follows:
# Counts of each ccategory are included between brackets

# Unrated: Not Rated (55), Unrated (17), Approved (7), Passed (2)
# General audiences: G (30)
# Parental guidance: PG-13 (299), PG (154), GP (1)
# Mature: R (473), X (6), NC-17 (3)

# Replace all these values
df['content_rating'].replace('Not Rated', 'Unrated', inplace = True)
df['content_rating'].replace('Approved', 'Unrated', inplace = True)
df['content_rating'].replace('Passed', 'Unrated', inplace = True)

df['content_rating'].replace('G', 'General', inplace = True)

df['content_rating'].replace('PG-13', 'Parental', inplace = True)
df['content_rating'].replace('PG', 'Parental', inplace = True)
df['content_rating'].replace('GP', 'Parental', inplace = True)

df['content_rating'].replace('R', 'Mature', inplace = True)
df['content_rating'].replace('X', 'Mature', inplace = True)
df['content_rating'].replace('NC-17', 'Mature', inplace = True)

df['content_rating'].value_counts()

# Now we will also change this column into dummy columns
ratings = set(['Mature', 'Parental', 'General', 'Unrated'])

# Create a new column for each unique value
for r in ratings:
    df[r] = 0
    df[r] = df[r].astype('int')

# Set the corresponding columns to 1
for i, row in df.iterrows():
    df.loc[i, row['content_rating']] = 1
   
# Drop the original column
df.drop('content_rating', axis=1, inplace=True)

#Here we can see that the actor likes and the total cast likes are quite heavily correlated 
#This is why we will remove the likes of the actors: 
df.drop(['actor_3_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes'], axis=1, inplace=True)

# First scale all the variables using a standard scaler, so we can use them for predictions
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalerY = StandardScaler()

scaling = True

if scaling:
    # scale the X features (not the one-hot encodings)
    df[['duration','director_facebook_likes', 'cast_total_facebook_likes', 'budget']] = scalerX.fit_transform(df[['duration','director_facebook_likes', 'cast_total_facebook_likes', 'budget']])
    # Scale the y variable: use a different scaler to scale back the predictions
    df[to_predict] = scalerY.fit_transform(df[[to_predict]])

# Define the X and y variables
y = df[to_predict]
#df.drop(to_predict, axis=1, inplace=True)
X = df.copy()
X_train, X_test, y_train, y_test = train_test_split(df, df[to_predict], test_size=0.2)
# First we will start with simple linear regression from sklearn, using KFold validtion
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 



# The best model I found in the project
model = RandomForestRegressor(n_estimators=160, criterion="squared_error", max_depth=5)
model.fit(X_train, y_train) # Use the whole dataset to prepare for training

prediction = model.predict(X_test)
real_prediction = scalerY.inverse_transform([prediction]).flatten()
#print("We predict this movie will have an imdb score of ", real_prediction)

real_prediction = pd.DataFrame(real_prediction, columns=["y_pred"])

spark_df = spark.createDataFrame(real_prediction)

#spark_df.write.format('json').json("s3://dmacademy-course-assets/vlerick/GlenMartens/predictions.json")
spark_df.write.option("header",True).json("s3a://dmacademy-course-assets/vlerick/Glen/final_prediction.json")
#spark_df.write.option("header",True).mode(SaveMode.Overwrite).json("s3a://dmacademy-course-assets/vlerick/Glen/predictions.json")
#spark_df.write.option("header",True).json(f"s3a://{BUCKET}/{KEY_PRE}")
print("done")