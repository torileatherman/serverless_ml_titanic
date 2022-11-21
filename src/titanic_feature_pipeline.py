import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

# Preprocessing

# Drop columns that have low predictive power
titanic_df = titanic_df.drop(columns=["Name", "PassengerId", "Ticket", "Cabin","Embarked","SibSp","Parch"])
# fill missing values with mean column values for int and float columns
titanic_df.fillna(titanic_df.median(), inplace=True)
# fill missing values with mode column values for object columns
titanic_df.fillna(titanic_df.mode().iloc[0], inplace=True)
# encode sex
titanic_df["Sex"] = titanic_df["Sex"].apply(lambda x: 0 if x == "male" else 1)
# aggregate ages into bins
titanic_df["Age_bin"] = pd.cut(titanic_df["Age"], bins = [0,12,18,55,100], labels=['child','teenager','adult','elder'])
# aggregate fares into bins
titanic_df["Fare_bin"] = pd.qcut(titanic_df['Fare'], q=4, labels=['low','low_med','high_med','high'])

titanic_df = titanic_df.drop(columns=['Age','Fare'])

titanic_fg = fs.get_or_create_feature_group(name="titanic_modal",version=1,
    primary_key=['survived','pclass','sex','age_bin','fare_bin'], 
    description="Titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})