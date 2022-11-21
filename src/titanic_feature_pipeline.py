import os
import modal
import hopsworks
import pandas as pd

LOCAL = True
BACKFILL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name(modal_secret_name))
   def f():
       g()

def g():
    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = read_preprocess_data()
    else:
        titanic_df = generate_random_passenger()

    titanic_fg = fs.get_or_create_feature_group(name="titanic_modal",version=1,
        primary_key=['Pclass','Sex','Age_bin','Fare_bin'], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})


def read_preprocess_data():
    '''
    Reads csv file and returns preprocessed dataframe
    '''

    # Read the csv data
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

    # aggregate ages into bins, and map to key
    age_labels = ['child','teenager','adult','elder']
    age_mapping = dict([(age_labels[i], i) for i in range(0, len(age_labels))])

    titanic_df["Age_bin"] = pd.cut(titanic_df["Age"], bins = [0,12,18,55,100], labels=age_labels)
    titanic_df["Age_bin"] = titanic_df["Age_bin"].apply(lambda a: age_mapping[a])

    # aggregate fares into bins, and map to key
    fare_labels = ['low','low_med','high_med','high']
    fare_mapping = dict([(fare_labels[i], i) for i in range(0, len(fare_labels))])

    titanic_df["Fare_bin"] = pd.qcut(titanic_df['Fare'], q=len(fare_labels), labels=['low','low_med','high_med','high'])
    titanic_df["Fare_bin"] = titanic_df["Fare_bin"].apply(lambda a: fare_mapping[a])

    titanic_df = titanic_df.drop(columns=['Age','Fare'])

    # convert all values to int
    titanic_df = titanic_df.astype(int)

def generate_random_passenger():
    '''
    Returns dataframe containing one random passenger
    '''
    # passenger who survived
    survived_df = pd.DataFrame({"Pclass": [np.random.choice([1,2,3],p=[0.5,0.35,0.15])],
    "Sex": [np.random.choice([0,1],p=[0.25,0.75])],
    "Age_bin": [np.random.choice([0,1,2,3],p=[0.25,0.25,0.35,0.15])],
    "Fare_bin": [np.random.choice([0,1,2,3],p=[0.15,0.20,0.30,0.35])],
    "Survived": 1
    })
    # passenger who died
    died_df = pd.DataFrame({"Pclass": [np.random.choice([1,2,3],p=[0.15,0.25,0.60])],
    "Sex": [np.random.choice([0,1],p=[0.75,0.25])],
    "Age_bin": [np.random.choice([0,1,2,3],p=[0.10,0.15,0.55,0.20])],
    "Fare_bin": [np.random.choice([0,1,2,3],p=[0.35,0.30,0.20,0.15])],
    "Survived": 0
    })

    rand = np.random.choice([0,1])
    if rand == 1:
        random_passenger = survived_df
    else:
        random_passenger = died_df
    
    return random_passenger

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()