# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/DrishVij/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

dataset = df;

# Drop the unique identifier
dataset.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore', inplace=True)

# Define the target variable for the classification task
target_col = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = ['Age',
                    'CityTier',
                    'DurationOfPitch',
                    'NumberOfPersonVisiting',
                    'NumberOfFollowups',
                    'PreferredPropertyStar',
                    'NumberOfTrips',
                    'Passport',
                    'PitchSatisfactionScore',
                    'NumberOfChildrenVisiting',
                    'MonthlyIncome']

# List of categorical features in the dataset
categorical_features = ['TypeofContact',
                        'Occupation',
                        'Gender',
                        'OwnCar',
                        'ProductPitched',
                        'MaritalStatus',
                        'Designation'
                        ]

# Define predictor matrix (X) using selected numeric and categorical features
X = dataset[numeric_features + categorical_features]
# Define target variable
y = dataset[target_col]


# Split into X (features) and y (target)
#X = dataset.drop(columns=[target_col])
#y = dataset[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="DrishVij/Tourism-Package-Prediction",
        repo_type="dataset",
    )
