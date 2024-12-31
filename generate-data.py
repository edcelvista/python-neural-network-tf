#########################################################
# Author: Edcel Vista                                   #
#########################################################

import os, re
import pandas as pd # We need Pandas for data manipulation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder #  Sklearn libraries are used for machine learning operations
import matplotlib.pyplot as plt # MatplotLib, and Seaborn for visualizations
import seaborn as sns
from dotenv import load_dotenv
from utils import printFlush, print_progress_bar, cleanUp

# Load environment variables from .env file
load_dotenv()

def _loadReadData():
    cleanUp([os.getenv("inputStagePath"), os.getenv("inputStageStatsPath"), os.getenv("outputModelfile")])

    # Example CSV file path (replace with your file path)
    csv_file = os.getenv("inputFilePath")
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # Display the first few rows of the DataFrame

    df = _handleMissingData(df, True)
    df = _removeDuplicates(df, "review")
    df = _removeSpecialCharacters(df, "review")
    # df = _handleOutliers(df, "column name") # for numeric data > Outliers are data points that significantly differ from other observations in a dataset. Addressing outliers is crucial for maintaining data integrity and model performance.
    # df = _featureScaling(df, "column name") # for numeric data > Scaling numerical features to ensure they have a consistent range, helps algorithms converge faster and prevents features with larger scales from dominating.
    df = _labelEncode(df, "sentiment") # convert string labels into int label data

    if not df.empty:
        printFlush(df.head())
        # printFlush(df.describe())
        printFlush(f"Rows: {df.shape[0]} Columns: {df.shape[1]}")
        generateStagefile(df, os.getenv("inputStagePath"))
        checkDataDistribution(df, "sentiment_encoded")
    else:
        printFlush(f"No data input.")

def _handleMissingData(df, isDropMissing = True):
    # Assuming 'df' is your DataFrame
    # Detect missing values
    missing_values = df.isnull()  # or df.notnull() to find non-missing values
    # Count missing values per column
    missing_counts = df.isnull().sum()
    # Print missing values counts
    printFlush("Missing Values Count:")
    printFlush(missing_counts)

    if isDropMissing:
        # Drop rows with any missing values
        df_cleaned = df.dropna()
        # Drop columns with any missing values
        # df_dropna_cols = df.dropna(axis=1)
    else: # FOR-LATER-SUPPORT
        # Impute missing values with mean
        df_cleaned = df.fillna(df.mean())
    
    return df_cleaned

def _removeDuplicates(df, targetCols = False):
    if targetCols:
        # Drop duplicates based on specific columns
        df_unique = df.drop_duplicates(subset=[f"{targetCols}"])
    else:
        # Drop duplicates based on all columns
        df_unique = df.drop_duplicates()

    return df_unique
    
def _removeSpecialCharacters(df, targetCols = False):
    # Apply sanitization to the column
    df[f"{targetCols}"] = df[f"{targetCols}"].apply(_cleanText)
    return df

def _handleOutliers(df, colName): # for numeric data
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[f"{colName}"].quantile(0.25)
    Q3 = df[f"{colName}"].quantile(0.75)
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers based on the bounds
    outliers = df[(df[f"{colName}"] < lower_bound) | (df[f"{colName}"] > upper_bound)]
    # Remove outliers (optional)
    df_no_outliers = df[(df[f"{colName}"] >= lower_bound) & (df[f"{colName}"] <= upper_bound)]

    return df_no_outliers

def _featureScaling(df, type, colName): # for numeric data
    if type == "Normalization":  
        # Create an instance of the MinMaxScaler
        scaler = MinMaxScaler()

        # Fit the scaler to the numerical features in the dataframe and transform them
        # This scales the features to a given range, typically between 0 and 1
        df[f"{colName}"] = scaler.fit_transform(df[f"{colName}"])
    elif type == "Standardization":
        # Create an instance of the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler to the numerical features in the data frame and transform them
        # This standardizes the features by removing the mean and scaling to unit variance
        df[f"{colName}"] = scaler.fit_transform(df[f"{colName}"])

def _labelEncode(df, colName):
    # Create an instance of the LabelEncoder
    encoder = LabelEncoder()

    # Fit the encoder to the categorical feature and transform it
    # This encodes categorical labels with values between 0 and n_classes-1
    df[f"{colName}_encoded"] = encoder.fit_transform(df[f"{colName}"])

    return df

def _cleanText(text):
    # text = text.lower()
    # text = re.sub(r'<br\s*/?>', ' ', text)
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # text = re.sub(r'\s+', ' ', text)
    # # Remove all occurrences of <br /><br />
    # return text

    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text.lower().strip()

def generateStagefile(df, fileOutput):
    df.to_csv(fileOutput)

def checkDataDistribution(df, colName):
    plt.title(f"{colName} Distribution Plot")
    sns.histplot(df[f"{colName}"])
    # Save the plot as an image
    plt.savefig(os.getenv("inputStageStatsPath"), dpi=300, bbox_inches='tight')  # Save with 300 DPI and no extra whitespace
    # plt.show()

def checkDataRelationship(df, colName1, colName2):
    # Relationship between Salary and Experience
    plt.scatter(df[f"{colName1}"], df[f"{colName2}"], color = 'lightcoral')
    plt.title(f"{colName1} vs {colName2}")
    plt.xlabel(f"{colName2}")
    plt.ylabel(f"{colName1}")
    plt.box(False)
    plt.show()

_loadReadData()