#########################################################
# Author: Edcel Vista                                   #
# Note: _Func(x) = Sync | Func(x) = Async               #
#########################################################

import os, re, nltk
import pandas as pd # We need Pandas for data manipulation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder #  Sklearn libraries are used for machine learning operations
import matplotlib.pyplot as plt # MatplotLib, and Seaborn for visualizations
import seaborn as sns
from dotenv import load_dotenv
from _utils import cleanUp
from nltk.corpus import stopwords

# Load environment variables from .env file
load_dotenv()

class Data():
    def __init__(self):
        # Load the NLTK stopwords
        nltk.download('stopwords')

        # Create an instance of the LabelEncoder
        self.encoder = LabelEncoder()

        # Parameters Config
        self.inputStagePath              = os.getenv("inputStagePath")
        self.inputStageStatsPath         = os.getenv("inputStageStatsPath")
        self.inputStageStatsTrainingPath = os.getenv("inputStageStatsTrainingPath")

        self.xColumnName = "review"

    def _loadReadData(self):
        print("")
        print("Cleaning up previous data:")
        cleanUp([self.inputStagePath, self.inputStageStatsPath, self.inputStageStatsTrainingPath])

        # Example CSV file path (replace with your file path)
        csv_file = os.getenv("inputFilePath")
        # Read CSV file into a pandas DataFrame
        df       = pd.read_csv(csv_file)
        # Display the first few rows of the DataFrame

        print("")
        df = self._handleMissingData(df, True)
        df = self._removeDuplicates(df, f"{self.xColumnName}")
        df = self._removeSpecialCharacters(df, f"{self.xColumnName}")
        df = self._removeStopWords(df, f"{self.xColumnName}")
        # df = _handleOutliers(df, "column name") # for numeric data > Outliers are data points that significantly differ from other observations in a dataset. Addressing outliers is crucial for maintaining data integrity and model performance.
        # df = _featureScaling(df, "column name") # for numeric data > Scaling numerical features to ensure they have a consistent range, helps algorithms converge faster and prevents features with larger scales from dominating.
        df = self._labelEncode(df, "sentiment") # convert string labels into int label data

        if not df.empty:
            # print(df.head())
            # print(df.describe())
            print("")
            print(f"Training Data Summary: Rows = {df.shape[0]} | Columns = {df.shape[1]}")
            self.generateStagefile(df, os.getenv("inputStagePath"))
            self.checkDataDistribution(df, "sentiment_encoded")
        else:
            print(f"No data input.")

    def _handleMissingData(self, df, isDropMissing = True):
        # Assuming 'df' is your DataFrame
        # Detect missing values
        missing_values = df.isnull()  # or df.notnull() to find non-missing values
        # Count missing values per column
        missing_counts = df.isnull().sum()
        # Print missing values counts
        print("Missing Values Count:")
        print(missing_counts)

        if isDropMissing:
            # Drop rows with any missing values
            df_cleaned = df.dropna()
            # Drop columns with any missing values
            # df_dropna_cols = df.dropna(axis=1)
        else: # FOR-LATER-SUPPORT
            # Impute missing values with mean
            df_cleaned = df.fillna(df.mean())
        
        return df_cleaned

    def _removeDuplicates(self, df, targetCols = False):
        if targetCols:
            # Drop duplicates based on specific columns
            df_unique = df.drop_duplicates(subset=[f"{targetCols}"])
        else:
            # Drop duplicates based on all columns
            df_unique = df.drop_duplicates()

        return df_unique
        
    def _removeSpecialCharacters(self, df, targetCols = False):
        # Apply sanitization to the column
        df[f"{targetCols}"] = df[f"{targetCols}"].apply(self._cleanText)
        return df

    def _removeStopWords(self, df, targetCols = False):
        # Apply sanitization to the column
        df[f"{targetCols}"] = df[f"{targetCols}"].apply(self._stopWords)
        return df

    def _stopWords(self, text):
        # Remove Stopwords
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        text    = pattern.sub('', text)
        
        return text

    def _handleOutliers(self, df, colName): # for numeric data
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

    def _featureScaling(self, df, type, colName): # for numeric data
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

    def _labelEncode(self, df, colName):
        # Fit the encoder to the categorical feature and transform it
        # This encodes categorical labels with values between 0 and n_classes-1
        df[f"{colName}_encoded"] = self.encoder.fit_transform(df[f"{colName}"])

        return df

    def _cleanText(self, text):
        # text = text.lower()
        # text = re.sub(r'<br\s*/?>', ' ', text)
        # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # text = re.sub(r'\s+', ' ', text)
        # # Remove all occurrences of <br /><br />
        # return text

        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
        return text.lower().strip()

    def generateStagefile(self, df, fileOutput):
        df.to_csv(fileOutput)

    def checkDataDistribution(self, df, colName):
        plt.title(f"{colName} Distribution Plot")
        sns.histplot(df[f"{colName}"])
        # Save the plot as an image
        plt.savefig(self.inputStageStatsPath, dpi=300, bbox_inches='tight')  # Save with 300 DPI and no extra whitespace
        # plt.show()

dataClass = Data()
dataClass._loadReadData()