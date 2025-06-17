import pandas as pd
from utilities.kaggle_downloader import download_and_extract

download_and_extract("mlg-ulb/creditcardfraud", output_folder="data")
#load dataset
df = pd.read_csv("data/creditcard.csv")
print(df.shape)