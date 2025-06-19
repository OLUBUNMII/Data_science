import os
import pandas as pd
from utilities.kaggle_downloader import download_and_extract

download_and_extract("mlg-ulb/creditcardfraud", output_folder="data")
#load dataset 
# df = pd.read_csv("data/creditcard.csv")

#use os.path to load dataset safely from the right location incase code is moved
csv_path = os.path.join("data", "creditcard.csv")
df = pd.read_csv(csv_path)

# print(df.shape)
# print(df.head())
# print(df.isnull().sum())

#print(df["Class"].value_counts()) #284315 normal transactions, 492 Fraudulent Transactions. Dataset is higly inbalanced.

df.rename(columns={"Class": "is_fraud"}, inplace=True) #I renamed "class" to "is_fraud" to help me understand the code.
#print(df["is_fraud"].value_counts()) #0 = not fraud, 1 = fraud

#Seperate features and labels to scale and balance datset
features = df.drop("is_fraud", axis=1) #The features are everything except the label
labels = df["is_fraud"] #The labels are the "is_fraud" column. 
# print(features)
# print(labels)

#Split data into traina and test splits
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

#Scale features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features_train) #Fit and transform train features
scaled_test_features = scaler.transform(features_test) #Do not fit the test features to avoid data leakage.
# print(scaled_train_features)

# Apply SMOTE to only trained data
# Balance the datset by oversampling, using SMOTE so that the model can predict more accurately. 
# You can either do this or train the model with an undersampled datset, this is fast but risky. You sacrifice valuable data for speed.
from imblearn.over_sampling import SMOTE
smote = SMOTE()
balanced_train_features, balanced_train_labels = smote.fit_resample(scaled_train_features, labels_train)

#Confirm that SMOTE successfully balanced the non-fraud and the fraud samples
# print(df["is_fraud"].value_counts()) - 😅
from collections import Counter
print(Counter(balanced_train_labels))
# Counter({0: 284315, 1: 284315}) #- output before focusing on train data
# Counter({0: 227451, 1: 227451}) #- output after focusing on train data  👈🏾 More Accurate

# Train the model on the balanced training data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(balanced_train_features, balanced_train_labels)

# Predict on test data
predictions = model.predict(scaled_test_features)

# Evaluate
from sklearn.metrics import classification_report
print("Classification Report for Logistic Regression:")
print("Classification_report:\n", classification_report(labels_test, predictions))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(labels_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()