import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv('ETHEREUM_FRAUD_DETECTION-main/transaction_dataset.csv/transaction_dataset.csv', index_col=0)
df = df.iloc[:, 2:]

# Drop categorical features and fill missing values
categories = df.select_dtypes('O').columns
df.drop(columns=categories, inplace=True)
df.fillna(df.median(), inplace=True)

# Drop features with zero variance
no_var = df.var() == 0
df.drop(columns=df.columns[no_var], inplace=True)

# Drop specific columns based on prior analysis
drop_columns = ['total transactions (including tnx to create contract', 'total ether sent contracts', 
                'max val sent to contract', ' ERC20 avg val rec', ' ERC20 max val rec', ' ERC20 min val rec', 
                ' ERC20 uniq rec contract addr', 'max val sent', ' ERC20 avg val sent', ' ERC20 min val sent', 
                ' ERC20 max val sent', ' Total ERC20 tnxs', 'avg value sent to contract', 
                'Unique Sent To Addresses', 'Unique Received From Addresses', 'total ether received', 
                ' ERC20 uniq sent token name', 'min value received', 'min val sent', ' ERC20 uniq rec addr',
                'min value sent to contract', ' ERC20 uniq sent addr.1']
df.drop(columns=drop_columns, inplace=True)

# Split the dataset into features and target
y = df['FLAG']
X = df.drop(columns=['FLAG'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Normalize the features
norm = PowerTransformer()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train_norm, y_train)

# Train the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)

# Evaluate the model
y_pred = rf.predict(X_test_norm)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# Save the normalizer
with open('models/power_transformer.pkl', 'wb') as norm_file:
    pickle.dump(norm, norm_file)

# Test the model with an input address
original_df = pd.read_csv('ETHEREUM_FRAUD_DETECTION-main/transaction_dataset.csv/transaction_dataset.csv', index_col=0)
predictions = rf.predict(norm.transform(X_test))
fraud_indices = np.where(predictions == 1)[0]
non_fraud_indices = np.where(predictions == 0)[0]

fraudulent_addresses = original_df.iloc[X_test.index[fraud_indices]]['Address']
fraudulent_dataset = pd.DataFrame({'Address': fraudulent_addresses})

non_fraudulent_addresses = original_df.iloc[X_test.index[non_fraud_indices]]['Address']
non_fraudulent_dataset = pd.DataFrame({'Address': non_fraudulent_addresses})

print(fraudulent_dataset)
print(non_fraudulent_dataset)

input_addr = input("Input the receiver's Ethereum Account Address: ")

if len(input_addr) != 42:
    print("Invalid Address !!")
elif input_addr in fraudulent_dataset['Address'].values:
    print("The model predicts that the transaction associated with the provided address is a fraud account.")
else:
    print("The model predicts that the transaction associated with the provided address is a non-fraud account.")
