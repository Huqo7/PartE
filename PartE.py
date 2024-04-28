import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def bordered(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')
    return '\n'.join(res)

def HandleDuplicates(data):
    duplicates = data[(data.duplicated())].shape[0]
    print(bordered(" Found {} duplicates ".format(duplicates)))
    if (duplicates > 0):
        cleaned_data = data.drop_duplicates()
        if (cleaned_data[(cleaned_data.duplicated())].shape[0] == 0):
            print(bordered(" Removed {} duplicates ".format(duplicates)))
            return cleaned_data
    return data

def HandleMissing(data):
    missing = data[(data.isna().any(axis=1))].shape[0]
    print(bordered(" Found {} rows with missing values ".format(missing)))
    
    if (missing > 0):
        cleaned_data = data.dropna(axis=0, how='any')
        if (cleaned_data[(data.isna().any(axis=1))].shape[0] == 0):
            print(bordered(" Removed {} rows with missing values ".format(missing)))
            return cleaned_data
    return data

# -------------------------------------------

print(bordered("""
     _______ _________ _______  _______   __    _______   __          _______  _______  _______ _________        _______ 
    (  ___  )\__   __/(  ____ \/ ___   ) /  \  (  __   ) /  \        (  ____ )(  ___  )(  ____ )\__   __/       (  ____ \\  
    | (   ) |   ) (   | (    \/\/   )  | \/) ) | (  )  | \/) )       | (    )|| (   ) || (    )|   ) (          | (    \/   
    | (___) |   | |   | (_____     /   )   | | | | /   |   | |       | (____)|| (___) || (____)|   | |    _____ | (__       
    |  ___  |   | |   (_____  )  _/   /    | | | (/ /) |   | |       |  _____)|  ___  ||     __)   | |   (_____)|  __)      
    | (   ) |   | |         ) | /   _/     | | |   / | |   | |       | (      | (   ) || (\ (      | |          | (         
    | )   ( |___) (___/\____) |(   (__/\ __) (_|  (__) | __) (_      | )      | )   ( || ) \ \__   | |          | (____/\\  
    |/     \|\_______/\_______)\_______/ \____/(_______) \____/      |/       |/     \||/   \__/   )_(          (_______/   
                                                                                                                     

"""))

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Raisin_Dataset/Raisin_Dataset.xlsx')


data = pd.read_excel(filename)

# Handle duplicates
data = HandleDuplicates(data)

# Handle missing values from dataset
data = HandleMissing(data)


print("\n Info about dataset")
print(data.info())

print("\nStatistics of dataset")
print(bordered(data.describe().to_string()))

print(np.std(data[["Area"]], axis=0))

features = data.drop(columns=['Class'])
fig, axes = plt.subplots(nrows=1, ncols=len(features.columns), figsize=(15, 3))  # Adjust figsize as needed

for i, column in enumerate(features.columns):
    features[column].plot(kind='box', ax=axes[i], title=column)
plt.tight_layout()
plt.show()

# PART 2 Supervised

column_arrays = []
for column in data.columns:
    column_arrays.append(data[column].values)

npdata = np.column_stack(column_arrays)
x = npdata[:, slice(0,7)]
y = npdata[:, 7]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                   intercept_scaling=1, l1_ratio=None, max_iter=100,
#                   multi_class='ovr', n_jobs=None, penalty='l2',
#                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
#                   warm_start=False).fit(x_train, y_train)
#model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                   intercept_scaling=1.16, l1_ratio=None, max_iter=400,
#                   multi_class='ovr', n_jobs=None, penalty='l1',
#                   random_state=0, solver='liblinear', tol=0.01, verbose=0,
#                   warm_start=False).fit(x_train, y_train)
#model = LogisticRegression(C=0.9, class_weight=None, dual=False, fit_intercept=True,
#                   intercept_scaling=1.5, l1_ratio=None, max_iter=8000,
#                   multi_class='ovr', n_jobs=None, penalty='l2',
#                   random_state=0, solver='saga', tol=0.0001, verbose=0,
#                   warm_start=False).fit(x_train, y_train)

p_p = model.predict_proba(x_test)
y_p = model.predict(x_test)
score = model.score(x_test, y_test)
conf_m = confusion_matrix(y_test, y_p)
report = classification_report(y_test,y_p)
print(score)
print(report)