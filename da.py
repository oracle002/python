import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

# Load the insurance dataset
insurance_df = pd.read_csv('insurance.csv')

# Display the first few rows of the dataset
print(insurance_df.head())

### 5.1 Data Preprocessing

# 5.1.1 Handle Missing Data Values (if any)
print(insurance_df.isnull().sum())  # Check for missing values

# Filling missing values if needed (example: using mean or mode)
insurance_df.fillna(insurance_df.mean(numeric_only=True), inplace=True)

# 5.1.2 Encode the Categorical Data
# Encoding 'sex', 'smoker', and 'region' columns
label_encoders = {}
for column in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    insurance_df[column] = le.fit_transform(insurance_df[column])
    label_encoders[column] = le

# 5.1.3 Scale Your Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(insurance_df.drop(columns=['charges']))
scaled_insurance_df = pd.DataFrame(scaled_features, columns=insurance_df.columns[:-1])

### 5.2 Handling Data Issues

# 5.2.1 Handle empty cells (already handled above)
# 5.2.2 Replace empty cells using mean, median, and mode (already handled)
# 5.2.3 Handle data in the wrong format
# Check the data types and convert if necessary
print(insurance_df.dtypes)

# 5.2.4 Handle wrong data from the dataset (check ranges)
# Example: Ensure 'bmi' is within reasonable range
print(insurance_df[(insurance_df['bmi'] < 10) | (insurance_df['bmi'] > 60)])  # Outliers

# 5.2.5 Discover and remove duplicates
insurance_df = insurance_df.drop_duplicates()
print(f"Number of records after removing duplicates: {insurance_df.shape[0]}")

# Create a new attribute 'charges_category' using binning
bins = [0, 10000, 20000, 30000, 40000, float("inf")]
labels = ['Low', 'Medium', 'High', 'Very High', 'Extremely High']
insurance_df['charges_category'] = pd.cut(insurance_df['charges'], bins=bins, labels=labels)

### 5.4 Linear Regression

# 5.4.1 Draw the line of linear regression
X = scaled_insurance_df
y = insurance_df['charges']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# 5.4.2 Evaluate the fit
y_pred = lr.predict(X_test)

# Plotting
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

# 5.4.3 Predict
print("R^2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

### 5.5 Logistic Regression
# Logistic regression to predict if a person smokes based on other features
X_logistic = scaled_insurance_df.drop(columns=['charges'])
y_logistic = insurance_df['smoker']

# Train-Test Split
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.3, random_state=42)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_logistic, y_train_logistic)

# Predictions
y_pred_log = logreg.predict(X_test_logistic)

# Confusion Matrix and Classification Report
print(confusion_matrix(y_test_logistic, y_pred_log))
print(classification_report(y_test_logistic, y_pred_log))

### 5.6 Decision Tree Visualization
# Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)

# Plot Decision Tree
plt.figure(figsize=(15,10))
plot_tree(tree, filled=True, feature_names=insurance_df.columns[:-1], class_names=labels)
plt.show()

### 5.9 K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)

# Accuracy
knn_accuracy = knn.score(X_test, y_test)
print(f"KNN Model Accuracy: {knn_accuracy}")

### 5.10 Random Forest Regressor
# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# R^2 Score
rf_r2 = rf.score(X_test, y_test)
print(f"Random Forest R^2 Score: {rf_r2}")

### 5.11 3D Cluster Visualization
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Reduce dimensions for 3D visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(scaled_insurance_df)

# KMeans for clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_pca)

# 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis')
plt.title("3D Cluster Visualization")
plt.show()