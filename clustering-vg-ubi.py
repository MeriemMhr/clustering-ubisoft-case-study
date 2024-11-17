import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = "C:/Users/mehri/OneDrive/Desktop/Summer 2024/MRKT 671 - Advanced Marketing Analytics/Clustering/SEGMENTATION_DATA.CSV"

# Load the data
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

# Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Fill missing values if necessary (example: with the mean of the column)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Identify categorical columns
categorical_cols = ['TERRITORY']
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Create a pipeline with the preprocessor and the KMeans model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# Fit the pipeline to the data
pipeline.fit(data)

# Predict clusters
data['Cluster'] = pipeline.named_steps['kmeans'].labels_

# Evaluate the clustering
data_scaled = pipeline.named_steps['preprocessor'].transform(data)
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Apply K-Means clustering with optimal clusters (update if needed)
optimal_clusters = 3  # update based on elbow plot if necessary
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters (using the first two principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Cluster Visualization with PCA')
plt.show()


##### Clustering based on sessions & game times ######

# Select relevant columns for clustering (Sessions and Play Time)
session_cols = [
    'SHOOTER_SESSIONS', 'MOBA_SESSIONS', 
    'OPEN_WORLD_SESSIONS', 'OTHER_SESSIONS'
]
playtime_cols = [
    'SHOOTER_GAMETIME', 'MOBA_GAMETIME', 
    'OPEN_WORLD_GAMETIME', 'OTHER_GAMETIME'
]

# Combine session and playtime columns
features_cols = session_cols + playtime_cols

# Identify categorical columns
categorical_cols = ['TERRITORY']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Create a pipeline with the preprocessor and the KMeans model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# Fit the pipeline to the data
pipeline.fit(data)

# Predict clusters
data['Cluster'] = pipeline.named_steps['kmeans'].labels_

# Evaluate the clustering
data_scaled = pipeline.named_steps['preprocessor'].transform(data)
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Apply K-Means clustering with optimal clusters (update if needed)
optimal_clusters = 3  # update based on elbow plot if necessary
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters (using the first two principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Cluster Visualization with PCA')
plt.show()

##### Clustering based on age, sessions & game times ######

# Select relevant columns for clustering (Age, Sessions, and Play Time)
features_cols = [
    'AGE', 'SHOOTER_SESSIONS', 'MOBA_SESSIONS', 
    'OPEN_WORLD_SESSIONS', 'OTHER_SESSIONS',
    'SHOOTER_GAMETIME', 'MOBA_GAMETIME', 
    'OPEN_WORLD_GAMETIME', 'OTHER_GAMETIME'
]

# Identify categorical columns
categorical_cols = ['TERRITORY']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Create a pipeline with the preprocessor and the KMeans model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# Fit the pipeline to the data
pipeline.fit(data)

# Predict clusters
data['Cluster'] = pipeline.named_steps['kmeans'].labels_

# Evaluate the clustering
data_scaled = pipeline.named_steps['preprocessor'].transform(data)
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Apply K-Means clustering with optimal clusters (update if needed)
optimal_clusters = 7  # update based on elbow plot if necessary
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters (using the first two principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Cluster Visualization with PCA')
plt.show()

##### Clustering based on age, aggregated sessions & aggregated game times ######

# Create aggregate features for sessions and game time
data['TOTAL_SESSIONS'] = data['SHOOTER_SESSIONS'] + data['MOBA_SESSIONS'] + data['OPEN_WORLD_SESSIONS'] + data['OTHER_SESSIONS']
data['TOTAL_GAMETIME'] = data['SHOOTER_GAMETIME'] + data['MOBA_GAMETIME'] + data['OPEN_WORLD_GAMETIME'] + data['OTHER_GAMETIME']

# Select relevant columns for clustering (Age, Total Sessions, and Total Game Time)
features_cols = ['AGE', 'TOTAL_SESSIONS', 'TOTAL_GAMETIME']

# Identify categorical columns
categorical_cols = ['TERRITORY']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Create a pipeline with the preprocessor and the KMeans model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# Fit the pipeline to the data
pipeline.fit(data)

# Predict clusters
data['Cluster'] = pipeline.named_steps['kmeans'].labels_

# Evaluate the clustering
data_scaled = pipeline.named_steps['preprocessor'].transform(data)
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Apply K-Means clustering with optimal clusters (update if needed)
optimal_clusters = 4  # update based on elbow plot if necessary
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters (using the first two principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Cluster Visualization with PCA')
plt.show()
