
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('task2/mall_customers.csv')

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
data['cluster'] = kmeans.fit_predict(scaled_features)
print(data.head())
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters by Annual Income and Spending Score')
plt.colorbar(label='Cluster')
plt.show()

# Save the cluster results to a CSV file
submission = data[['CustomerID', 'cluster']]  
submission.rename(columns={'CustomerID': 'Customer_ID', 'cluster': 'Cluster_Label'}, inplace=True)
submission.to_csv('task2/customer_clusters.csv', index=False)
