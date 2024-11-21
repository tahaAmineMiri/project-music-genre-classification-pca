# Music Genre Classification with PCA

This project focuses on classifying music genres using Principal Component Analysis (PCA) for dimensionality reduction and logistic regression for classification. The dataset contains various musical features such as tempo, dynamics range, vocal presence, and more, which are used to predict the genre of a music track.

## Project Structure


## Key Steps

1. **Data Loading and Exploration**: Load the dataset and explore its structure and contents.
2. **Data Cleaning**: Handle missing values and ensure data quality.
3. **Feature Scaling**: Standardize the features to prepare for PCA.
4. **Dimensionality Reduction**: Apply PCA to reduce the number of features while retaining most of the variance.
5. **Model Training**: Train a logistic regression model using the PCA-transformed features.
6. **Model Evaluation**: Evaluate the model's performance using accuracy and classification reports.
7. **Visualization**: Visualize the explained variance by PCA components and the correlation matrix of features.

## Tools and Libraries

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning, including PCA and logistic regression.
- **Matplotlib & Seaborn**: For data visualization.

## Files

- `Music Data Legend.xlsx`: Contains metadata about the dataset.
- `music_dataset_mod.csv`: The main dataset used for classification.
- `Music Genre Classification with PCA - Project.ipynb`: The Jupyter notebook containing the project code and analysis.
- `Solution/Music Genre Classification with PCA - Solution.ipynb`: The solution notebook with detailed steps and explanations.

## Usage

To run this project, ensure you have the required libraries installed. You can install them using:
```
pip install seaborn openpyxl
```
Then, open the Jupyter notebook and execute the cells to see the analysis and results.

### Data loading and Exploration
```
import pandas as pd

df = pd.read_csv('music_dataset_mod.csv')
df.info()
```

### Data Cleaning
```
df_clean = df.dropna(subset=['Genre']).copy()
```

### Feature Scaling
Standardize the features to prepare for PCA:
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean.drop(columns='Genre'))
```

### Dimensionality Reduction
Apply PCA to reduce the number of features while retaining most of the variance:
```
from sklearn.decomposition import PCA

pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)
```

### Model Training
Train a logistic regression model using the PCA-transformed features:
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pca, df_clean['Genre'], test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

### Model Evaluation
Evaluate the model's performance using accuracy and classification reports:
```
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
```

### Visualization
Visualize the explained variance by PCA components and the correlation matrix of features:
```
import matplotlib.pyplot as plt
import seaborn as sns

# Explained variance
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.plot(range(1, 9), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Explained variance by components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Correlation matrix
corr_matrix = df_clean.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

### Conclusion
This project demonstrates the use of PCA for dimensionality reduction and logistic regression for classification in the context of music genre classification. By following the steps outlined above, you can replicate the analysis and gain insights into the relationships between musical features and genres. 