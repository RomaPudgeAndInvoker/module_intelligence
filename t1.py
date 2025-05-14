import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

fruit_data = pd.read_csv("fruit_data_with_colors.txt", sep='\t')
X_classification = fruit_data[['mass', 'width', 'height', 'color_score']]
y_classification = fruit_data['fruit_label']

X_classification, y_classification = np.array(X_classification), np.array(y_classification)
permutation = np.random.permutation(len(X_classification))
X_classification = X_classification[permutation]
y_classification = y_classification[permutation]

scaler_classification = MinMaxScaler()
X_classification = scaler_classification.fit_transform(X_classification)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42
)

# Test K from 1 to 20
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_class, y_train_class)
    y_pred_class = knn_classifier.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    accuracy_scores.append(accuracy)

best_k_class = k_values[np.argmax(accuracy_scores)]
best_accuracy = np.max(accuracy_scores)

print("KNN Класифікатор:")
print(f"Найкраще K: {best_k_class}, Найвища точність: {best_accuracy:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Точність класифікації KNN для різних K')
plt.xlabel('K')
plt.ylabel('Точність')
plt.xticks(k_values)
plt.grid(True)
plt.show()
