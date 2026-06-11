from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

# Load dataset
iris = load_iris()

# Features (use numpy array directly to avoid requiring pandas)
X = iris.data

# Labels
y = iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# Create model
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# F1 Score
f1 = f1_score(y_test, predictions, average="weighted")

print("\nF1 Score:")
print(f1)
print("First 10 Predictions:")
print(predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test[:10])