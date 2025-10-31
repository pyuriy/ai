# Comprehensive Lab for Python AI Development Practice

This lab builds directly on the [python_ai_cheatsheet.md](python_ai_cheatsheet.md) you referenced, providing hands-on exercises to reinforce key concepts. It's designed for beginners to intermediate users, assuming basic Python familiarity. Each section includes:

- **Objectives**: What you'll learn.
- **Setup**: Any prerequisites.
- **Exercises**: Step-by-step tasks with code starters and expected outputs.
- **Challenges**: Optional extensions for deeper practice.
- **Verification**: Ways to check your work (e.g., print statements or plots).

**Overall Setup**:
1. Install Python 3.10+ (use Anaconda for simplicity: [download here](https://www.anaconda.com/download)).
2. Create a virtual environment: `conda create -n ai_lab python=3.12; conda activate ai_lab`.
3. Install libraries: `conda install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter` (or `pip install` equivalents).
4. Launch Jupyter: `jupyter notebook` and create a new notebook (`ai_lab.ipynb`) to run exercises interactively.
5. For each exercise, copy code into cells, run with Shift+Enter, and add markdown for notes.

Run all exercises in the same notebook for a complete workflow. Estimated time: 4-6 hours.

## Lab 1: Python Basics
**Objectives**: Master data types, control structures, functions, and modules for AI scripting.

**Setup**: No additional installs needed.

### Exercise 1.1: Data Types & Collections
- Task: Create a dictionary of student scores, compute averages using comprehensions, and filter passing scores (>60).
- Starter Code:
  ```python
  scores = {'Alice': 85, 'Bob': 55, 'Charlie': 92, 'Diana': 48}
  # Compute average score using dict comprehension (hint: {k: v for k,v in scores.items() if v > 60})
  passing = {k: v for k, v in scores.items() if v > 60}
  avg_passing = sum(passing.values()) / len(passing) if passing else 0
  print(f"Passing students: {passing}")
  print(f"Average passing score: {avg_passing}")
  ```
- Expected Output:
  ```
  Passing students: {'Alice': 85, 'Charlie': 92}
  Average passing score: 88.5
  ```
- Verification: Run and check prints match.

### Exercise 1.2: Control Structures & Functions
- Task: Write a function using loops and conditionals to classify numbers (1-100) as prime, even, or odd. Use a lambda for even check.
- Starter Code:
  ```python
  def classify_numbers(n):
      results = []
      for i in range(1, n+1):
          if i > 1 and all(i % j != 0 for j in range(2, int(i**0.5)+1)):  # Prime check
              results.append(f"{i}: prime")
          else:
              parity = "even" if (lambda x: x % 2 == 0)(i) else "odd"
              results.append(f"{i}: {parity}")
      return results[:10]  # First 10 for brevity

  print(classify_numbers(10))
  ```
- Expected Output: `['1: odd', '2: even', '3: prime', '4: even', '5: prime', '6: even', '7: prime', '8: even', '9: odd', '10: even']`
- Verification: Extend to n=20 and inspect.

**Challenge**: Add pattern matching (Python 3.10+) to handle a list of mixed types (int, str, list) and extract sums.

## Lab 2: Data Manipulation
**Objectives**: Practice array operations, data cleaning, and grouping with NumPy and Pandas.

**Setup**: Download a sample CSV (e.g., Iris dataset via scikit-learn) or create one inline.

### Exercise 2.1: NumPy Arrays
- Task: Generate random data, perform slicing/indexing, and compute statistics.
- Starter Code:
  ```python
  import numpy as np

  # Generate 3x3 random matrix (integers 1-10)
  data = np.random.randint(1, 11, (3, 3))
  print("Matrix:\n", data)

  # Extract even numbers using boolean indexing
  evens = data[data % 2 == 0]
  print("Even numbers:", evens)

  # Reshape to 1D and compute mean/std
  flat = data.flatten()
  print(f"Mean: {np.mean(flat):.2f}, Std: {np.std(flat):.2f}")
  ```
- Expected Output (varies due to random; e.g.):
  ```
  Matrix:
   [[5 8 3]
    [1 9 6]
    [4 2 7]]
  Even numbers: [8 6 4 2]
  Mean: 5.00, Std: 2.58
  ```
- Verification: Run multiple times; ensure shape is (9,) after flatten.

### Exercise 2.2: Pandas DataFrames
- Task: Load Iris data, clean (handle any NaNs—simulate by adding), group by species, and merge with a dummy dataset.
- Starter Code:
  ```python
  import pandas as pd
  from sklearn.datasets import load_iris

  iris = load_iris()
  df = pd.DataFrame(iris.data, columns=iris.feature_names)
  df['species'] = iris.target  # 0=setosa, etc.

  # Simulate NaN: df.iloc[0, 0] = np.nan
  df.iloc[0, 0] = np.nan  # Add NaN
  df_filled = df.fillna(df.mean(numeric_only=True))  # Fill NaNs

  # Group: Mean sepal length by species
  grouped = df_filled.groupby('species')['sepal length (cm)'].mean()
  print(grouped)

  # Merge with dummy df2: {'species': [0,1], 'grade': ['A', 'B']}
  df2 = pd.DataFrame({'species': [0,1], 'grade': ['A', 'B']})
  merged = pd.merge(df_filled, df2, on='species', how='left')
  print(merged.head())
  ```
- Expected Output:
  ```
  species
  0    5.006000  # Approx for setosa
  1    5.936000  # Versicolor
  2    6.588000  # Virginica
  Name: sepal length (cm), dtype: float64
  # Head shows merged columns
  ```
- Verification: Check `df.shape` (150,5); no NaNs post-fill (`df.isnull().sum()`).

**Challenge**: Use Dask for a large simulated dataset (e.g., 1M rows) if installed, or stick to Pandas.

## Lab 3: Visualization
**Objectives**: Create plots for data exploration and model insights.

**Setup**: Use Iris DataFrame from Lab 2.

### Exercise 3.1: Matplotlib Basics
- Task: Plot sepal length vs. width with labels and subplots.
- Starter Code:
  ```python
  import matplotlib.pyplot as plt

  # From Lab 2: use df
  plt.figure(figsize=(8,4))
  plt.plot(df['sepal length (cm)'], df['sepal width (cm)'], 'o-', label='Sepal Data')
  plt.xlabel('Length (cm)'); plt.ylabel('Width (cm)')
  plt.title('Sepal Dimensions')
  plt.legend(); plt.grid(True)
  plt.show()

  # Subplots: Length hist and scatter
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
  ax1.hist(df['sepal length (cm)'], bins=20, alpha=0.7); ax1.set_title('Histogram')
  ax2.scatter(df['sepal length (cm)'], df['sepal width (cm)']); ax2.set_title('Scatter')
  plt.tight_layout(); plt.show()
  ```
- Expected Output: Line/scatter plot and subplots showing distributions (bell-shaped hist, clustered scatter).
- Verification: Save as PNG (`plt.savefig('sepal_plot.png')`) and inspect.

### Exercise 3.2: Seaborn Enhancements
- Task: Heatmap of correlations colored by species.
- Starter Code:
  ```python
  import seaborn as sns

  corr = df.corr(numeric_only=True)
  plt.figure(figsize=(8,6))
  sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
  plt.title('Feature Correlations')
  plt.show()

  # Pairplot by species
  sns.pairplot(df, hue='species', diag_kind='hist')
  plt.show()
  ```
- Expected Output: Heatmap with values like 0.87 for petal length/width; pairplot grids with clusters.
- Verification: High correlation (>0.9) between petal features.

**Challenge**: Animate a scatter plot evolution using Matplotlib's `FuncAnimation` (import from `matplotlib.animation`).

## Lab 4: Machine Learning with Scikit-learn
**Objectives**: Build, evaluate, and tune ML models on Iris dataset.

**Setup**: Use preprocessed data from Lab 2.

### Exercise 4.1: Preprocessing & Splitting
- Task: Scale features and split data.
- Starter Code:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler

  X = df.drop('species', axis=1).values
  y = df['species'].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  print(f"Train shape: {X_train_scaled.shape}, Mean: {X_train_scaled.mean(axis=0)[0]:.2f}")  # ~0
  ```
- Expected Output: `Train shape: (120, 4), Mean: 0.00`
- Verification: Test mean ≈0, std≈1 (`X_train_scaled.std(axis=0)`).

### Exercise 4.2: Train & Evaluate Models
- Task: Train KNN classifier, compute accuracy and confusion matrix.
- Starter Code:
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(X_train_scaled, y_train)
  y_pred = knn.predict(X_test_scaled)

  print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  ```
- Expected Output: `Accuracy: 1.00` (Iris is easy); diagonal-heavy matrix; precision/recall ~1.0.
- Verification: Cross-validate (`cross_val_score(knn, X_train_scaled, y_train, cv=5).mean() > 0.95`).

### Exercise 4.3: Tuning with GridSearch
- Task: Tune KNN hyperparameters.
- Starter Code:
  ```python
  from sklearn.model_selection import GridSearchCV

  params = {'n_neighbors': [3,5,7], 'weights': ['uniform', 'distance']}
  grid = GridSearchCV(knn, params, cv=5)
  grid.fit(X_train_scaled, y_train)
  print(f"Best params: {grid.best_params_}, Best score: {grid.best_score_:.2f}")
  ```
- Expected Output: e.g., `Best params: {'n_neighbors': 5, 'weights': 'uniform'}, Best score: 0.98`
- Verification: Refit with best params; test accuracy > baseline.

**Challenge**: Try SVM (`SVC()`) and compare ROC curves (`from sklearn.metrics import roc_curve`).

## Lab 5: Deep Learning with Keras
**Objectives**: Build and train neural networks for image classification.

**Setup**: TensorFlow/Keras installed.

### Exercise 5.1: MLP on MNIST
- Task: Train a simple dense network.
- Starter Code:
  ```python
  import tensorflow as tf
  from tensorflow.keras.datasets import mnist
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(60000, 784).astype('float32') / 255
  x_test = x_test.reshape(10000, 784).astype('float32') / 255
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)

  model = Sequential([
      Dense(512, activation='relu', input_shape=(784,)),
      Dropout(0.2),
      Dense(10, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)

  loss, acc = model.evaluate(x_test, y_test)
  print(f"Test accuracy: {acc:.2f}")
  ```
- Expected Output: Training logs show acc rising to ~0.98; `Test accuracy: 0.98`
- Verification: Plot history (`plt.plot(history.history['accuracy'])`); no overfitting (val_acc close to train).

### Exercise 5.2: CNN Extension
- Task: Upgrade to CNN.
- Starter Code:
  ```python
  from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D

  # Reshape for CNN
  x_train_cnn = x_train.reshape(60000, 28, 28, 1)
  x_test_cnn = x_test.reshape(10000, 28, 28, 1)

  model_cnn = Sequential([
      Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
      MaxPooling2D((2,2)),
      Flatten(),
      Dense(10, activation='softmax')
  ])
  model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  history_cnn = model_cnn.fit(x_train_cnn, y_train, epochs=3, batch_size=128, validation_split=0.2)

  print(f"CNN Test accuracy: {model_cnn.evaluate(x_test_cnn, y_test)[1]:.2f}")
  ```
- Expected Output: `CNN Test accuracy: 0.99` (better than MLP).
- Verification: Compare histories; CNN converges faster.

**Challenge**: Add LSTM for sequential MNIST variants (treat rows as time steps) or use callbacks like EarlyStopping.

## Conclusion & Next Steps
- **Full Pipeline**: Combine Labs 2-5: Preprocess Iris, train CNN (adapt shapes), visualize predictions.
- **Debugging Tips**: Use `%debug` in Jupyter for errors; check shapes with `.shape`.
- **Extensions**: Deploy a model with Flask (`pip install flask`); explore PyTorch equivalent.
- **Resources**: Scikit-learn user guide, Keras examples, Kaggle datasets for real data.

Save your notebook and share outputs if stuck—practice iterating on errors! If you need solutions or expansions, ask.
