# Comprehensive Cheatsheet for Python AI Development

This cheatsheet covers essential Python concepts, libraries, and workflows for AI development, including data handling, machine learning (ML), deep learning (DL), and visualization. It's structured for quick reference with code snippets. Focus on practical usage for building, training, and evaluating models.

## 1. Python Basics
Core syntax for scripting AI pipelines. (Based on standard Python references.)

### Data Types & Collections
| Type | Example | Key Operations |
|------|---------|----------------|
| **int/float/complex** | `x = 5; y = 3.14; z = 1+2j` | `int('5'); float(3); complex(1,2)` |
| **str** | `s = 'hello'` | `s.upper(); s.split(); s.strip()` |
| **list** (mutable) | `lst = [1,2,3]` | `lst.append(4); lst[1:3]; lst.sort()` |
| **tuple** (immutable) | `tup = (1,2,3)` | `a,b,c = tup; tup[0]` |
| **dict** | `d = {'a':1, 'b':2}` | `d['c']=3; d.get('a',0); d.keys()` |
| **set** | `st = {1,2,3}` | `st.add(4); st.union({5}); st & {1,2}` |

- **Type Checking**: `type(x); isinstance(x, list)`
- **Comprehensions**: `[x**2 for x in range(5) if x>2]` → `[9,16]`

### Control Structures
- **If/Elif/Else**: 
  ```python
  if cond: pass
  elif cond2: pass
  else: pass
  # Ternary: val = 'even' if x%2==0 else 'odd'
  ```
- **Loops**:
  ```python
  for i, val in enumerate([1,2,3]): print(i, val)  # 0 1, 1 2, 2 3
  while cond: pass
  ```
- **Match (Pattern Matching, Python 3.10+)**:
  ```python
  match x:
      case [a,b]: print(a+b)
      case {'key': val}: print(val)
      case _: print('default')
  ```

### Functions
- **Definition**:
  ```python
  def func(a=1, *args, **kwargs): return a + sum(args)
  # Lambda: add = lambda x,y: x+y
  ```
- **Decorators** (e.g., for timing):
  ```python
  import functools
  def timer(f):
      @functools.wraps(f)
      def wrapper(*args, **kwargs):  # Timing code here
          return f(*args, **kwargs)
      return wrapper
  ```

### Imports & Modules
```python
import numpy as np  # Standard
from sklearn import datasets  # Specific
import sys; sys.path.append('/path')  # Custom
if __name__ == '__main__': main()  # Script guard
```

## 2. Data Manipulation
Essential for preprocessing in AI workflows.

### NumPy (Arrays & Linear Algebra)
```python
import numpy as np
# Creation
arr = np.array([1,2,3])  # From list
zeros = np.zeros((2,3))  # Zeros
rand = np.random.randn(4,4)  # Random normal
lin = np.linspace(0,10,5)  # Even spacing

# Properties & Indexing
print(arr.shape, arr.dtype)  # (3,) int64
sub = arr[1:3]  # [2,3]
bool_idx = arr[arr > 1]  # [2,3]
arr[0] = 10  # Mutable

# Ops
arr2 = np.array([4,5,6])
add = arr + arr2  # [5,7,9]
dot = np.dot(arr, arr2)  # 32
mean = np.mean(arr, axis=0)  # Scalar mean

# Manipulation
trans = arr.reshape(3,1).T  # Transpose
stack = np.vstack([arr, arr2])  # Vertical stack
```

### Pandas (DataFrames & Series)
```python
import pandas as pd
# Creation
df = pd.DataFrame({'A': [1,2], 'B': [3,4]})
s = pd.Series([1,2], index=['x','y'])

# I/O
df = pd.read_csv('data.csv')
df.to_csv('out.csv', index=False)

# Selection
row = df.iloc[0]  # First row
col = df['A']  # Column
subset = df[df['A'] > 1]  # Filter

# Cleaning
df.fillna(0)  # NaNs to 0
df.dropna()  # Drop NaNs
df['A'] = df['A'].astype(float)

# Grouping
grouped = df.groupby('key').agg({'A': 'mean', 'B': 'sum'})

# Merge
df2 = pd.merge(df, other_df, on='key', how='inner')
```

## 3. Visualization
For model diagnostics and data exploration.

### Matplotlib & Seaborn
```python
import matplotlib.pyplot as plt
import seaborn as sns  # For styled plots

# Basic Plot
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.xlabel('X'); plt.ylabel('Y'); plt.title('Sine Wave')
plt.legend(); plt.show()

# Subplots
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
ax1.plot(x,y); ax1.set_title('Plot 1')
ax2.scatter(x,y); ax2.set_title('Scatter')

# Pandas Integration
df.plot(x='A', y='B', kind='scatter')
sns.heatmap(df.corr())  # Correlation matrix
plt.show()
```

## 4. Machine Learning with Scikit-learn
Workflow: Load → Preprocess → Train → Evaluate → Tune.

### Preprocessing & Splitting
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

enc = LabelEncoder()
y_enc = enc.fit_transform(y)  # For categorical y
```

### Models (Supervised)
| Task | Estimator | Example |
|------|-----------|---------|
| **Regression** | LinearRegression | `from sklearn.linear_model import LinearRegression; lr = LinearRegression(); lr.fit(X_train, y_train)` |
| **Classification** | SVC, KNN, GaussianNB | `from sklearn.svm import SVC; svc = SVC(kernel='linear'); svc.fit(X_train, y_train)`<br>`from sklearn.neighbors import KNeighborsClassifier; knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_train, y_train)` |
| **Clustering (Unsupervised)** | KMeans, PCA | `from sklearn.cluster import KMeans; kmeans = KMeans(n_clusters=3); kmeans.fit(X)`<br>`from sklearn.decomposition import PCA; pca = PCA(n_components=2); X_pca = pca.fit_transform(X)` |

### Evaluation & Metrics
```python
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report

y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))  # Classification
print(mean_squared_error(y_test, y_pred))  # Regression
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X_train, y_train, cv=5)  # Mean accuracy
```

### Tuning
```python
from sklearn.model_selection import GridSearchCV
params = {'n_neighbors': [3,5,7], 'weights': ['uniform', 'distance']}
grid = GridSearchCV(knn, params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)  # Best hyperparameters
```

## 5. Deep Learning
Focus on Keras (TensorFlow backend) for simplicity; concepts apply to PyTorch. For PyTorch basics: Tensors (`torch.tensor([1,2])`), models (`nn.Module`), training loops with `optimizer.step()`.

### Keras Workflow
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Data (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28*28)).astype('float32') / 255
y_train = to_categorical(y_train, 10)  # One-hot

# Sequential Model (MLP for Classification)
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# CNN Example
model_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    Flatten(),
    Dense(10, activation='softmax')
])
model_cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Reshape data: x_train = x_train.reshape((60000,28,28,1))

# RNN (LSTM for Sequences)
model_rnn = Sequential([
    LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(seq_len, features)),
    Dense(1, activation='sigmoid')
])

# Evaluate & Predict
loss, acc = model.evaluate(x_test, to_categorical(y_test,10))
preds = model.predict(x_test)
model.save('model.h5'); from tensorflow.keras.models import load_model; loaded = load_model('model.h5')
```

### DL Concepts (Neural Networks)
| Component | Formula/Key Idea | Common Activations |
|-----------|------------------|--------------------|
| **Forward Pass** | \( z = Wx + b; a = g(z) \) | ReLU: \(\max(0,z)\); Sigmoid: \(1/(1+e^{-z})\); Tanh: \((e^z - e^{-z})/(e^z + e^{-z})\) |
| **Loss** | Binary Cross-Entropy: \( -[y\log(z) + (1-y)\log(1-z)] \) | - |
| **Backprop** | \( \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} \); Update: \( w \leftarrow w - \alpha \nabla L \) | - |
| **CNN** | Output size: \( N = \frac{W - F + 2P}{S} + 1 \) (W=input, F=filter, P=pad, S=stride) | Batch Norm: \( x' = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \) |
| **RNN/LSTM** | Gates: Input/Forget/Output for memory | Vanishing gradients mitigated by LSTM |

- **Overfitting Prevention**: Dropout (drop units with prob p); Early Stopping: `from tensorflow.keras.callbacks import EarlyStopping; cb = EarlyStopping(patience=3)`

## 6. Advanced Tips
- **Pipelines**: `from sklearn.pipeline import Pipeline; pipe = Pipeline([('scale', StandardScaler()), ('model', KNN())]); pipe.fit(X_train, y_train)`
- **Deployment**: Use Flask/FastAPI for APIs; ONNX for model export.
- **Best Practices**: Use virtual envs (`venv`), profile with `timeit`, handle large data with Dask.
- **Resources**: For more, see Scikit-learn docs , Keras examples .

This cheatsheet is current as of Oct 2025; libraries evolve—check docs for updates.
