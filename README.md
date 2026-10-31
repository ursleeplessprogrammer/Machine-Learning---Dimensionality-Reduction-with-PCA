# 🧭 Dimensionality Reduction with PCA

> **Find the main patterns in your data—and represent them with fewer features.**

## 🗺️ Trail Map

- [What Is Dimensionality Reduction?](#-what-is-dimensionality-reduction)
- [What Is PCA?](#-what-is-pca)
- [How PCA Works](#-how-pca-works)
- [Principal Components](#-principal-components)
- [Explained Variance](#-explained-variance)
- [Why Scale Features?](#-why-scale-features)
- [When PCA Is Useful](#-when-pca-is-useful)
- [Python Example](#-python-example)
- [PCA for Visualization](#-pca-for-visualization)
- [Limitations and Common Mistakes](#️-limitations-and-common-mistakes)
- [PCA and Other Methods](#-pca-and-other-methods)
- [Field Notes](#-field-notes)
- [Further Reading](#-further-reading)

## 🔍 What Is Dimensionality Reduction?

A dataset’s **dimensionality** is the number of features used to describe each example. A dataset with 100 features has 100 dimensions.

Dimensionality reduction transforms data into fewer features while trying to preserve information useful for a particular purpose. It can help:

- Visualize high-dimensional data
- Reduce storage or computation
- Remove some redundancy between features
- Create inputs for another machine-learning model

Reducing dimensions can also discard useful information. Fewer features do not automatically mean a better model.

## 🧠 What Is PCA?

**Principal Component Analysis (PCA)** is a linear dimensionality-reduction technique. It finds new directions—called **principal components**—that capture as much variance in the data as possible, in sequence. The components are orthogonal to one another. ([scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))

PCA combines information from the original features to create new features. These new features are not usually the original measurements; each is a weighted combination of them.

For example, measurements of height, weight, and body measurements might share information. PCA may represent some of that shared variation with fewer combined dimensions.

## ⚙️ How PCA Works

At a high level, PCA follows these steps:

1. Center the features by subtracting each feature’s mean.
2. Find directions along which the data varies most.
3. Rank those directions by the amount of variance they capture.
4. Select the desired number of directions.
5. Project the original data onto those directions.

```text
Original features
       ↓
Center and, when appropriate, scale
       ↓
Find directions of greatest variance
       ↓
Select principal components
       ↓
Represent each example in fewer dimensions
```

PCA is commonly calculated using **singular value decomposition (SVD)**. Its results depend on the data, preprocessing, and number of components selected.

## 🧭 Principal Components

A **principal component** is a direction through the feature space.

- **First principal component:** Captures the greatest possible variance.
- **Second principal component:** Captures the greatest remaining variance while being orthogonal to the first.
- **Later components:** Capture progressively less remaining variance.

A component is a weighted combination of the original features. The weights are called **loadings** or component coefficients.

The components can be useful, but they may be harder to interpret than the original features. A component that combines many variables does not necessarily correspond to a clear real-world concept.

## 📊 Explained Variance

PCA reports how much variance each component captures.

- **Explained variance:** The amount of variance captured by a component.
- **Explained variance ratio:** The fraction of total variance captured by that component.

For example, if the first two components explain 80% of the variance, projecting onto them retains 80% of the measured variation according to PCA’s objective.

That does **not** mean the projection preserves 80% of all useful information for every task. Variance is not the same as predictive value. A low-variance feature can still be important for predicting a target.

A cumulative explained-variance plot can help explore how many components to keep, but choose the number based on the intended use and validate downstream performance when PCA is part of a predictive model.

## 📐 Why Scale Features?

PCA is sensitive to feature scales. By default, scikit-learn’s PCA centers features but does **not** scale each feature to unit variance. ([scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))

Suppose one feature measures age from 0 to 100 and another measures income from 0 to 200,000. The larger numerical range can dominate the variance and therefore influence the principal components more strongly.

Standardizing numerical features before PCA is often sensible when features use different units:

```text
StandardScaler → PCA
```

Scaling is not automatic in every situation. If feature scales have meaningful, intentional importance, decide whether standardization is appropriate rather than applying it without thought.

## 🌎 When PCA Is Useful

PCA may be useful for:

- Reducing many correlated numerical features
- Visualizing data in two or three dimensions
- Compressing data while retaining much of its variance
- Preparing numerical data for some downstream models
- Exploring broad patterns in a dataset

PCA is not a clustering algorithm or a classifier. It transforms data; another method may then use that transformed data.

## 🧪 Python Example

This example standardizes features, reduces them to two components, and checks how much variance those components explain. It uses scikit-learn’s built-in Iris dataset.

Install the dependencies:

```bash
pip install scikit-learn matplotlib
```

```python
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = load_iris()
X = dataset.data
y = dataset.target

# Standardize because the input measurements use different scales.
X_scaled = StandardScaler().fit_transform(X)

# Project the data onto two principal components.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total variance explained by two components:",
    pca.explained_variance_ratio_.sum(),
)

# Plot the two-dimensional representation.
for class_index, class_name in enumerate(dataset.target_names):
    class_mask = y == class_index

    plt.scatter(
        X_pca[class_mask, 0],
        X_pca[class_mask, 1],
        label=class_name,
        alpha=0.8,
    )

plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Iris dataset projected with PCA")
plt.legend()
plt.tight_layout()
plt.show()
```

The plot shows the dataset after projection, not the original measurements. Some separation in a 2D plot does not prove that PCA is suitable for a predictive task.

## 📈 Choosing the Number of Components

You can choose a fixed number of components:

```python
PCA(n_components=2)
```

Or, with the full SVD solver, choose a variance-retention threshold:

```python
PCA(n_components=0.95, svd_solver="full")
```

A threshold such as `0.95` asks PCA to retain enough components to explain at least 95% of the training data’s variance. It does not guarantee the best downstream model.

A practical approach:

1. Fit preprocessing and PCA on training data only.
2. Inspect the explained-variance ratios.
3. Compare reasonable component counts using cross-validation.
4. Evaluate the selected pipeline on untouched test data.

## 🔗 Using PCA with a Predictive Model

If PCA is part of a supervised-learning pipeline, fit it only on the training data. A scikit-learn pipeline helps keep scaling, PCA, and prediction together:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95, svd_solver="full"),
    LogisticRegression(max_iter=2000),
)
```

When this pipeline is fitted on training data, the scaler and PCA learn their parameters from that data. They can then transform new data consistently. Fitting PCA on the full dataset before splitting risks data leakage.

## ⚠️ Limitations and Common Mistakes

### PCA is linear

PCA finds linear combinations of the input features. It may not capture curved or more complex structures. Kernel PCA and nonlinear methods are alternatives, but have different assumptions and trade-offs.

### High variance is not always useful

PCA keeps directions with the most variance, not necessarily the directions most useful for a prediction target.

### Components can be difficult to explain

Principal components combine original features. This can make them less interpretable than the original variables.

### Scaling choices matter

Features with large numerical ranges may dominate PCA if data is not appropriately scaled.

### Information is discarded

Keeping fewer components loses some information. The discarded variance may or may not matter for the task.

### Sparse data needs consideration

Standard PCA centers input data, which can be inefficient for sparse matrices. For some sparse-data workflows, **TruncatedSVD** may be a more suitable alternative because it does not center the data. ([scikit-learn: Decomposing signals](https://scikit-learn.org/stable/modules/decomposition.html))

### Large datasets can require different tools

Standard PCA processes data in batches that must fit in memory. **IncrementalPCA** supports batch-wise processing and can be useful when the entire dataset does not fit in memory. ([scikit-learn: IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html))

## 🔄 PCA and Other Methods

| Method | Main idea | Linear? |
|---|---|---:|
| PCA | Finds directions that capture maximum variance | Yes |
| TruncatedSVD | Reduces dimensions without centering the input | Yes |
| Kernel PCA | Uses a kernel to capture some nonlinear structure | No |
| t-SNE | Often used to visualize local neighborhoods in low dimensions | No |
| UMAP | Nonlinear method often used for visualization and structure exploration | No |

These methods serve different purposes. A 2D visualization method is not automatically an appropriate preprocessing step for a predictive model.

## 📝 Field Notes

- PCA replaces original features with orthogonal combinations called principal components.
- Components are ordered by the variance they explain.
- PCA centers data but does not scale features by default in scikit-learn.
- Scaling is often important when features use different units.
- Explained variance is not the same as predictive usefulness.
- Fit PCA on training data only when evaluating a model.
- Use IncrementalPCA or alternatives when data size or sparsity makes standard PCA unsuitable.

## 🔗 Further Reading

- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn: Decomposing Signals and Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- [scikit-learn: IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
- [scikit-learn: PCA with a Logistic Regression Pipeline](https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html)

---

*PCA gives a compact view of numerical data by keeping directions with high variance. Whether that compact view is useful depends on the problem, the preprocessing, and what information the discarded components contain.*
