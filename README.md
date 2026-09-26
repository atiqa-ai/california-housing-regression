# California Housing Regression

Predicting median house values for California districts from the
[1990 California census](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html),
built as a step-by-step walkthrough of a regression workflow.

## Results

20% test hold-out, `random_state=42`:

| Metric | Value |
| --- | --- |
| RMSE | **0.632** |
| R² | **0.575** |

An R² of 0.575 means the model explains about 57% of the variance in median
house value. That is a realistic result for a linear model on this dataset —
median income alone carries most of the signal, and the remaining features are
genuinely weak predictors.

## Dataset

`fetch_california_housing()` returns 20,640 rows with 8 numeric features and a
continuous target. The notebook first clips two outliers, which is why the
target axis stops at 5.0:

```python
df = df[df["MedHouseValue"] < 5.0]   # 20640 -> 19648 rows
df = df[df["MedInc"] < 15]            # 19648 -> 19645 rows
```

Values are in units of $100,000, so a target of 2.0 means roughly $200,000.

| Feature | Meaning |
| --- | --- |
| `MedInc` | Median income of households in the district |
| `HouseAge` | Median age of the houses |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | District population |
| `AveOccup` | Average occupants per household |
| `Latitude` / `Longitude` | Location, which turns out to matter a lot |

## Pipeline

```
fetch_california_housing()
   ↓  clip target and income outliers
   ↓  exploratory analysis: histograms, correlation matrix, pairplot
   ↓  StandardScaler (required — income and population are on wildly different scales)
   ↓  PolynomialFeatures(degree=2) + LinearRegression
   ↓  RMSE, R², predicted-vs-actual scatter
```

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

model = LinearRegression()
model.fit(X_poly, y_train)
```

**Why StandardScaler is not optional here.** `MedInc` ranges roughly 0.5–15
while `Population` reaches tens of thousands. Unscaled, gradient-free least
squares still converges, but the coefficients become dominated by the
large-magnitude features and the fit is poorly conditioned.

**Why degree-2 polynomial features.** The underlying relationship is curved
rather than linear, and squaring the scaled features lets the model bend without
switching to a more complex estimator.

## Requirements

Python 3.9+ and Jupyter.

```bash
git clone https://github.com/atiqa-ai/california-housing-regression.git
cd california-housing-regression

python -m venv .venv
# Windows:     .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

The dataset downloads automatically from scikit-learn on first run and is
cached in `~/scikit_learn_data`, so the first execution needs an internet
connection.

## Usage

Open `California.ipynb` and run all cells. Stored outputs are cleared in the
committed file, so the notebook starts blank — which keeps the diff readable
and the repository small.

## Trying to improve it

The notebook is a baseline, and there is clear headroom. Natural next steps:

- **Gradient boosting or random forest.** Tree ensembles handle the curved,
  interacting relationships far better than a linear model on expanded features.
- **Geographic features.** Latitude and longitude carry real signal. Clustering
  districts into regions, or adding distance-to-coast and distance-to-major-city
  features, would likely help substantially.
- **Better outlier handling.** The clipping is crude. A quantile-based cut, or
  simply leaving the outliers in, may score better.
- **Cross-validation.** A single split on a small dataset is noisy. `KFold`
  would give a more trustworthy estimate.

## What I learned

- Scaling is a correctness issue, not a formality, whenever features differ by orders of magnitude.
- An R² of 0.575 is not a failure — it is an honest measurement of what a linear model can extract here.
- Exploratory analysis of correlations often suggests the next feature to engineer faster than trying models at random.
