# Exploring High Dimentional Data

Assuming working on a tidy dataset: each row is an observation and each column is a feature.

df.shape # (rows, columns)
df.describe() # exclude = 'number' # for non numerical columns

high dimentional data: more than 10 features

## Feature selection vs Feature extraction

They are both  methods of reducing dimentionality

Easiest way: drop irrelevant columns, need domain knowledge.
.drop(columns = ['column1', 'column2'], axis = 1)
Seaborns pairplot: sns.pairplot(df, hue = 'column1', diag_kind = 'hist') use to visualize and identify irrelevant columns.

Feature extraction: create new features which have few irrelevant information. eg. PCA
Downside: lose interpretability.

## t-SNE visualization for high dimentional data

t-SNE: t-distributed stochastic neighbor embedding
Dimentionality reduction rechnique
Feature extraction, used for visualization 

[sklearn doc](https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html)

## The curse of dimentionality

Models overfit with high dimentional data
which feature to drop? 
To avoid overfitting the number of observations should increase exponentially with the number of features.
Solution: reduce dimentionality

## Features with missing values or little varaince

```python
from sklearn.feature_selection import VarianceThreshold 
selector = VarianceThreshold(threshold = 1)
selector.fit(df)
mask = selector.get_support()
print(mask)
reduced_df = df.loc[:, mask]
```

cons: varaince values aren't comparable across features

```python
sel = VarianceThreshold(threshold = 0.005) # inspect data visually when choosing threshold
sel.fit(df/df.mean())
```

If there are a lot of missing values in features, drop them.


```python
mask = df.isna().sum()/len(df) < 0.3 # drop columns with more than 30% missing values
```

## Pairwise correlation

```python
corr = df.corr()
cmap = sns.diverging_palette(h_neg = 10, h_pos = 240, as_cmap = True)
sns.heatmap(corr, annot = True, cmap = cmap)

mask = np.triu(np.ones_like(corr, dtype = bool)) # set upper triangle to false
sns.heatmap(corr, mask = mask, annot = True, cmap = cmap)
```

## Removing highly correlated features

Features that are perfectly correlated , 1 or -1 add no new information.

```python
corr_df = df.corr().abs()
mask = np.triu(np.ones_like(corr, dtype = bool))
tri_df = corr_df.mask(mask)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
```

This is a brute force method.

Correlation caveat: Anscombe's quartet

## Selecting features for model performance

```python
print(dict(zip(Xdf.columns,abs(lr_coef_[0])))) # example with logistic regression model
```
if this was standardized first, they are comparable. Lower coefficients are less important.

Recursive Feature Elimination (RFE): 

```python
from sklearn.feature_selection import RFE
rfe = RFE(estimator = LogisticRegression(), n_features_to_select = 2, verbose = 1)
rfe.fit(X_train_std, y_train)
X.columns[rfe.support_]
print(dict(zip(X.columns, rfe.ranking_)))
# value of 1 measures the feature was kept till the end
print(accuracy_score(y_test, rfe.predict(X_test_std)))
```

## Tree based feature selection

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(y_test, rf.predict(X_test)))

print(rf.feature_importances_) # they sum upto 1
mask = rf.feature_importances_ > 0.1
```

Based on what features are used more often to make decisions, and what are more near the root ie important decisions, we can find the most important features.

We should drop one by one.
Wrap model with RFE, better than having a single threshold.
RFE uses the model to calculate, so need to use it wtih a suitable model, usually tree based

```python
rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 6, verbose = 1, step = 10)
# each iteration drops 10 features
rfe.fit(X_train, y_train)
print(X.columns[rfe.support_])
```

## Regularized Linear regression

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)
# r squared, how much variance in the target variable is explained by the model
print(lr.score(X_test, y_test))
```

Loss Function: Mean Squared
We dont want overfitting, so we add regularization term to the loss function.

\[
\text{Loss Function} = \text{MSE} + \alpha \sum_{i=1}^n |\beta_i|
\]
Too low {\alpha} leads to overfitting, too high leads to underfitting, too simplistic.

One linear model that includes regularization is Lasso regression. Least Absolute Shrinkage and Selection Operator. - L1 regularization

## Combining feature selectors

Automating finding the alpha value for Lasso regressor:

```python
from sklearn.linear_model import LassoCV
lcv = LassoCV()
lcv.fit(X_train, y_train)
print(lcv.alpha_) # best alpha value

mask = lcv.coef_ != 0 # features that are not zero

lcv.score(X_test, y_test)
```

Feature selection with random forest:

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# make it select teh same number of features as Lasso
rfe_rf = RFE(estimator = RandomForestClassifier(), n_features_to_select = 66, steps = 5, verbose = 1)
rfe_rf.fit(X_train, y_train)
rf_mask = rfe_rf.support_
```

Feature selection with gradient boosting:

```python
from sklearn.ensemble import GradientBoostingClassifier

rfe_gb = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = 66, steps = 5, verbose = 1)
rfe_gb.fit(X_train, y_train)
gb_mask = rfe_gb.support_
```

Combining the features:

```python
votes = np.sum([mask, rf_mask, gb_mask], axis = 0)
print(votes)
final_mask = votes >= 2
```

# Feature extraction

Feature generation: eg BMI from weight and height, or polynomial features, or average of right and left eye measurements.

## PCA

Scale the features first 
```python
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
df_std = scaler.fit_transform(df)
pca = PCA()
pca.fit_transform(df_std)
print(pca.explained_variance_ratio_) # typically concertrated in the first few components
print(pca.explained_varaince_ratio_.cumsum()) # check how many components to keep

#Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)

#This changes the numpy array output back to a DataFrame
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])
```

Resulting point cloud shows no correlation hence no more duplicate information

## PCA applications

```python
print(pca.components_) # the loadings
```
use to interpret the new features

since we scale the data first, combine in pipeline

```python
from sklearn.preprocessing import StandardScaler()
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', PCA())
])

pc = pipe.fit_transform(df)
```

PCA is not preferred for categorical datasets

```python
sns.scatterplot(data = df, x = pc[:, 0], y = pc[:, 1], hue = df['column1'], alpha = 0.4)
```

PCA in a model pipeline

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', PCA(n_components = 3)),
    ('classifier', RandomForestClassifier())
])
print(pipe['reducer'].explained_variance_ratio_.sum())
print(pipe.score(X_test, y_test))
```
Combine the catagorical features dataframe with PCA features
Increasing number of components increases the information but not necessarily the model performance

PCA selection:

pass in n_components = 0.9 to keep 90% of the variance

```python
plt.plot(pipe['reducer'].explained_variance_ratio_)
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance')
```
Elbow in the plot shows where to cut off

X -> pca.fit(X) -> Fitted pca (numpy array of pc) -> pca.transform(X) -> Transformed X pc
pc -> pca.inverse_transform(pc) -> X

relevant for image compression

Look into categorical features and text data for feature extraction