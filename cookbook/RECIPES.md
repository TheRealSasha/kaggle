# Recipes: A Companion to the Cookbook

These are checklists to follow. The notebooks show the code. Links point to the notebook where each step is shown.

| Notebook | Covers |
|---|---|
| [`time_series/`](time_series) | trend, seasonality, lags, hybrids, multistep forecasting |
| [`regression/`](regression/linear_models_ols_ridge_lasso_pca.ipynb) | OLS → Ridge → Lasso → PCA |
| [`classification/`](classification/logistic_regression_and_nonlinear_classifiers.ipynb) | logistic regression + non-linear classifiers |
| [`pipelines_and_metrics/`](pipelines_and_metrics) | transformers, pipelines, regression & classification metrics, statsmodels tests |
| [`model_comparison/`](model_comparison/model_comparison_framework.ipynb) | baseline → many models → compare → tune top few → test (the framework for step 9) |

---

## Recipe 1 · New dataset → first submission

1. **Frame the task.**
   - What is the target? What type is it: regression, binary, multiclass, or a forecast?
   - What is the **competition metric**? It decides every later choice.
   - What will be known at prediction time? Anything that isn't is leakage.
2. **Load and inspect.** `df.shape`, `df.info()`, `df.describe()`, `df.isna().sum()`, `y.value_counts()` or `y.hist()`.
3. **Choose the validation split before any modelling.** See the table below. Hold out a test set, or treat Kaggle's test set as the hold-out.
4. **Baseline.** `DummyRegressor(strategy="mean")` or `DummyClassifier(strategy="most_frequent")`. Every model must beat it.
5. **EDA, to choose features.**
   - Target distribution: skewed → model `log1p(y)`.
   - Class balance: imbalanced → use AUC or F1 and `class_weight`.
   - Correlations: collinear → Ridge or PCA.
   - Feature vs target plots.
   - For time series, see Recipe 4.
6. **Clean.**
   - Drop leaky columns (filled in only after the outcome) and ID columns.
   - Fix dtypes and decide what missing means: "absent" (fill a constant) or "unknown" (impute).
7. **Feature engineering.** Ratios, counts, dates (day of week, month), text extracts (e.g. title from name), group aggregates.
8. **Preprocessing pipeline.** `ColumnTransformer` with a numeric branch (impute → scale) and a categorical branch (impute → one-hot). See Recipe 2.
9. **Model ladder.** Linear model → random forest → gradient boosting. Compare them with the **same CV folds** using [`compare()`](model_comparison/model_comparison_framework.ipynb).
10. **Tune** the best one or two with `GridSearchCV` on the whole pipeline.
11. **Diagnose.** Use out-of-fold predictions: residual plots, confusion matrix, learning curve, the worst errors.
12. **Final fit** on all the training data → predict on test → clip to the valid range → write the submission.

### Choosing the validation split
| Data | Split | sklearn |
|---|---|---|
| i.i.d. rows | random K-fold | `KFold(5, shuffle=True)` |
| classification | stratified K-fold | `StratifiedKFold(5, shuffle=True)` |
| repeated entities (users, patients, stores) | group K-fold, so the same entity is never in both train and validation | `GroupKFold(5)` |
| time series | past → future, never shuffled | `TimeSeriesSplit`, or hold out the last H periods |

---

## Recipe 2 · Fitting any model in 7 steps

```
1. split        X_train, X_test, y_train, y_test = train_test_split(..., stratify=y)  # or a time split
2. preprocess   ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
3. pipeline     model = Pipeline([("preprocess", preprocess), ("model", Estimator())])
4. CV score     cross_validate(model, X_train, y_train, cv=cv, scoring=[...])
5. tune         GridSearchCV(model, {"model__param": [...]}, cv=cv)
6. diagnose     cross_val_predict → residual / confusion / calibration / learning curve
7. final        model.fit(X_train, y_train); score once on X_test; refit on all → predict
```
Rules:
- **Fit nothing outside the pipeline.** That includes scalers, imputers, encoders and feature selection.
- **Use the same folds for every model you compare.**
- **Touch the test set once, at the end.**

### Preprocessing by model family
| | Scaling | Categoricals | Missing values | Skewed target / features |
|---|---|---|---|---|
| **Linear** (OLS, Ridge, Lasso, Logistic) | **required** for penalized models; recommended for OLS | one-hot | impute | log-transform helps |
| **Distance / kernel** (k-NN, SVM) | **required** | one-hot | impute | log-transform helps |
| **Neural net** (MLP) | **required** | one-hot | impute | log-transform helps |
| **Trees** (DecisionTree, RF, ExtraTrees) | not needed | ordinal or one-hot | NaN OK in sklearn ≥ 1.4, otherwise impute | no effect on splits |
| **Boosting** (HistGB, XGBoost) | not needed | ordinal / native categorical | **native NaN support** | log target can still help |
| **PCA** | **required** (standardize first) | numeric only | impute | — |

---

## Recipe 3 · Model reference

### Regression and general-purpose models
| Model | sklearn / lib | Learns | Key assumptions | Scale? | Tune first | Extrapolates? | Notebook |
|---|---|---|---|---|---|---|---|
| **OLS** | `LinearRegression` | linear | linear relationship; independent errors; constant variance; normal residuals *(for inference only)*; no perfect collinearity | recommended | — | ✅ | regression, pm/02 |
| **Ridge** (L2) | `Ridge`, `RidgeCV` | linear, shrunk | as OLS; handles collinearity | **yes** | `alpha` | ✅ | regression, pm/02 |
| **Lasso** (L1) | `Lasso`, `LassoCV` | sparse linear (selects features) | as OLS; among correlated features it picks one arbitrarily | **yes** | `alpha` | ✅ | regression, pm/02 |
| **Elastic Net** | `ElasticNetCV` | L1 + L2 blend | as OLS | **yes** | `alpha`, `l1_ratio` | ✅ | (mentioned) |
| **PCA + OLS** | `PCA` → `LinearRegression` | regression on components | linear; high-variance components are assumed predictive (not guaranteed) | **yes** | `n_components` | ✅ | regression |
| **k-NN** | `KNeighbors{Regressor,Classifier}` | local average | nearby points have similar targets; few dimensions | **yes** | `n_neighbors` | ❌ | classification, ts/05 |
| **Decision tree** | `DecisionTree*` | axis-aligned splits | none distributional; overfits alone | no | `max_depth`, `min_samples_leaf` | ❌ | classification |
| **Random forest** | `RandomForest*` | average of de-correlated trees | none; needs enough rows | no | `max_features`, `min_samples_leaf` (`n_estimators`: more is safe) | ❌ | classification, ts/05 |
| **Extra trees** | `ExtraTrees*` | random-split forest | as RF | no | as RF | ❌ | (mentioned) |
| **Gradient boosting** | `HistGradientBoosting*` | trees fit to the previous trees' errors | none | no | `learning_rate`, `max_iter`, `max_depth` | ❌ | classification |
| **XGBoost** | `XGBRegressor` / `XGBClassifier` | boosted trees | none | no | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | ❌ | ts/05, ts/06 |
| **SVM (RBF)** | `SVC`, `SVR` | smooth non-linear boundary | scaled features; medium-sized data | **yes** | `C`, `gamma` | ❌ | classification |
| **MLP** | `MLP{Regressor,Classifier}` | non-linear function | scaled features; more data | **yes** | `hidden_layer_sizes`, `alpha` | partly | ts/05 |

### Classification
| Model | Key assumptions | Scale? | Tune first | Probabilities calibrated? |
|---|---|---|---|---|
| **Logistic regression** | log-odds linear in the features; independent rows; little collinearity; roughly ≥ 10 events per feature | **yes** | `C`, `penalty` (`l1` needs `solver="liblinear"`) | ✅ usually |
| **k-NN, tree, RF, boosting, SVM, MLP** | as in the table above | as above | as above | ⚠️ often not, so check `CalibrationDisplay` |

### Inference (statsmodels), for p-values rather than prediction
| Model | Call | Checks |
|---|---|---|
| **OLS** | `sm.OLS(y, sm.add_constant(X)).fit()` | t-tests, `f_test`, Breusch–Pagan (→ `cov_type="HC3"`), VIF > 5–10 means collinearity, Durbin–Watson ≈ 2, Jarque–Bera |
| **Logit** | `smf.logit("y ~ C(cat) + x", df).fit()` | Wald z-tests, odds ratios `exp(coef)`, likelihood-ratio test between nested models |

### Time-series models
| Model | Use for | Notes | Notebook |
|---|---|---|---|
| `DeterministicProcess(order=k)` + LR | trend | keep the order at 1–2; high orders blow up | ts/02 |
| `seasonal=True` + `CalendarFourier(freq, order)` + LR | seasonality | indicators for short seasons, Fourier for long; the periodogram sets the order | ts/03 |
| LR / GBDT on lags, leads, rolling stats | cycles, leading indicators | shift before rolling; use leads only for values known in advance | ts/04 |
| **Boosted hybrid** (LR → XGB on residuals) | trend + interactions | trees can't extrapolate, so the LR model takes the trend | ts/05 |
| MultiOutput / Direct / Recursive / DirRec | horizons longer than 1 step | `LinearRegression` / `MultiOutputRegressor` / a loop / `RegressorChain` | ts/06 |

---

## Recipe 4 · Time series dataset

1. **Index.** Parse dates → `to_period("D")` → sort. Check for missing dates (`period_range(...).difference(idx)`).
2. **Define the task.** Origin (last known time), horizon (H steps), lead time. This decides which lags are allowed.
3. **Trend.** Plot a moving average with a window longer than the longest season → `DeterministicProcess(order=1 or 2)`.
4. **Seasonality.** Seasonal plot + periodogram → `seasonal=True` + `CalendarFourier`. Check that the deseasonalized periodogram is flat.
5. **Holidays and events.** One-hot dummies joined on date.
6. **Cycles.** Deseasonalize → PACF and lag plots → `make_lags`. Add leading indicators (`make_leads` only for planned values).
7. **Model.** LR on deterministic features, then a boosted hybrid (LR trend → GBDT on residuals).
8. **Multistep.** For H > 1 with lags, build `make_multistep_target` → choose Direct, DirRec or Recursive.
9. **Validate** on the last H periods (or `TimeSeriesSplit`). Leave a gap of H−1 rows in multistep splits.

---

## Recipe 5 · Choosing the metric

| Task | Metric | When |
|---|---|---|
| Regression | **RMSE** | default; large misses matter most |
| | **MAE** | outliers shouldn't dominate |
| | **RMSLE** / MAPE | relative error matters; the target spans orders of magnitude (prices, sales) |
| | **R²** | share of variance explained (0 = no better than the mean) |
| Classification (labels) | accuracy | balanced classes, equal error costs |
| | **precision / recall / F1** | uneven costs: recall if misses are costly, precision if false alarms are |
| | balanced accuracy | imbalanced classes |
| Classification (scores) | **ROC AUC** | ranking quality, regardless of threshold |
| | **PR AUC** (average precision) | rare positive class |
| | **log loss / Brier** | the probabilities themselves matter |

The threshold is chosen separately from the model: `TunedThresholdClassifierCV(model, scoring="f1")`.

---

## Diagnosis: symptom → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| CV score ≫ leaderboard or test score | leakage, or a split that doesn't match the test | put preprocessing in the pipeline; use a time or group split |
| Train ≈ validation, both poor | underfitting (high bias) | more features, interactions, a non-linear model |
| Train ≫ validation | overfitting (high variance) | regularize (↑`alpha`, ↓`C`, ↓depth), more data, fewer features |
| Curved residuals vs predicted | missing non-linearity | log or polynomial features, splines, a tree model |
| Funnel-shaped residuals | heteroskedasticity | log target; robust SEs (`HC3`) for inference |
| Unstable or huge coefficients | collinearity | Ridge, PCA, drop features (check VIF) |
| `ConvergenceWarning` | unscaled features or weak regularization | scale; raise `max_iter` |
| Tree forecast flat beyond the training range | trees can't extrapolate | boosted hybrid with a linear trend |
| Lag model can't forecast past step 1 | lags aren't available in the future | multistep strategy (ts/06) |
| Accuracy high but model is useless | class imbalance | balanced accuracy, AUC, F1; `class_weight="balanced"` |
