# Day 16 Capstone — Predict Customer Churn

## End-to-End Classical Machine Learning

This capstone consolidates the complete Classical ML workflow into a realistic customer-churn prediction project.

The project answers the business question:

> Given everything we know about a customer today, can we identify customers who are likely to churn so the retention team can intervene before they leave?

The project covers the full workflow from raw data to an actionable business recommendation.

---

## Deliverable

```text
day16_capstone.ipynb
README_day16_capstone.md
```

The notebook should be run from top to bottom before submission.

---

# Dataset

No internet connection is required.

The telecom-style customer dataset is generated entirely in memory using:

```python
sklearn.datasets.make_classification
```

The generated dataset contains:

```text
4000 customers
11 legitimate predictive features
1 deliberate leakage feature
1 binary churn target
```

The churn rate is approximately:

```text
27.5%
```

Therefore this is an **imbalanced binary classification problem**.

---

# Business Objective

The prediction target is:

```text
churn = 0 → customer stays
churn = 1 → customer leaves
```

Because losing a customer can be much more expensive than contacting a customer who would have stayed, the project prioritizes:

```text
Recall on the churn class
```

For threshold-independent model comparison, it uses:

```text
ROC-AUC
```

---

# Workflow

```text
Business Problem
      ↓
EDA
      ↓
Leakage Detection
      ↓
Train / Validation / Test Split
      ↓
ColumnTransformer
      ↓
Scaling + One-Hot Encoding
      ↓
Model Comparison
      ↓
GridSearchCV
      ↓
Validation Threshold Selection
      ↓
Final Test Evaluation
      ↓
Model Interpretation
      ↓
Business Recommendation
```

---

# Step 1 — Exploratory Data Analysis

The notebook investigates:

- churn-class balance;
- average numeric values for stayers and churners;
- tenure differences;
- support-ticket behaviour;
- login inactivity;
- churn by contract type; and
- visual differences between churners and non-churners.

Important hypotheses can then be compared with the final model interpretation.

---

# Step 2 — Leakage Detection

The synthetic dataset intentionally contains:

```text
exit_survey_sent
```

This is a leakage feature.

An exit survey is normally triggered because the customer has already left or entered the cancellation process. Therefore it would not be legitimately available when predicting future churn.

The notebook calculates correlations and flags features with:

```text
absolute correlation > 0.90
```

The leakage feature is removed before modelling.

This demonstrates an important rule:

> A predictive feature is only valid if it is available at the real prediction time.

---

# Train / Validation / Test Split

The clean dataset is divided into:

```text
70% training
15% validation
15% test
```

Approximate sizes:

```text
Training   → 2800 customers
Validation → 600 customers
Test       → 600 customers
```

Stratification preserves the churn ratio across all three datasets.

The three datasets have different jobs.

### Training

Used for:

```text
model fitting
cross-validation
GridSearchCV
```

### Validation

Used for:

```text
confirming generalization
choosing the operating threshold
permutation importance
```

### Test

Used only for the final performance estimate.

This prevents the test set from becoming part of model development.

---

# Step 3 — Preprocessing Pipeline

The project contains:

```text
7 numeric features
4 categorical features
```

Numeric features are processed using:

```python
StandardScaler()
```

Categorical features use:

```python
OneHotEncoder(handle_unknown="ignore")
```

Both transformations are combined using:

```python
ColumnTransformer
```

and included inside a:

```python
Pipeline
```

The complete preprocessing flow is therefore:

```text
Raw Customer Data
        ↓
ColumnTransformer
        ↓
┌────────────────────┬──────────────────────┐
│ Numeric            │ Categorical          │
│ StandardScaler     │ OneHotEncoder        │
└────────────────────┴──────────────────────┘
        ↓
19 model features
        ↓
Classifier
```

Keeping preprocessing inside the Pipeline prevents train/test preprocessing leakage.

---

# Step 4 — Model Comparison

Four model families are compared:

```text
Logistic Regression
Random Forest
Gradient Boosting
RBF SVM
```

Every model uses:

```text
5-fold Stratified Cross-Validation
```

The comparison metric is:

```text
ROC-AUC
```

All models use the same preprocessing structure so the comparison is fair.

Expected scores are around:

```text
ROC-AUC ≈ 0.94
```

Exact scores may vary slightly by scikit-learn version.

---

# Step 5 — Hyperparameter Tuning

Gradient Boosting is tuned using:

```python
GridSearchCV
```

The search investigates:

```text
n_estimators:
150
300

learning_rate:
0.05
0.10

max_depth:
2
3
```

The grid contains:

```text
2 × 2 × 2 = 8
```

parameter combinations.

With 5-fold cross-validation this results in:

```text
40 fits
```

plus subsequent refitting of the selected estimator.

The validation set is not used by GridSearchCV.

---

# Step 6 — Threshold Selection and Test Evaluation

A classifier produces a churn probability such as:

```text
Customer A → 0.82
Customer B → 0.61
Customer C → 0.34
Customer D → 0.12
```

Using the default threshold:

```text
threshold = 0.50
```

means:

```text
probability >= 0.50 → churn
probability < 0.50  → stay
```

For retention, missing churners may be especially expensive.

The notebook therefore compares thresholds such as:

```text
0.50
0.40
0.35
0.30
0.25
```

using the **validation data**.

The selected operating point is:

```text
0.30
```

The threshold is frozen before evaluating the test set.

This is important because repeatedly choosing thresholds from test performance would leak test-set information into the final decision.

---

# Evaluation Metrics

The final model is evaluated with:

```text
ROC-AUC
Precision
Recall
F1
Confusion Matrix
```

## Recall

For churn:

```text
Recall = TP / (TP + FN)
```

It answers:

> Of all customers who actually churned, how many did we identify?

## Precision

```text
Precision = TP / (TP + FP)
```

It answers:

> Of all customers we flagged for retention, how many actually churned?

## Business Trade-Off

Lowering the threshold typically:

```text
increases recall
decreases precision
```

This means:

```text
more churners caught
        +
more unnecessary retention contacts
```

The correct operating threshold therefore depends on the relative cost of:

```text
missed customer churn
vs
retention intervention
```

---

# Step 7 — Model Interpretation

The project uses two interpretation methods.

## Permutation Importance

Permutation importance measures how much ROC-AUC decreases when one input column is randomly shuffled.

If shuffling a feature substantially damages performance, that feature is important to the model.

Potential important features include:

```text
tenure
total charges
contract type
support tickets
```

The actual ranking must come from the executed notebook.

---

## Logistic Regression Coefficients

An additional Logistic Regression model provides signed coefficients.

Positive coefficients push predictions toward:

```text
churn
```

Negative coefficients push predictions toward:

```text
stay
```

This makes the model easier to communicate to business stakeholders.

For example, the generated data may show risk associated with patterns such as:

```text
month-to-month contracts
electronic-check payments
support-ticket activity
fiber-optic service
```

while longer tenure and longer contracts may be protective.

The executed notebook should be treated as the source of truth for the final ranking.

---

# Step 8 — Business Recommendation

The final deliverable is not just an ROC-AUC score.

The notebook ends with a recommendation for the retention team.

A good final recommendation should include:

1. How accurately the model ranks churn risk.
2. The chosen operating threshold.
3. Recall at that threshold.
4. Precision at that threshold.
5. The most actionable churn drivers.
6. A concrete retention intervention.

An example structure is:

> RetailMind can identify customers at elevated churn risk using account and plan information available before cancellation. We recommend operating the tuned Gradient Boosting model at the threshold selected from validation data, prioritizing recall because a missed churner is expected to cost more than an unnecessary retention contact. Retention campaigns should focus on high-risk customers showing the strongest actionable risk patterns, such as risky contract structures or repeated support friction. The interventions should be tested through a controlled retention experiment so that model accuracy translates into measurable incremental retained revenue.

Replace generic wording with the actual metrics produced by the notebook.

---

# Reflection

The notebook addresses four important production ML questions.

## Leakage

A column such as:

```text
account_closed_date
cancellation_reason
final_exit_status
```

could leak churn if it is created only after cancellation.

Leakage should be investigated by understanding **when a feature becomes available**, not only by looking at correlation.

---

## Metric Choice

Recall makes sense when:

```text
cost of missing a churner
>>
cost of contacting a stayer
```

Precision becomes more important when the intervention itself is expensive.

For example, if every churn alert causes a costly service credit or human retention call, too many false positives could make the program unprofitable.

---

## Model Choice

If Logistic Regression, Random Forest, Gradient Boosting, and SVM all produce ROC-AUC within approximately `0.005`, increasing model complexity may provide little value.

Further improvement may instead come from better inputs such as:

```text
recent usage decline
failed payments
support-call history
customer complaints
plan-price changes
network-quality history
promotion expiration
```

---

## Honest Test Evaluation

The test set should simulate future unseen customers.

If the test set is repeatedly used to choose:

```text
models
hyperparameters
features
thresholds
```

then it is no longer truly unseen.

The reported test performance becomes optimistically biased.

This project therefore uses:

```text
training → model fitting and CV
validation → threshold selection
test → final evaluation
```

---

# Final Scorecard

The notebook automatically prints:

```text
primary_metric
chosen_model
test_roc_auc
operating_threshold
recall_at_0.30
precision_at_0.30
top_drivers
```

Do not manually invent these values.

Run the notebook and use the measured results in the final recommendation.

---

# Expected Project Structure

```text
RetailMind/
│
├── day16_capstone.ipynb
└── README_day16_capstone.md
```

No CSV is necessary because the customer dataset is generated inside the notebook.

---

# Submission Checklist

Before submission:

- [ ] Restart the Jupyter kernel.
- [ ] Run all cells from top to bottom.
- [ ] Confirm dataset shape is `(4000, 13)`.
- [ ] Confirm churn rate is approximately `0.275`.
- [ ] Confirm `exit_survey_sent` is detected as leakage.
- [ ] Confirm the leakage feature is removed.
- [ ] Confirm splits are `2800 / 600 / 600`.
- [ ] Confirm preprocessing produces 19 features.
- [ ] Confirm all four model families complete cross-validation.
- [ ] Confirm GridSearchCV completes successfully.
- [ ] Confirm the threshold is selected using validation data.
- [ ] Confirm final test metrics print.
- [ ] Confirm confusion matrices render.
- [ ] Confirm permutation importance renders.
- [ ] Update the manager recommendation with actual results.
- [ ] Review the reflection answers.
- [ ] Save the notebook with outputs visible.
- [ ] Confirm `Kernel → Restart & Run All` completes without errors.

---

# Git Upload

```bash
git add day16_capstone.ipynb README_day16_capstone.md
git commit -m "Complete Day 16 customer churn capstone"
git push
```

---

# Key Takeaway

This capstone implements the complete Classical ML workflow:

```text
Frame
  ↓
Explore
  ↓
Find Leakage
  ↓
Split
  ↓
Preprocess
  ↓
Cross-Validate
  ↓
Compare Models
  ↓
Tune
  ↓
Select Threshold
  ↓
Test Once
  ↓
Interpret
  ↓
Recommend
```

The final output is not merely a classifier. It is a **leakage-aware, evaluated, interpreted churn decision system** with an operating threshold tied to the business cost of false negatives and false positives.
