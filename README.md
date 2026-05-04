# Loan Default Prediction - LendingClub Credit Risk Modeling

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lending-club-credit-risk-modeling.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project was developed as the **Final Task** of the **Project-Based Virtual Internship Program** at **ID/X Partners** in collaboration with **Rakamin Academy**. The objective is to build a machine learning model that accurately predicts whether a loan applicant will default, enabling data-driven lending decisions that maximize financial returns.

**Live Demo:** [lending-club-credit-risk-modeling.streamlit.app](https://lending-club-credit-risk-modeling.streamlit.app)

**Key Achievements:**
- ✅ **ROC-AUC: 0.71** - Good discrimination between good and bad loans
- ✅ **Recall (Bad Loans): 56%** - Catches 56% of potential defaults
- ✅ **Net Business Impact: $33.3 Million** - Financial benefit on test portfolio
- ✅ **Deployed Streamlit App** - Interactive web application for real-time predictions

---

## 🎯 Business Problem

**How can LendingClub accurately predict which loan applicants are likely to default, enabling the platform to make data-driven approval decisions that maximize net financial return?**

In the lending business, approving a bad loan (default) results in a direct loss of the principal amount (\$13k on average), while rejecting a good loan results only in foregone interest (\$1.5k on average). This **1:8 loss-to-profit ratio** means catching defaults is far more important than avoiding false rejections.

### Key Business Metrics

| Metric | Value |
|--------|-------|
| Baseline Default Rate | 19.1% |
| **Model Default Rate (at threshold 0.55)** | **~11%** (56% reduction) |
| False Negative Cost (missed default) | $12,000 per loan |
| False Positive Cost (rejected good) | $1,500 per loan |
| **Net Business Impact** | **$33.3 Million** |

---

## 📊 Dataset

### Source
LendingClub loan data from 2007-2014 (publicly available)

### Size
- **Initial records:** 466,285 loans
- **Final training set:** 184,611 loans (after cleaning and filtering)
- **Features:** 35+ after feature engineering

### Target Variable
- **Bad Loan (1):** Charged Off, Default, Late (31-120 days)
- **Good Loan (0):** Fully Paid

---

## 🔧 Data Cleaning & Feature Engineering

### Data Cleaning
- Removed columns with >70% missing values
- Handled informative missing values (creating flags for `ever_delinquent`, `ever_public_record`, etc.)
- Capped outliers at 99th percentile for numeric features
- Converted `revol_util` from percentage to decimal format

### Feature Engineering

| Feature | Description | Business Value |
|---------|-------------|----------------|
| `credit_history_months` | Credit age in months | Long history = lower risk |
| `loan_to_income_ratio` | Loan amount / annual income | High ratio = higher risk |
| `total_negative_events` | Count of derogatory events | More events = higher risk |
| `recent_delinquency` | Delinquency in last 12 months | Recent = very high risk |
| `avg_balance_per_account` | Total balance / open accounts | High avg = potential stress |

### Key Insights Discovered

1. **Grade is the strongest predictor**: Grade G has 43.7% default rate (2.3x baseline)
2. **DTI risk is linear**: Each 5% DTI increase adds ~17% more risk
3. **Business investment is riskiest purpose**: 29.5% default rate (1.5x baseline)
4. **Term length has independent causal effect**: 60-month loans are 31% riskier even after controlling for grade
5. **Recent delinquency matters**: Defaults within 12 months increase risk by 12%
6. **Old delinquencies (>36 months) are safe**: Risk returns to normal levels

---

## 🧠 Modeling Approach

### Model Selection
**XGBoost Classifier** - Chosen for its ability to handle:
- Missing values natively
- Non-linear relationships
- Feature interactions automatically
- Class imbalance with scale_pos_weight

### Handling Class Imbalance
- Initial ratio: 81% Good / 19% Bad
- Applied **scale_pos_weight** parameter in XGBoost
- Optimized decision threshold for business metrics (not default 0.5)

### Hyperparameter Tuning
Used **HalvingGridSearchCV** (successive halving) for efficient search:
- 3-fold inner cross-validation
- 5-fold nested cross-validation for unbiased evaluation
- Tuned parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

### Feature Preprocessing (ColumnTransformer)

| Column Type | Transformation |
|-------------|----------------|
| Numeric | StandardScaler + median imputation |
| Ordinal (grade) | OrdinalEncoder (A→0, B→1, ..., G→6) |
| Nominal (purpose, region, etc.) | OneHotEncoder |

---

## 📈 Model Performance

### Test Set Results (n=46,155 loans)

| Metric | Default Threshold (0.5) | Optimized Threshold (0.55) |
|--------|------------------------|----------------------------|
| ROC-AUC | 0.71 | 0.71 |
| Recall (Bad) | 4% | **56%** |
| Precision (Bad) | 57% | 33% |
| F1-Score (Bad) | 0.08 | **0.42** |
| Net Business Impact | - | **$33.3M** |

### Confusion Matrix (Optimal Threshold = 0.55)

| | Predicted Good | Predicted Bad |
|---|---------------|---------------|
| **Actual Good** | 27,359 (TN) | 9,982 (FP) |
| **Actual Bad** | 3,841 (FN) | 4,973 (TP) |

### Financial Impact (Test Set)

| Component | Amount |
|-----------|--------|
| ✅ Profit from correct approvals (TN) | $69.4M |
| ✅ Savings from correct rejections (TP) | $79.7M |
| ❌ Loss from missed defaults (FN) | -$47.7M |
| ❌ Opportunity loss from false rejections (FP) | -$68.1M |
| **💰 NET BUSINESS IMPACT** | **$33.3M** |

---

## 🔍 Top 10 Business Insights from EDA

### Insight 1: Grade Analysis
Grade G has **43.7% default rate** (2.3x baseline). Approving 100 grade G loans causes $330k loss.

### Insight 2: DTI Threshold
Risk increases steadily with DTI. Above 25% DTI is dangerous (26.5% default rate).

### Insight 3: Loan Purpose
Business investment loans have **29.5% default rate** - 1.5x riskier than average.

### Insight 4: Verification Status (Counterintuitive!)
**"Not Verified" borrowers have the lowest default rate (15.2%)** - Verified borrowers perform worse!

### Insight 5: Delinquency Recency
Recent delinquencies (<12 months) increase risk, but **old delinquencies (>36 months) are safe** (18.7% default rate, below average).

### Insight 6: Grade × Purpose Interaction
Grade G + "Other" purpose = **50.9% default rate** - auto-reject territory.

### Insight 7: Temporal Trend
Default rates peaked in 2007 (27.1%) and stabilized around 16-21% post-crisis.

### Insight 8: Loan-to-Income Ratio
LTI > 50% shows **+74% risk spike** from 19.1% to 33.3% default rate.

### Insight 9: High-Risk Segment
Combination of (Grade E/F/G + DTI>25% + Business Purpose + Ever Delinquent) = **54% default rate** - only 0.02% of loans but extremely dangerous.

### Insight 10: Term Length (Causal Analysis)
After controlling for grade, 60-month loans still have **31% additional risk** - term has independent causal effect!

---

## 🚀 Live Demo

The model is deployed as an interactive web application:

**[https://lending-club-credit-risk-modeling.streamlit.app](https://lending-club-credit-risk-modeling.streamlit.app)**

### Features

**Single Prediction Mode:**
- Fill out borrower information form (loan details, credit history, financials)
- Get instant default probability and approval recommendation
- View risk meter and business impact assessment

**Batch Prediction Mode:**
- Upload CSV file with multiple loan applications
- Download predictions with risk levels and decisions
- View aggregate statistics and risk distribution

### Sample API Request (for developers)

```python
# The app accepts CSV uploads with the following required columns:
required_columns = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'annual_inc', 'verification_status',
    'issue_d', 'earliest_cr_line', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_last_major_derog', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'tot_coll_amt', 'tot_cur_bal',
    'total_rev_hi_lim', 'home_ownership', 'purpose_group', 'addr_region'
]
```

---

## 💻 Local Development

### Prerequisites
- Python 3.10
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/RevanzaIfannyA/lending-club-credit-risk-modeling.git
cd loan-default-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run loan_prediction_app.py
```

### Requirements

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
```

---

## 📊 Key Technical Decisions & Trade-offs

### Why XGBoost over Deep Learning?
- Tabular data with clear feature importance needs
- Faster training and inference (<1 second per prediction)
- Better interpretability for regulatory compliance
- Handles missing values natively

### Why Optimized Threshold (0.55) over Default (0.50)?
| Metric | Default (0.5) | Optimized (0.55) |
|--------|---------------|-------------------|
| Recall (Bad) | 4% | **56%** |
| Missed Defaults | 8,432 | **3,841** |
| Net Impact | Low | **$33.3M** |

**Rationale:** In lending, 1 missed default ($12k loss) costs 8x more than 1 rejected good loan ($1.5k opportunity loss). Lower threshold prioritizes catching defaults.

### Why No SMOTE?
SMOTE was tested but resulted in **15% recall** (vs 56% with weighted model). Weighted XGBoost with threshold tuning performed significantly better.

### Why No Deep Neural Networks?
- XGBoost outperforms DNNs on tabular data
- Faster training and inference
- Better interpretability for business stakeholders
- Lower computational requirements

---

## 📝 Lessons Learned

### Data Leakage Prevention
Initially computed statistics (median, quantiles) on full dataset before split. **Correct approach:** Calculate all statistics ONLY on training data, then transform test data.

### Missing Value Strategy
Not all missing values are equal. In credit data:
- Missing in `mths_since_last_delinq` = "Never delinquent" (good signal)
- Missing in `emp_length` = "Not disclosed" (neutral signal)

Created separate flag columns to preserve this information.

### Business Metrics > Technical Metrics
Model with 0.71 ROC-AUC but 4% recall was useless for business. Optimized for **recall (bad loans)** resulted in 56% recall with acceptable precision trade-off.

### Threshold Optimization > Class Weight
Class weight improved recall to 66%, but at cost of higher false positives. Best result came from **weighted model + threshold tuning** (56% recall, optimal business impact).

---

## 🔮 Future Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| **Economic indicator features** | Add unemployment rate, GDP growth as features | High |
| **Model retraining pipeline** | Automated monthly retraining with new data | High |
| **SHAP explanations** | Add model interpretability for each prediction | Medium |
| **A/B testing framework** | Test different thresholds in production | Medium |
| **API endpoint** | REST API for third-party integration | Low |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LendingClub** for providing public loan data
- **XGBoost** team for the excellent gradient boosting library
- **Streamlit** for making ML deployment accessible

## 📧 Contact

For questions or collaboration opportunities:
- **LinkedIn:** [Revanza Ifanny A.](www.linkedin.com/in/revanza-ifanny-ardiansyah-731513290)

---

## ⭐ Key Takeaways for Recruiters

This project demonstrates:

1. **End-to-end ML competence** - From data cleaning to deployment
2. **Business acumen** - Optimized for financial impact, not just accuracy
3. **Technical depth** - Nested CV, hyperparameter tuning, feature engineering
4. **Communication skills** - Clear insights with actionable recommendations
5. **Production readiness** - Deployed Streamlit app with batch prediction
6. **Data leakage awareness** - Correct train/test split handling
7. **Trade-off understanding** - Explainable decisions (why threshold 0.55 not 0.50)

**Portfolio differentiators:**
- ✅ $33.3M business impact calculation
- ✅ Counterintuitive insights (Verified borrowers perform worse!)
- ✅ Causal analysis (term length independent effect)
- ✅ Live interactive demo

---

**Built with ❤️ as part of ID/X Partners × Rakamin Academy Virtual Internship Program**
```