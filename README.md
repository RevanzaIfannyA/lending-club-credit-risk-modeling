# 📊 Credit Risk Prediction Model - Lending Company Portfolio Analysis

## Project Status
This project is currently being **enhanced** to improve modeling robustness and production readiness. Originally completed as a Project-Based Internship (PBI) deliverable, it is now being upgraded with production-grade ML pipelines and deployment capabilities.

---

## 📌 Project Overview
This project was developed as part of a **Project-Based Internship (PBI)** with ID/X Partners in collaboration with Rakamin. It simulates a real-world business case from a lending company operating in the consumer lending sector.

**Business Problem:** Lending companies face challenges in accurately assessing credit risk, leading to high default rates and inefficient capital allocation.

**Objective:** Build a machine learning model to predict credit risk by classifying loans as "good" or "bad" based on historical loan data from 2007-2014.

---

## 🎯 Project Objectives
1. Perform end-to-end data science workflow: EDA, cleaning, feature engineering, and modeling
2. Develop and compare multiple classification models for credit risk prediction
3. Identify the most suitable model based on business-relevant metrics
4. Generate actionable insights for stakeholders through clear visualizations
5. **Enhance with production-ready pipelines and deployment capabilities** (Current focus)

---

## 📊 Dataset
- **Source:** Internal dataset from internship program
- **File:** `loan_data_2007_2014.csv`
- **Records:** 466,285 loan applications
- **Period:** 2007-2014
- **Features:** 75+ columns including borrower demographics, loan characteristics, credit history, and loan performance

---

## 🔧 Technical Approach

### Current Enhancement Focus
The project is undergoing significant upgrades to transform it from an internship deliverable to a production-ready solution:

1. **Pipeline Refactoring:** Implementing `ColumnTransformer` and `Pipeline` for proper data separation
2. **Leakage Prevention:** Ensuring all preprocessing steps are fitted only on training data
3. **Deployment Architecture:** Adding model serving capabilities with FastAPI/Streamlit
4. **Code Modularization:** Separating concerns into configurable, reusable components

### Data Preprocessing & Feature Engineering
- **Data Cleaning:** Removed irrelevant columns (identifiers, post-loan leakage variables, noisy text fields)
- **Missing Value Handling:** Domain-informed imputation strategies with missing indicators
- **Feature Engineering:** 
  - Credit history duration calculation
  - Regional mapping from state codes
  - Employment length normalization
  - Temporal feature extraction
- **Outlier Treatment:** Business-aware capping and transformation strategies

### Target Variable Engineering
- **Good Loans (0):** "Fully Paid", "Does not meet the credit policy. Status:Fully Paid"
- **Bad Loans (1):** "Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"
- **Excluded:** Ongoing/transitional states to ensure clean target labeling
- **Class Distribution:** 81% Good Loans, 19% Bad Loans (after filtering)

---

## 🧠 Modeling & Evaluation

### Data Preparation Strategy
- **Train-Test Split:** 80-20 split with proper stratification
- **Class Imbalance:** SMOTE oversampling applied with pipeline integration
- **Feature Scaling:** StandardScaler within pipeline to prevent leakage
- **Cross-Validation:** Nested CV for hyperparameter tuning

### Models Evaluated
1. Logistic Regression (Baseline)
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. **Ensemble approaches** (in progress)

### Key Engineering Improvements
- **Pipeline Architecture:** All preprocessing encapsulated in scikit-learn compatible transformers
- **Column-specific Treatment:** Different strategies for numerical, categorical, and temporal features
- **Reproducibility:** Deterministic transformations with proper random state management
- **Validation Strategy:** Proper separation of validation data for early stopping

---

## 🚀 Deployment Architecture (In Progress)

The enhanced version includes deployment capabilities:
- **REST API** with FastAPI for model serving
- **Interactive Dashboard** with Streamlit for business user exploration
- **Model Monitoring** infrastructure for performance tracking
- **A/B Testing** framework for model comparison

### Reference Implementation
For a complete example of a production-ready ML pipeline with proper data leakage prevention and deployment, see my other project:  
**[Customer Churn Modeling with ANN](https://github.com/RevanzaIfannyA/Churn_Modeling_Classification_ANN_DLProject)**

This reference project demonstrates:
- ✅ Proper train-test separation with pipeline integration
- ✅ Deployment-ready architecture
- ✅ Comprehensive model evaluation
- ✅ Business metric alignment

---

## 📈 Key Insights from Analysis

### Business Impact Findings
- Debt consolidation loans represent the highest volume but show variable risk patterns
- Geographic regions show distinct risk profiles requiring localized strategies
- Interest rate is a strong predictor but must be balanced with business constraints
- Credit history features show non-linear relationships with default risk

### Model Performance
The enhanced pipeline focuses on:
- **Business metrics** over pure accuracy (focus on recall for high-risk detection)
- **Interpretability** for stakeholder trust
- **Operational efficiency** for real-time predictions
- **Robustness** across different economic cycles

---

## 🔄 Project Evolution Timeline

### Phase 1: Internship Deliverable (Completed)
- Exploratory analysis and baseline modeling
- Business insight generation
- Initial model comparison

### Phase 2: Pipeline Enhancement (Current)
- Refactoring preprocessing for production use
- Implementing proper data separation patterns
- Adding deployment infrastructure
- Enhancing documentation and reproducibility

### Phase 3: Advanced Features (Planned)
- Model interpretability with SHAP/LIME
- Automated retraining pipelines
- Real-time prediction capabilities
- Integration with business systems

---

## 🛠️ Technical Stack
- **Core:** Python 3.8+, Jupyter, Git
- **Data Processing:** Pandas, NumPy, Scikit-learn pipelines
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** FastAPI, Streamlit, Docker
- **Monitoring:** MLflow, Prometheus, Grafana (planned)

---

## 📚 Learning & Professional Development

This project represents a significant learning journey:

### Technical Skills Demonstrated
- End-to-end ML system design
- Data leakage prevention and pipeline architecture
- Business metric alignment in model evaluation
- Production deployment considerations

### Professional Growth
- **Iterative Improvement:** Recognizing limitations in initial approach and implementing solutions
- **Engineering Mindset:** Moving from proof-of-concept to production-ready code
- **Stakeholder Communication:** Balancing technical rigor with business practicality
- **Quality Focus:** Implementing best practices for maintainable, testable code

### Key Takeaways
- The importance of proper data separation cannot be overstated
- Pipeline architecture is fundamental, not optional
- Business context drives technical decisions
- Professional growth comes from recognizing and addressing gaps

---

## 🔗 Related Resources & References

1. **Production ML Pipeline Reference:** [Churn Modeling Project](https://github.com/RevanzaIfannyA/Churn_Modeling_Classification_ANN_DLProject)
2. **Scikit-learn Pipeline Documentation**
3. **MLOps Best Practices Guides**
4. **Credit Risk Modeling Literature**

---

## 📄 License & Attribution
- **Dataset:** Provided for educational purposes as part of the internship program
- **Code:** Open-source for educational and portfolio purposes
- **Internship Program:** ID/X Partners x Rakamin Project-Based Internship
- **Enhancements:** Independent work demonstrating professional growth

---

*This project showcases both technical implementation skills and the professional maturity to iteratively improve solutions. The current enhancements focus on transforming an academic exercise into a production-ready system while maintaining clear documentation of the engineering decisions and learning process.*
