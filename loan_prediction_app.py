# ============================================================================
# STREAMLIT APP - loan_prediction_app.py (DENGAN CSV UPLOAD & CONDITIONAL UI)
# ============================================================================
# RUN WITH: streamlit run loan_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Default Prediction App")
st.markdown("Predict whether a borrower will default on their loan using XGBoost")

# ============================================================================
# LOAD ALL PREPROCESSING OBJECTS AND MODEL
# ============================================================================

@st.cache_resource
def load_all_objects():
    """Load all saved preprocessing objects and model"""
    model = joblib.load('models/best_model.joblib')
    threshold = joblib.load('models/optimal_threshold.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    column_info = joblib.load('models/column_info.joblib')
    mapping_emp_length = joblib.load('models/mapping_emp_length.joblib')
    avg_balance_cap = joblib.load('models/avg_balance_cap.joblib')
    cap_values = joblib.load('models/cap_values.joblib')
    purpose_options = joblib.load('models/purpose_options.joblib')
    home_ownership_options = joblib.load('models/home_ownership_options.joblib')
    return model, threshold, preprocessor, column_info, mapping_emp_length, avg_balance_cap, cap_values, purpose_options, home_ownership_options

model, threshold, preprocessor, column_info, mapping_emp_length, avg_balance_cap, cap_values, purpose_options, home_ownership_options = load_all_objects()

# ============================================================================
# FEATURE ENGINEERING FUNCTION (MUST MATCH TRAINING)
# ============================================================================

def create_features_from_input(df):
    """Create all derived features from basic user inputs"""
    df = df.copy()
    
    # Credit history months
    df['credit_history_months'] = (
        pd.to_datetime(df['issue_d']).dt.to_period('M').astype(int) - 
        pd.to_datetime(df['earliest_cr_line']).dt.to_period('M').astype(int)
    )
    df['credit_history_months'] = df['credit_history_months'].clip(lower=0).astype('Int64')
    
    # Loan to income ratio
    df['loan_to_income_ratio'] = (df['loan_amnt'] / df['annual_inc']).clip(upper=5.0)
    
    # Total negative events
    df['total_negative_events'] = (
        (df['delinq_2yrs'] > 0).astype(int) +
        (df['pub_rec'] > 0).astype(int) +
        df['ever_delinquent'] +
        df['ever_public_record'] +
        df['ever_major_derog']
    )
    
    # Recent delinquency
    df['recent_delinquency'] = ((df['mths_since_last_delinq'] > 0) & 
                                 (df['mths_since_last_delinq'] <= 12)).astype(int)
    
    # Average balance per account
    df['avg_balance_per_account'] = (df['tot_cur_bal'] / df['open_acc'].replace(0, np.nan)).fillna(0)
    
    # Cap at 99th percentile (use fixed value from training)
    df['avg_balance_per_account'] = df['avg_balance_per_account'].clip(upper=avg_balance_cap)
    
    return df

# ============================================================================
# APPLY CAPPING (SAME AS TRAINING)
# ============================================================================

def apply_capping(df):
    """Apply the same quantile capping as training"""
    df = df.copy()
    
    capping_columns = ['annual_inc', 'tot_cur_bal', 'tot_coll_amt', 'total_rev_hi_lim', 'revol_bal']
    
    for col in capping_columns:
        if col in df.columns and col in cap_values:
            df[col] = df[col].clip(upper=cap_values[col])
    
    return df

# ============================================================================
# PREPROCESS FUNCTION FOR DATAFRAME (BATCH PROCESSING)
# ============================================================================

def preprocess_dataframe(df):
    """Full preprocessing pipeline for a dataframe"""
    df = df.copy()
    
    # Apply feature engineering
    df = create_features_from_input(df)
    
    # Apply capping
    df = apply_capping(df)
    
    # Transform emp_length using mapping
    df['emp_length'] = df['emp_length'].map(mapping_emp_length)

    # Transform home_ownership using mapping
    df['home_ownership'] = df['home_ownership'].map(home_ownership_options)
    
    # Transform term to numeric
    df['term'] = df['term'].str.extract('(\d+)').astype('Int64')
    
    return df

# ============================================================================
# BATCH PREDICTION FUNCTION
# ============================================================================

def predict_batch(df):
    """Make predictions for a batch of data"""
    # Preprocess all data
    df_processed = preprocess_dataframe(df)
    
    # Make predictions
    probabilities = model.predict_proba(df_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Create results dataframe
    results = df.copy()
    results['default_probability'] = probabilities
    results['prediction'] = predictions
    results['decision'] = results['prediction'].map({1: 'REJECT', 0: 'APPROVE'})
    results['risk_level'] = results['default_probability'].apply(
        lambda x: 'Very High' if x > 0.50 else 'High' if x > 0.30 else 'Medium' if x > threshold else 'Low'
    )
    
    return results

# ============================================================================
# CREATE SAMPLE DATASET FROM X_TEST
# ============================================================================

@st.cache_data
def get_sample_data():
    """Create sample data from X_test (already saved during training)"""
    try:
        sample_df = pd.read_csv('Data/sample_test_data.csv')
        return sample_df
    except:
        st.warning("Sample data not found. Please ensure 'Data/sample_test_data.csv' exists.")
        return None

def create_sample_csv():
    """Create downloadable sample CSV template"""
    sample_template = pd.DataFrame({
        'loan_amnt': [10000, 20000, 5000],
        'term': ['36 months', '60 months', '36 months'],
        'int_rate': [0.13, 0.18, 0.10],
        'installment': [300, 500, 150],
        'grade': ['C', 'D', 'B'],
        'emp_length': ['5 years', '2 years', '10+ years'],
        'annual_inc': [60000, 45000, 80000],
        'verification_status': ['Verified', 'Source Verified', 'Not Verified'],
        'issue_d': [datetime.now(), datetime.now(), datetime.now()],
        'earliest_cr_line': [datetime(2015, 1, 1), datetime(2018, 6, 1), datetime(2010, 1, 1)],
        'dti': [0.16, 0.25, 0.12],
        'delinq_2yrs': [0, 1, 0],
        'inq_last_6mths': [1, 3, 0],
        'mths_since_last_delinq': [0, 15, 0],
        'mths_since_last_record': [0, 0, 0],
        'mths_since_last_major_derog': [0, 0, 0],
        'open_acc': [10, 8, 15],
        'pub_rec': [0, 0, 0],
        'revol_bal': [10000, 5000, 20000],
        'revol_util': [0.50, 0.70, 0.30],
        'total_acc': [20, 15, 25],
        'tot_coll_amt': [0, 500, 0],
        'tot_cur_bal': [80000, 50000, 120000],
        'total_rev_hi_lim': [22000, 15000, 35000],
        'home_ownership': ['MORTGAGE', 'RENT', 'OWN'],
        'purpose_group': ['debt_related', 'major_purchase', 'home_related'],
        'addr_region': ['Northeast', 'South', 'West'],
    })
    return sample_template

# ============================================================================
# SIDEBAR - MODE SELECTION
# ============================================================================

st.sidebar.header("📊 Prediction Mode")

prediction_mode = st.sidebar.radio(
    "Select prediction mode:",
    ["Single Prediction (Form)", "Batch Prediction (CSV Upload)"],
    index=0
)

# ============================================================================
# MODE 1: SINGLE PREDICTION (FORM) - DENGAN CONDITIONAL UI DI LUAR FORM
# ============================================================================

if prediction_mode == "Single Prediction (Form)":
    
    # ========================================================================
    # SEMUA INPUT YANG TIDAK CONDITIONAL DITARUH DI SIDEBAR (TANPA FORM)
    # ========================================================================
    
    st.sidebar.subheader("🏦 Loan Details")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=35000, value=10000, key="loan_amnt")
        term = st.selectbox("Loan Term", ["36 months", "60 months"], key="term")
    with col2:
        int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=26.0, value=13.0, key="int_rate") / 100
        installment = st.number_input("Monthly Installment ($)", min_value=15, max_value=1500, value=300, key="installment")
    
    grade = st.sidebar.select_slider("Loan Grade (A=Best, G=Worst)", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], value='C', key="grade")
    
    st.sidebar.subheader("👤 Borrower Profile")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        emp_length_options = list(mapping_emp_length.keys())
        emp_length = st.selectbox("Employment Length", emp_length_options, key="emp_length")
        annual_inc = st.number_input("Annual Income ($)", min_value=0, value=60000, key="annual_inc")
    with col2:
        home_ownership_list = list(home_ownership_options.keys())
        home_ownership = st.selectbox("Home Ownership", home_ownership_list, key="home_ownership")
        verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"], key="verification_status")
    
    st.sidebar.subheader("📅 Important Dates")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        issue_date = st.date_input("Loan Issue Date", datetime.now(), key="issue_date")
    with col2:
        earliest_credit_date = st.date_input("Earliest Credit Line Date", datetime(2010, 1, 1), key="earliest_credit_date")
    
    st.sidebar.subheader("📊 Financial Health")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        dti = st.slider("Debt-to-Income Ratio (%)", min_value=0, max_value=40, value=16, key="dti") / 100
        delinq_2yrs = st.number_input("Delinquencies (last 2 years)", min_value=0, max_value=29, value=0, key="delinq_2yrs")
    with col2:
        inq_last_6mths = st.number_input("Credit Inquiries (last 6 months)", min_value=0, max_value=33, value=1, key="inq_last_6mths")
        pub_rec = st.number_input("Public Records", min_value=0, max_value=11, value=0, key="pub_rec")
    
    st.sidebar.subheader("📈 Credit History")
    st.sidebar.markdown("⚠️ **Important:** Select 'Yes' below to enter months since event. '0 months' means NEVER occurred.")
    
    # ========================================================================
    # DELINQUENCY SECTION (Conditional UI - langsung bereaksi)
    # ========================================================================
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**Delinquency (Late Payment)**")
        has_delinquent = st.radio(
            "Has borrower ever been delinquent?",
            options=["No (0 months)", "Yes"],
            index=0,
            key="has_delinquent",
            horizontal=True,
            help="Delinquency = 30+ days late payment"
        )
        
        if has_delinquent == "Yes":
            months_delinq = st.number_input(
                "Months since last delinquency",
                min_value=1,
                max_value=152,
                value=12,
                step=1,
                key="months_delinq",
                help="1 = within last month, 12 = one year ago"
            )
            mths_since_last_delinq = months_delinq - 1
            ever_delinquent = 1
        else:
            mths_since_last_delinq = 0
            ever_delinquent = 0
    
    # ========================================================================
    # PUBLIC RECORD SECTION
    # ========================================================================
    
    with col2:
        st.markdown("**Public Record**")
        has_public_record = st.radio(
            "Has borrower ever had a public record?",
            options=["No (0 months)", "Yes"],
            index=0,
            key="has_public_record",
            horizontal=True,
            help="Public record = bankruptcy, tax lien, judgment"
        )
        
        if has_public_record == "Yes":
            months_public = st.number_input(
                "Months since last public record",
                min_value=1,
                max_value=129,
                value=24,
                step=1,
                key="months_public"
            )
            mths_since_last_record = months_public - 1
            ever_public_record = 1
        else:
            mths_since_last_record = 0
            ever_public_record = 0
    
    # ========================================================================
    # MAJOR DEROGATORY SECTION
    # ========================================================================
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**Major Derogatory**")
        has_major_derog = st.radio(
            "Has borrower ever had a major derogatory?",
            options=["No (0 months)", "Yes"],
            index=0,
            key="has_major_derog",
            horizontal=True,
            help="Major derogatory = charge-off, default, foreclosure"
        )
        
        if has_major_derog == "Yes":
            months_major = st.number_input(
                "Months since major derogatory",
                min_value=1,
                max_value=154,
                value=36,
                step=1,
                key="months_major"
            )
            mths_since_last_major_derog = months_major - 1
            ever_major_derog = 1
        else:
            mths_since_last_major_derog = 0
            ever_major_derog = 0
    
    # ========================================================================
    # OTHER CREDIT HISTORY
    # ========================================================================
    
    with col2:
        open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=76, value=10, key="open_acc")
    
    # ========================================================================
    # CREDIT CARDS & BALANCES
    # ========================================================================
    
    st.sidebar.subheader("💳 Credit Cards & Balances")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=80000, value=10000, key="revol_bal")
        revol_util = st.slider("Revolving Utilization (%)", min_value=0, max_value=100, value=50, key="revol_util") / 100
    with col2:
        total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=150, value=20, key="total_acc")
        tot_cur_bal = st.number_input("Total Current Balance ($)", min_value=0, max_value=600000, value=80000, key="tot_cur_bal")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tot_coll_amt = st.number_input("Total Collection Amount ($)", min_value=0, max_value=2000, value=0, key="tot_coll_amt")
    with col2:
        total_rev_hi_lim = st.number_input("Total Revolving Credit Limit ($)", min_value=0, max_value=120000, value=22000, key="total_rev_hi_lim")
    
    # ========================================================================
    # ADDITIONAL INFORMATION
    # ========================================================================
    
    st.sidebar.subheader("📍 Additional Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_label = st.selectbox("Loan Purpose", list(purpose_options.keys()), key="purpose")
        purpose_group = purpose_options[selected_label]
    with col2:
        addr_region = st.selectbox("Region", ["Northeast", "Midwest", "South", "West"], key="addr_region")
    
    # Auto-calculated flags
    emp_length_missing = 0
    has_credit_history = 1 if (tot_cur_bal > 0 or tot_coll_amt > 0 or total_rev_hi_lim > 0) else 0
    
    # ========================================================================
    # PREDICTION BUTTON (DI LUAR FORM, TAPI TETAP ADA)
    # ========================================================================
    
    submitted = st.sidebar.button("🔮 Predict Default Risk", type="primary", use_container_width=True)
    
    # ========================================================================
    # MAKE SINGLE PREDICTION
    # ========================================================================
    
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term],
            'int_rate': [int_rate],
            'installment': [installment],
            'grade': [grade],
            'emp_length': [emp_length],
            'annual_inc': [annual_inc],
            'verification_status': [verification_status],
            'issue_d': [issue_date],
            'earliest_cr_line': [earliest_credit_date],
            'dti': [dti],
            'delinq_2yrs': [delinq_2yrs],
            'inq_last_6mths': [inq_last_6mths],
            'mths_since_last_delinq': [mths_since_last_delinq],
            'mths_since_last_record': [mths_since_last_record],
            'mths_since_last_major_derog': [mths_since_last_major_derog],
            'open_acc': [open_acc],
            'pub_rec': [pub_rec],
            'revol_bal': [revol_bal],
            'revol_util': [revol_util],
            'total_acc': [total_acc],
            'tot_coll_amt': [tot_coll_amt],
            'tot_cur_bal': [tot_cur_bal],
            'total_rev_hi_lim': [total_rev_hi_lim],
            'home_ownership': [home_ownership],
            'purpose_group': [purpose_group],
            'addr_region': [addr_region],
            'ever_public_record': [ever_public_record],
            'ever_major_derog': [ever_major_derog],
            'ever_delinquent': [ever_delinquent],
            'has_credit_history': [has_credit_history],
            'emp_length_missing': [emp_length_missing]
        })
        
        # Step 1: Apply feature engineering
        input_engineered = create_features_from_input(input_data)
        
        # Step 2: Apply capping
        input_capped = apply_capping(input_engineered)
        
        # Step 3: Transform emp_length using mapping
        input_capped['emp_length'] = input_capped['emp_length'].map(mapping_emp_length)
        
        # Step 4: Transform term to numeric
        input_capped['term'] = input_capped['term'].str.extract('(\d+)').astype('Int64')

        # Step 5: Transform home_ownership using mapping
        input_capped['home_ownership'] = input_capped['home_ownership'].map(home_ownership_options)
        
        # Step 6: Predict
        with st.spinner("Analyzing loan application..."):
            probability = model.predict_proba(input_capped)[0, 1]
            prediction = 1 if probability >= threshold else 0
        
        # ====================================================================
        # DISPLAY RESULTS (SAMA SEPERTI SEBELUMNYA)
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Default Probability", f"{probability:.1%}")
        
        with col2:
            if prediction == 1:
                st.metric("Prediction", "⚠️ HIGH RISK", delta="Default Likely", delta_color="inverse")
            else:
                st.metric("Prediction", "✅ LOW RISK", delta="Default Unlikely")
        
        with col3:
            decision = "REJECT" if prediction == 1 else "APPROVE"
            decision_color = "🔴" if prediction == 1 else "🟢"
            st.metric("Decision", f"{decision_color} {decision}")
        
        # Credit History Summary
        st.subheader("📋 Credit History Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if ever_delinquent == 1:
                st.warning(f"⚠️ Delinquency: {mths_since_last_delinq} months ago")
            else:
                st.success("✅ Delinquency: NEVER (0 months)")
        
        with col2:
            if ever_public_record == 1:
                st.warning(f"⚠️ Public Record: {mths_since_last_record} months ago")
            else:
                st.success("✅ Public Record: NEVER (0 months)")
        
        with col3:
            if ever_major_derog == 1:
                st.error(f"❌ Major Derogatory: {mths_since_last_major_derog} months ago")
            else:
                st.success("✅ Major Derogatory: NEVER (0 months)")
        
        # Risk meter
        st.subheader("Risk Meter")
        
        if probability > 0.50:
            risk_color = "red"
            risk_text = "Very High Risk"
        elif probability > 0.30:
            risk_color = "orange"
            risk_text = "High Risk"
        elif probability > threshold:
            risk_color = "yellow"
            risk_text = "Medium Risk"
        else:
            risk_color = "green"
            risk_text = "Low Risk"
        
        st.progress(float(probability))
        st.markdown(f"<span style='color:{risk_color}; font-size:20px; font-weight:bold;'>Risk Level: {risk_text} ({probability:.1%})</span>", 
                    unsafe_allow_html=True)
        
        st.caption(f"Decision threshold: {threshold:.0%} (optimized for business impact)")
        
        st.markdown("---")
        st.subheader("💡 Recommendation")
        
        if prediction == 1:
            st.error("""
            **⚠️ RECOMMENDATION: REJECT THIS LOAN APPLICATION**
            
            The model predicts a high probability of default. Consider:
            - Requesting additional collateral
            - Offering a lower loan amount
            - Suggesting a shorter loan term
            - Requiring a co-signer
            """)
        else:
            st.success("""
            **✅ RECOMMENDATION: APPROVE THIS LOAN APPLICATION**
            
            The model predicts a low probability of default. The borrower appears creditworthy.
            """)
        
        with st.expander("ℹ️ About This Prediction"):
            st.info(f"""
            **How this prediction works:**
            
            - Model: XGBoost with threshold tuning
            - Decision threshold: {threshold:.0%}
            - Default probability: {probability:.1%}
            
            **Note:** This prediction is for educational purposes only.
            """)

# ============================================================================
# MODE 2: BATCH PREDICTION (CSV UPLOAD) - SAMA SEPERTI SEBELUMNYA
# ============================================================================

else:
    st.header("📁 Batch Prediction from CSV File")
    st.markdown("Upload a CSV file containing multiple loan applications for batch prediction.")
    
    # ========================================================================
    # SAMPLE DATA SECTION
    # ========================================================================
    
    st.subheader("📋 Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Download sample template:**")
        sample_template = create_sample_csv()
        
        csv_buffer = io.StringIO()
        sample_template.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_str,
            file_name="loan_prediction_template.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("**Or use sample dataset:**")
        st.info("Click below to load a sample dataset for testing")
        
        sample_data = None
        try:
            sample_data = pd.read_csv('Data/sample_test_data.csv', nrows=10)
            st.success("✅ Sample data available!")
        except:
            st.warning("Sample data not found. Using template as example.")
            sample_data = sample_template
        
        if sample_data is not None:
            st.download_button(
                label="📊 Download Sample Dataset (10 rows)",
                data=sample_data.to_csv(index=False),
                file_name="loan_prediction_sample.csv",
                mime="text/csv"
            )
    
    # ========================================================================
    # CSV UPLOAD SECTION
    # ========================================================================
    
    st.subheader("⬆️ Upload Your CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the required columns. Download the template above for reference."
    )
    
    required_columns = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length',
        'annual_inc', 'verification_status', 'issue_d', 'earliest_cr_line',
        'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq',
        'mths_since_last_record', 'mths_since_last_major_derog', 'open_acc',
        'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'tot_coll_amt',
        'tot_cur_bal', 'total_rev_hi_lim', 'home_ownership', 'purpose_group',
        'addr_region'
    ]
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            missing_cols = set(required_columns) - set(df_upload.columns)
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV contains all required columns. Download the template for reference.")
            else:
                st.success(f"✅ File uploaded successfully! {len(df_upload)} rows found.")
                
                with st.expander("📊 Preview Uploaded Data"):
                    st.dataframe(df_upload.head(10))
                    
                    col_info = pd.DataFrame({
                        'Column': df_upload.columns,
                        'Dtype': df_upload.dtypes.values,
                        'Non-Null': df_upload.count().values,
                        'Null': df_upload.isnull().sum().values
                    })
                    st.dataframe(col_info)
                
                auto_cols = ['ever_public_record', 'ever_major_derog', 'ever_delinquent', 
                            'has_credit_history', 'emp_length_missing']
                
                for col in auto_cols:
                    if col not in df_upload.columns:
                        if col == 'ever_delinquent':
                            df_upload['ever_delinquent'] = (df_upload['mths_since_last_delinq'] > 0).astype(int)
                        elif col == 'ever_public_record':
                            df_upload['ever_public_record'] = (df_upload['mths_since_last_record'] > 0).astype(int)
                        elif col == 'ever_major_derog':
                            df_upload['ever_major_derog'] = (df_upload['mths_since_last_major_derog'] > 0).astype(int)
                        elif col == 'has_credit_history':
                            df_upload['has_credit_history'] = (
                                (df_upload['tot_cur_bal'] > 0) | 
                                (df_upload['tot_coll_amt'] > 0) | 
                                (df_upload['total_rev_hi_lim'] > 0)
                            ).astype(int)
                        elif col == 'emp_length_missing':
                            df_upload['emp_length_missing'] = df_upload['emp_length'].isnull().astype(int)
                
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    with st.spinner(f"Processing {len(df_upload)} loan applications..."):
                        results_df = predict_batch(df_upload)
                    
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Applications", len(results_df))
                    with col2:
                        approve_count = (results_df['prediction'] == 0).sum()
                        st.metric("✅ Approved", approve_count, f"{approve_count/len(results_df)*100:.1f}%")
                    with col3:
                        reject_count = (results_df['prediction'] == 1).sum()
                        st.metric("❌ Rejected", reject_count, f"{reject_count/len(results_df)*100:.1f}%")
                    with col4:
                        avg_risk = results_df['default_probability'].mean()
                        st.metric("Avg Risk Score", f"{avg_risk:.1%}")
                    
                    st.subheader("Risk Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    
                    risk_colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Very High': 'red'}
                    
                    for risk in ['Low', 'Medium', 'High', 'Very High']:
                        if risk in risk_counts.index:
                            count = risk_counts[risk]
                            st.markdown(f"<span style='color:{risk_colors[risk]};'>●</span> {risk}: {count} ({count/len(results_df)*100:.1f}%)", 
                                      unsafe_allow_html=True)
                            st.progress(count/len(results_df))
                    
                    with st.expander("📋 View Detailed Results"):
                        display_cols = ['loan_amnt', 'term', 'annual_inc', 'grade', 
                                       'default_probability', 'decision', 'risk_level']
                        available_cols = [col for col in display_cols if col in results_df.columns]
                        st.dataframe(
                            results_df[available_cols].style.format({
                                'default_probability': '{:.1%}'
                            })
                        )
                    
                    st.subheader("📥 Download Results")
                    
                    csv_result = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download Predictions as CSV",
                        data=csv_result,
                        file_name=f"loan_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.success("✅ Batch prediction completed successfully!")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please make sure your CSV has the correct format. Download the template for reference.")
    
    else:
        st.info("👆 Please upload a CSV file to start batch prediction.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Loan Default Prediction App | Powered by XGBoost | Data: LendingClub 2007-2014</p>",
    unsafe_allow_html=True
)