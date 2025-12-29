# Credit Risk Prediction for a Lending Company  
**Final Project – Data Scientist Internship (ID/X Partners × Rakamin)**

---

## 1. Project Background

As part of the final assignment of the Data Scientist internship program at **ID/X Partners**, in collaboration with **Rakamin**, this project simulates a real-world business case from a **lending company**.  

The company operates in the consumer lending sector and faces increasing challenges in managing **credit risk**, particularly in distinguishing between borrowers who are likely to **repay their loans** and those who may **default**. Ineffective credit risk assessment can lead to high default rates, financial losses, and inefficient capital allocation.

To address this issue, the company seeks a **data-driven credit risk prediction model** that can support better lending decisions and risk management strategies.

---

## 2. Business Problem

Lending companies must evaluate loan applications accurately to balance **profitability** and **risk exposure**. However, traditional rule-based or manual assessment methods often fail to capture complex patterns in borrower behavior.

The main business problems are:

- High risk of **loan default** due to inaccurate borrower assessment  
- Difficulty in identifying **high-risk applicants** early in the loan lifecycle  
- Lack of data-driven insights to support **credit approval decisions**  
- Inefficient risk segmentation that may lead to either:
  - Approving risky borrowers, or  
  - Rejecting potentially profitable customers

---

## 3. Company Goals

The lending company aims to:

- Minimize **credit losses** caused by bad loans  
- Improve **loan approval accuracy**  
- Enhance **risk-based decision making**  
- Develop a scalable and explainable **credit risk scoring system**  
- Support business teams (credit, risk, and management) with actionable insights

---

## 4. Project Objective

The primary objective of this project is to:

> **Build a machine learning model that predicts credit risk by classifying loans as “good” or “bad” based on historical loan data.**

Specifically, this project aims to:

- Perform **end-to-end data science workflow**, including:
  - Data understanding
  - Data cleaning and preprocessing
  - Feature engineering
  - Outlier handling
  - Model training and evaluation
- Develop and compare multiple **classification models**
- Identify the most suitable model based on business-relevant metrics
- Prepare **clear and communicative visualizations** to present insights and results to stakeholders

---

## 5. Dataset Overview

The dataset used in this project is:

- **File name:** `loan_data_2007_2014.csv`
- **Source:** Internal dataset provided as part of the internship program
- **Description:** Historical loan records from a lending company between 2007 and 2014

The dataset contains information related to:

- Borrower demographics and employment
- Loan characteristics (amount, term, interest rate)
- Credit history and behavior
- Loan performance and status

Each record represents a **loan application**, with the target variable indicating the loan outcome (e.g., fully paid, charged off, default).

---

## 6. Target Variable

The target variable is derived from the `loan_status` column and is transformed into a **binary classification label**:

- **0 → Good Loan** (e.g., fully paid)
- **1 → Bad Loan** (e.g., charged off, default)

This binary formulation aligns with the business goal of **credit risk prediction**, where identifying bad loans is critical for minimizing financial losses.

---

## 7. Scope & Methodology

This project follows a standard **Data Science methodology**, including:

1. Data Understanding & Exploratory Data Analysis (EDA)  
2. Data Cleaning & Missing Value Handling  
3. Feature Engineering & Encoding  
4. Outlier Handling (business-driven approach)  
5. Handling Imbalanced Data  
6. Model Training & Evaluation  
7. Model Comparison & Selection  
8. Insight Generation & Visualization

The solution is implemented using **Python**, leveraging common data science and machine learning libraries.

---

## 8. Expected Impact

By implementing this solution, the lending company is expected to:

- Reduce default-related financial losses  
- Improve decision consistency and transparency  
- Enable data-driven credit risk management  
- Gain deeper insights into borrower risk profiles  

---

## 9. Deliverables

The final deliverables of this project include:

- A trained and evaluated **credit risk prediction model**
- Supporting **visualizations and performance metrics**
- A clear, structured **Jupyter Notebook** documenting the end-to-end process
- Business-oriented insights that can be communicated to non-technical stakeholders

---

This project demonstrates the application of data science techniques to solve a real-world **credit risk problem**, bridging technical modeling with business objectives.
