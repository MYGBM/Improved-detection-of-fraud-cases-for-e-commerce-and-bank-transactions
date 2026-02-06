# Fraud Detection Analysis and Feature Engineering Report

## 1. Business Objective
The primary objective of this project is to improve the detection of fraud cases for e-commerce and bank transactions. By analyzing transaction data and user behavior, I aim to build a machine learning model that can accurately identify fraudulent activities. This will enable the business to:
- Minimize financial losses due to chargebacks and fraudulent purchases.
- Enhance security measures by flagging high-risk transactions in real-time.
- Improve customer trust by reducing false positives and ensuring legitimate transactions are processed smoothly.

## 2. Accomplishments

### Exploratory Data Analysis (EDA)
I conducted a comprehensive analysis to understand the data distribution and identify key patterns associated with fraud. This involved Univariate, Bivariate, and Multivariate analysis.

*   **Univariate Analysis**: 
    *   Analyzed the distribution of individual features like `purchase_value`, `age`, `source`, `browser`, and `sex`.
    *   *Observation*: The dataset is highly imbalanced, with legitimate transactions vastly outnumbering fraudulent ones. `purchase_value` is highly right-skewed.
    *   ![Placeholder: Class Balance Pie Chart]
    *   ![Placeholder: Purchase Value Distribution Plot]

*   **Bivariate Analysis (Transaction Count vs. Fraud Rate)**:
    *   I analyzed the relationship between features and the target variable (`class`) by plotting transaction counts (volume) alongside fraud rates (risk). This helped identify high-risk segments that might be overlooked due to low volume.
    *   **Categorical Features**:
        *   *Source*: "Direct" traffic showed the highest fraud rate despite lower volume compared to SEO or Ads.
        *   *Browser*: While Chrome has the highest volume, niche browsers like Opera and Firefox showed disproportionately higher fraud rates.
        *   *Country*: Identified "Red Flag" countries (e.g., Mexico, Sweden) which have low transaction volumes but very high fraud rates (~11-13%). Conversely, the US has high volume but a moderate fraud rate.
    *   **Continuous Features**:
        *   *Purchase Value*: Detected a spike in fraud rate for transactions in the $86-$92 range.
        *   *Age*: Observed that the 41-64 age group has a higher susceptibility to fraud compared to other demographics.
    *   ![Placeholder: Fraud Rate by Browser Plot]
    *   ![Placeholder: Fraud Rate by Country Plot]

*   **Geolocation Analysis**: 
    *   Mapped IP addresses to countries to identify high-risk regions using the `IpAddress_to_Country` dataset.
    *   Converted IP addresses from float to integer format to enable accurate range matching.

*   **Temporal Analysis**:
    *   Analyzed fraud rates by hour of the day and day of the week.
    *   *Observation*: Fraudulent activities peaked during specific hours (early morning/late night) and specific days, suggesting automated attacks or specific fraudster working hours.

### Feature Engineering
I transformed the raw data into a robust format suitable for machine learning models, creating new features to capture fraud patterns.

*   **Feature Creation**:
    *   **`purchase_time_hour`**: Extracted the hour from the purchase timestamp to capture time-of-day patterns (e.g., late-night fraud).
    *   **`purchase_time_day_of_week`**: Extracted the day to identify weekly trends.
    *   **`purchase_time_since_signup_seconds`**: Calculated the time difference between signup and purchase. This proved to be a critical feature, revealing a massive spike in fraud for transactions occurring immediately after signup (indicative of bots).
    *   **`purchase_time_since_signup_hours`**: A converted version of the above for interpretability.
    *   *Note*: `user_transaction_count` was created but dropped as it provided no variance (all users had 1 transaction).

*   **Normalization & Scaling**:
    *   **Log Transformation**: Applied `np.log1p` to `purchase_value` and `purchase_time_since_signup_seconds` to handle their extreme skewness and long tails, making the distributions more Gaussian-like for the models.
    *   **StandardScaler**: Applied to `age` and the log-transformed features to ensure all numerical inputs have a mean of 0 and variance of 1, preventing features with larger magnitudes from dominating the model.

*   **Categorical Encoding**:
    *   **One-Hot Encoding**: Applied to low-cardinality features (`source`, `browser`, `sex`) to convert them into binary vectors.
    *   **Frequency Encoding**: Applied to the `country` feature. Since `country` has high cardinality (many unique values), One-Hot encoding would create too many sparse columns. Frequency encoding maps each country to its probability of occurrence, preserving information without exploding dimensionality.

*   **Handling Class Imbalance**:
    *   The dataset was heavily imbalanced (Fraud cases << Legitimate cases).
    *   I applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the **training data only**. This creates synthetic fraud examples to balance the classes, ensuring the model learns to detect fraud effectively without overfitting to the majority class.
    *   ![Placeholder: Class Distribution Before vs After SMOTE]

## 3. Challenges Faced

During the analysis and engineering phase, I encountered and resolved several challenges:

1.  **IP Address Mapping**: 
    *   *Challenge*: Mapping IP addresses to countries involved range matching, which is computationally expensive with standard merges. The IP addresses were also in float format.
    *   *Solution*: I converted IPs to integers and used `pandas.merge_asof` for efficient range-based merging.

2.  **Highly Skewed Data**:
    *   *Challenge*: Features like `purchase_value` and `time_since_signup` had extreme outliers and long tails, which can distort linear models and neural networks.
    *   *Solution*: I applied Log transformation (`np.log1p`) to compress the range and make the distribution more Gaussian-like.

3.  **Class Imbalance**:
    *   *Challenge*: The rarity of fraud cases meant a standard model would likely predict "No Fraud" for everything and achieve high accuracy but 0 recall.
    *   *Solution*: I implemented SMOTE on the training set only, preserving the test set's integrity for fair evaluation.

## 4. Model Building and Evaluation

I implemented and compared two models: **Logistic Regression** (Baseline) and **Random Forest** (Advanced). I used **Stratified K-Fold Cross-Validation** to ensure stability and **AUC-PR** as the primary metric due to the class imbalance.

### Results: E-Commerce Fraud Data (`Fraud_Data.csv`)
*   **Logistic Regression**: High Recall (0.70) but very low Precision (0.17). It generates too many false alarms.
*   **Random Forest**: Lower Recall (0.52) but near-perfect Precision (0.97).
*   **Winner**: **Random Forest**.
    *   *Reason*: A Precision of 0.17 is unacceptable for user experience (blocking 5 legitimate users for 1 fraudster). Random Forest is much safer for the business, and its Recall can be improved by adjusting the probability threshold.

### Results: Credit Card Data (`creditcard.csv`)
*   **Logistic Regression**: Excellent Recall (0.92) but abysmal Precision (0.06).
*   **Random Forest**: High Recall (0.84) and High Precision (0.86). AUC-PR of **0.88**.
*   **Winner**: **Random Forest**.
    *   *Reason*: It achieved the "Sweet Spot" of high security and low friction. This is a production-ready model.

## 5. Next Steps

Following the successful modeling phase, I will proceed to the final phase:

*   **Task 3: Model Explainability and Deployment**
    *   Use SHAP or LIME to explain model predictions (e.g., "Why was this transaction flagged?").
    *   Build a dashboard (using Flask or Streamlit) to visualize fraud insights and serve model predictions.
    *   Deploy the model as a REST API for real-time inference.
