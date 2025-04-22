Project Overview
The objective of this project was to analyze a dataset containing financial records of previous loan recipients and build a machine learning model capable of forecasting credit or loan risk for new applicants. Specifically, the aim was to develop a classifier that predicts whether a given applicant poses a high risk (1) or low risk (0), helping financial institutions—such as banks and private lenders—make better-informed lending decisions to minimize the likelihood of default. This model serves as a simplified prototype of the types of predictive systems that are commonly integrated into credit evaluation and risk management operations, offering the potential to flag risky applicants before loans are issued or go delinquent.

The dataset included key financial attributes like loan_size, interest_rate, debt_to_income, and derogatory_marks, which are commonly used to assess a borrower's financial reliability. The target variable selected for modeling was loan_status, which resulted in an imbalanced dataset: out of the total records, 18,765 entries were marked as low-risk (0), and only 619 were labeled as high-risk (1).

During exploratory analysis using the .describe() method, it was noted that the feature values varied significantly in scale. For instance, the borrower_income feature had a mean exceeding 49,000 with a standard deviation above 8,000, whereas derogatory_marks only ranged from 0 to 3. To account for this discrepancy, a normalization step using StandardScaler was introduced to scale the numerical features in X, while leaving the target variable y unchanged. This scaling process ensured that no feature disproportionately influenced the learning process due to larger numerical values.

Machine Learning Process
Initial Data Exploration using .head() and .describe() to understand variable types, ranges, and normalization requirements

Feature and Label Separation: X for input features, y for the output label (loan_status)

Feature Scaling using StandardScaler to bring all numeric features to a standard scale

Train-Test Split to divide the data for model training and evaluation, ensuring reproducibility with a fixed random_state

Model Training using the LogisticRegression algorithm on the preprocessed training data

Model Evaluation using metrics such as accuracy, precision, recall, and a confusion matrix to assess classification performance

This structured workflow helped establish a strong baseline classifier capable of identifying high- and low-risk loan applicants based on financial history.

Evaluation Metrics
After training the logistic regression model, its effectiveness was assessed using accuracy, precision, recall, and a confusion matrix. These metrics were chosen to reflect how well the model can differentiate between low-risk and high-risk applicants, and how dependable its predictions are in a real-world scenario.

Model: LogisticRegression
Accuracy: 99.36%

Precision: 84.07%

Recall: 98.39%

Confusion Matrix:

True negatives: 18,652
False positives: 113
False negatives: 10
True positives: 609

From the results, the model correctly identified 609 out of 619 true high-risk applicants, resulting in a recall of 98.39%, indicating a high capability for identifying default-prone individuals. It also accurately predicted 18,652 of 18,765 low-risk applicants, with only 113 false positives.

The model’s overall accuracy of 99.36% demonstrates strong general performance across both classes. Additionally, the precision score of 84.07% shows that the majority of individuals flagged as high-risk were indeed accurate predictions. While a small portion of low-risk applicants were incorrectly labeled, this is considered an acceptable compromise in financial contexts where minimizing default is paramount.

Final Thoughts and Recommendations
The logistic regression model achieved outstanding results, particularly in terms of recall, which reached 98.39%. This metric is especially critical in credit risk contexts where identifying high-risk borrowers is essential to avoid loan defaults. The high accuracy of 99.36% also reinforces the model’s reliability, while the precision of 84.07% shows that the high-risk predictions were mostly accurate.

In this domain, recall takes precedence over precision, since failing to identify a true high-risk applicant (a false negative) could have more severe financial implications than incorrectly labeling a low-risk one (a false positive). The model’s tendency toward caution is thus appropriate for the use case.

Recommendation:
This logistic regression model serves as a reliable starting point for loan risk classification and can be adopted as a foundational tool in credit decision-making systems. It demonstrates strong capability in handling class imbalance and is effective in identifying potentially risky clients. Further improvements could be explored through advanced techniques or ensemble models, but this implementation already provides a robust baseline for practical applications.

