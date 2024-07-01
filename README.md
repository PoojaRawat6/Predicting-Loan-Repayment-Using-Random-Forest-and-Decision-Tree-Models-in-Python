
# Predicting Loan Repayment Using Random Forest and Decision Tree Models in Python

This project explores publicly available data from [LendingClub.com](https://www.lendingclub.com/), aiming to predict the likelihood of borrowers repaying their loans. The focus is on data from 2007-2010, with a dataset cleaned of NA values.


## Dataset Description

The dataset includes several key columns:

- credit.policy: - Indicates if the borrower meets LendingClub's credit criteria.
- purpose: Describes the purpose of the loan.
- int.rate: Interest rate of the loan.
- installment: Monthly installments owed by the borrower.
- log.annual.inc: Natural log of the borrower's annual income.
- dti: Debt-to-income ratio.
- fico: Borrower's FICO credit score.
- days.with.cr.line: Days borrower has had a credit line.
- revol.bal: Revolving balance.
- revol.util: Revolving line utilization rate.
- inq.last.6mths: Inquiries by creditors in the last 6 months.
- delinq.2yrs: Times borrower was 30+ days past due in the last 2 years.
- pub.rec: Number of derogatory public records. 


# Exploratory Data Analysis (EDA)
### Data visualization: Plotting Histograms



```bash
import pandas as pd
import matplotlib.pyplot as plt

# Plotting histograms using pandas built-in .hist() method
loans.hist(column='fico', by='credit.policy', bins=30, figsize=(10, 6), alpha=0.5)
plt.tight_layout()
plt.show()


```
    
### Data Transformation 

The 'purpose' column is categorical. This requires transformation into dummy variables for compatibility with scikit-learn:
```bash
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True, dtype=float)


```

# Model Training and Evaluation

## Decision Tree Model
Classification Report:
```bash
Precision    Recall  F1-Score   Support
0       0.85      0.82      0.84      2431
1       0.19      0.23      0.21       443
Accuracy: 0.73


```
## Random Forest Model

Classification Report:

``` bash
 Precision    Recall  F1-Score   Support
0       0.85      1.00      0.92      2431
1       0.56      0.02      0.04       443
Accuracy: 0.85

```


# Conclusion

The random forest model outperformed the decision tree in terms of accuracy and F1-score, indicating its superior performance in predicting loan repayment..
