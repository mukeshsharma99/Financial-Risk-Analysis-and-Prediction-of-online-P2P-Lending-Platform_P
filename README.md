## Financial Risk Analysis and prediction of Online P2P 
Lending Platform
Peer-To-Peer (P2P) lending platforms are online services provided by financial 
institutions as an intermediary to initiate loans for private individuals. Loans for borrowers 
are funded by multiple investors, bound with agreed-upon terms and conditions, with 
profits generated from the interest made on the loans as the borrowers are given a certain 
duration to pay back the loan and interest. P2P lending has gained popularity for personal, 
small business start-ups allowing individuals and businesses to loan money directly from 
investors or lenders without going through the strict requirements and criteria of traditional 
banks and financial institutions.This study presents a supervised machine learning model 
that predicts the probability of default by considering more information related to the 
clients rather than just evaluating their credit score. In this project , we have done the 
predictive analysis of the parameters like(** Loan Status ** , ** EMI**,**PROI**,**ELA**) 
in an Online P2P lending market related to determinants of performance predictability by 
conceptualizing its financial benefits to make further predictions whether one can  
successfully predict which loans will default. There is thus a significant financial incentive 
to accurately predicting which of the loans would eventually default or not.  

## Initial Data Overview
The Prosper Loan Dataset contains 113,937 loans with 81 variables on each loan, 
including loan amount, borrower rate (or interest rate), current loan status, borrower 
income, and many others. Some columns are numeric and some are categorical 
variables.Categorical contains both (ordinal and nominal) and datetime variables.
Now we will work with features such as (Borrower Rate, Borrower APR, Prosper Score, 
Credit Score, Original Loan Amount, Monthly Payment, Borrower Occupation, Borrower 
State and others if needed).
There are some important features to look at including:
● BorrowerAPR: The Borrower's Annual Percentage Rate (APR) for the loan.
● ProsperScore: A custom risk score built using historical Prosper data. The score 
ranges from 1-10, with 10 being the best or lowest risk score. Applicable for 
loans originated after July 2009.
● LoanOriginationDate: The date the loan was originated.
● LenderYield: The Lender yields on the loan. Lender yield is equal to the interest 
rate on the loan less the servicing fee.
Other Features that will help us to support investigation in the Prosper Loan Data 
features are Loan Status and Employment Status will have a strong impact on loan and the features 
we are trying to explore further are also the Monthly Income will play a role here and the 
Term may have an effect. " Borrower APR is normally distributed with the peak 
between 15 and 20 percent in addition we have some increase in the 35 percent" 
"Prosper Scores are almost normally distributed and values 4, 6 and 8 are the most 
common." "Lender Yield is normally distributed with most of the values between 0.1 and 
0.2 and we notice an increase at 0.3 

## Data Exploration
We began by loading the Prosper Loan dataset and performing an initial inspection. 
This included checking the shape of the data, listing the columns, and obtaining basic 
information about the dataset such as the number of non-null entries in each column 
and their respective data types. This initial exploration revealed that the dataset 
contains both numerical and categorical columns, some of which have missing values 
that need to be addressed.


## Handling Missing Values
The first step in data preprocessing was to identify columns with missing values. We 
calculated the number of null values in each column and filtered out those that 
contained missing data. This provided a clear understanding of which columns required 
attention.
## Categorical Variables
We focused on categorical variables by isolating them and inspecting the extent of 
missing data within these columns. For categorical variables, missing values were filled 
with the mode (the most frequent value) of each respective column. This approach was
chosen because it is a simple and effective way to handle missing categorical data 
without introducing bias.
We also created a new binary variable, LoanStatus, derived from the ClosedDate 
column. If ClosedDate was null, it indicated that the loan was still active (coded as 1). 
Additionally, if the LoanCurrentDaysDelinquent was greater than 180 days, it indicated a 
problematic loan, and LoanStatus was set to 1.
## Dropping Unnecessary Columns
To reduce complexity and focus on the most relevant information, we dropped several 
columns deemed unnecessary for further analysis. These included ListingKey, 
LoanKey, GroupKey, MemberKey, LoanOriginationQuarter, and ListingNumber.
Filling Missing Values in Categorical Columns
For columns such as BorrowerState, Occupation, EmploymentStatus, and 
FirstRecordedCreditLine, missing values were filled with the mode of each column. This 
ensured that no categorical data was left missing.
## Prosper Rating and Credit Grade
We filled missing values in the ProsperRating (Alpha) column with values from the 
CreditGrade column. Following this, the CreditGrade column was dropped from the 
dataset as it was no longer needed.
## Continuous Variables
We also identified and addressed missing values in continuous (numerical) variables. 
Columns with missing values were filled with either the mean or median of the 
respective columns, depending on the nature of the data.
## Exploratory Data Analysis
### Univariate Analysis
Univariate analysis focuses on examining the distribution of individual variables in the 
dataset. This analysis helps us understand the range, central tendency, and variability 
of the data. Various graphs and visualizations are plotted to analyze the results 
effectively.
## Distribution of Key Features
We plotted histograms and kernel density estimates (KDE) for several numerical 
   features to visualize their distributions:
● Loan Term: The duration of the loan in months.
● Borrower APR and Rate: The annual percentage rate and interest rate charged 
  to the borrower.
● Lender Yield: The yield earned by the lender.
● Estimated Effective Yield, Loss, and Return: Key financial metrics estimating
   yields and losses.
● Prosper Rating (numeric): A numerical rating assigned to each loan.
● Employment Status Duration: The length of time the borrower has been 
employed.
● Credit Scores and Credit Lines: Various metrics related to the borrower's credit 
history.

These visualizations provided insights into the distribution and skewness of these 
features.

# Categorical Variables
For categorical variables, bar and pie plots were used to analyze their frequency 
distributions:
● Prosper Rating (Alpha): A proprietary rating system used to evaluate 
applicants.
● Employment Status: The employment status of the borrower, including 
categories like Employed, Full-time, Self-employed, etc.
● Income Range: The income range of the borrower.
● Loan Status: The current status of the loan, such as Canceled, Chargedoff, 
Completed, etc.

# Target Variable: Loan Status
The target variable, LoanStatus, initially contained 12 different categorical values. To 
simplify analysis and perform binary classification, we converted this column into a 
binary variable using the ClosedDate and LoanCurrentDaysDelinquent columns:
● ClosedDate: If null, the loan is considered active.
● LoanCurrentDaysDelinquent: If greater than 180 days, the loan is considered 
problematic.
This conversion resulted in a binary column where 0 indicates incomplete loans, and 1 
indicates completed loans.

# Bivariate Analysis
Bivariate analysis examines the relationship between two variables. We used scatter 
plots to visualize these relationships:
● Borrower Rate vs. Prosper Score: Explores how the interest rate charged to 
the borrower relates to the Prosper Score.
● Borrower Rate vs. Loan Term: Examines the relationship between the loan 
term and the borrower's interest rate.
● Estimated Effective Yield vs. Estimated Return: Analyzes the relationship 
between estimated effective yield and estimated return, with the estimated loss 
represented by color.
● Total Prosper Payments Billed vs. On Time Prosper Payments: Investigates 
how the number of total payments relates to on-time payments.
● Monthly Loan Payment vs. Debt to Income Ratio: Studies the relationship 
between monthly loan payments and the borrower's debt-to-income ratio.


**Descriptive Statistics:**

Using **describe()** we could get the following result for the numerical features

||Open|Close|High|Low|Volume|Adj Close|
| :-- |:---------------:| -----:|-------:|:---------------:| -----:|------:|
|count|1989.000000|1989.000000|1989.000000|1989.000000|1.989000e+03|1989.000000|
|mean|13459.116049|13463.032255|13541.303173|13372.931728|1.628110e+08|13463.032255|
|std|3143.281634|3144.006996|3136.271725|3150.420934|9.392343e+07|3144.006996|
|min|6547.009766|6547.049805|6709.609863|6469.950195|8.410000e+06|6547.049805|
|25%|10907.339840|10913.379880|11000.980470|10824.759770|1.000000e+08|10913.379880|
|50%|13022.049810|13025.580080|13088.110350|12953.129880|1.351700e+08|13025.580080|
|75%|16477.699220|16478.410160|16550.070310|16392.769530|1.926000e+08|16478.410160|
|max|18315.060550|18312.390630|18351.359380|18272.560550|6.749200e+08|18312.390630|

**Correlation Plot of Numerical Variables:**

All the continuous variables are positively correlated with each other with correlation coefficient of 1 except **Volume** which has negative correlation of around 0.7 with all other variables



## Feature Engineering:
<ul>
<li><a href="#wrangling">Handling outliers</a></li>
<li><a href="#wrangling">Feature Selection</a></li>
<li><a href="#eda">Categorical Features Encoding</a></li>
<li><a href="#conclusions">Feature scaling</a></li>
<li><a href="#conclusions">Dimensionality Reduction using PCA</a></li>
</ul>

## **Feature Extraction and Dimensionality-reduction using (PCA)**

Principal component analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information.

The idea of PCA is simple — reduce the number of variables of a data set, while preserving as much information as possible.

PCA steps:
- Feature Scaling.
- Covariance Matrix computation.
- Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components.
- Feature Vector.
- Recast the Data Along the Principal Components Axes.

# Percentage of information we have after apllying 2-d PCA

77.82576677891772


## Model Building Classification 

Logistic Regression-----------------

Model performance for training set

- Accuracy: 0.9953
- F1 score: 0.9953
- Precision: 0.9808
- Recall: 0.9994
- Roc Auc Score: 0.9967

Model performance for testing set

- Accuracy: 0.9970
- F1 score: 0.9970
- Precision: 0.9878
- Recall: 0.9997
- Roc Auc Score: 0.9980


GaussianNB------------------------

Model performance for training set

- Accuracy: 0.8596
- F1 score: 0.8631
- Precision: 0.6731
- Recall: 0.7818
- Roc Auc Score: 0.8327
- 
Model performance for testing set

- Accuracy: 0.8666
- F1 score: 0.8696
- Precision: 0.6908
- Recall: 0.7904
- Roc Auc Score: 0.8403



Support Vector Machine-------------

Model performance for training set

- Accuracy: 0.9921
- F1 score: 0.9922
- Precision: 0.9680
- Recall: 0.9994
- Roc Auc Score: 0.9946
  
Model performance for testing set

- Accuracy: 0.9934
- F1 score: 0.9934
- Precision: 0.9732
- Recall: 0.9997
- Roc Auc Score: 0.9956


AdaBoost-----------------------

Model performance for training set

- Accuracy: 0.9842
- F1 score: 0.9843
- Precision: 0.9450
- Recall: 0.9902
- Roc Auc Score: 0.9863
  
Model performance for testing set

- Accuracy: 0.9838
- F1 score: 0.9839
- Precision: 0.9464
- Recall: 0.9874
- Roc Auc Score: 0.9850


### **Confusion Matrix and ROC Curve**
![Capture](https://github.com/user-attachments/assets/be3f0f99-a031-40a5-b2d9-2eb5b15a4331)
![Capture1](https://github.com/user-attachments/assets/fd5e09e0-ad1e-46b2-8261-962c33510447)
![Capture2](https://github.com/user-attachments/assets/3fca5665-755b-400b-946d-0b093af11fd6)
![Capture3](https://github.com/user-attachments/assets/60a5dba7-cb6c-43cc-816b-861cf299e374)

# Multi regression 

## Creating New Target Variables
From the dataset, new target variables are created as continuous output variables.
LoanTenure = (𝑀𝑎𝑡𝑢𝑟𝑖𝑡𝑦𝐷𝑎𝑡_𝑂𝑟𝑖𝑔𝑖𝑛𝑎𝑙𝑦𝑒𝑎𝑟 − 𝐿𝑜𝑎𝑛𝐷𝑎𝑡𝑒𝑦𝑒𝑎𝑟) 𝑥 12 − 
(𝑀𝑎𝑡𝑢𝑟𝑖𝑡𝑦𝐷𝑎𝑡𝑒_𝑂𝑟𝑖𝑔𝑖𝑛𝑎𝑙𝑚𝑜𝑛𝑡ℎ − 𝐿𝑜𝑎𝑛𝐷𝑎𝑡𝑒𝑚𝑜𝑛𝑡ℎ)
𝑀𝑎𝑡𝑢𝑟𝑖𝑡𝑦𝐷𝑎𝑡_𝑂𝑟𝑖𝑔𝑖𝑛𝑎𝑙𝑦𝑒𝑎𝑟 and 𝑀𝑎𝑡𝑢𝑟𝑖𝑡𝑦𝐷𝑎𝑡𝑒_𝑂𝑟𝑖𝑔𝑖𝑛𝑎𝑙𝑚𝑜𝑛𝑡ℎ are taken from 
ClosedDate column.
𝐿𝑜𝑎𝑛𝐷𝑎𝑡𝑒𝑦𝑒𝑎𝑟 and 𝐿𝑜𝑎𝑛𝐷𝑎𝑡𝑒𝑚𝑜𝑛𝑡ℎ are taken from LoanOriginationDate.
1.Equated Monthly Installments (EMI):
For each row in the dataset:
Calculate result_1 = P * r * 〖(1+r)〗^n
Calculate result_2 = 〖(1+r)〗^n – 1
Calculate EMI = result_1 / result_2
Tenure ---> Loan Tenure
Principle repayment ---> LP_CustomerPrinciplePayments
Interest ---> BorrowerRate
2.Eligible Loan Amount (ELA):
Calculation Procedure: For each row in the dataset:
Calculate: Total Payment Due = (A + (A*r)) * n
Calculate: Max allowable amount = I * 12 * 30%
If ( Total Payment Due <= Max allowable amount)
Then ELA = AppliedAmount
Else ELA = Max allowable amount
A: “AppliedAmount” --- LoanOriginalAmount
R: “Interest” --- BorrowerRate
N: “Loan Tenure” --- Loan Tenure
I: “IncomeTotal” ---StatedMonthlyIncome
3.PreferredROI:
ROI= Interest amount / Total Amount
Interest Amount= loanOriginalAmount * Borrower Rate
Total amount = Interest amount * LoanOriginalAmount

## Multiregression model

### Model: *Linear Regression*
Model performance for training set
- Mean Squared Error: 57398576.5676
- R² Score: 0.8501

Model performance for testing set
- Mean Squared Error: 66406843.0962
- R² Score: 0.8485


### Model: *Ridge Regression*
Model performance for training set
- Mean Squared Error: 57567646.9807
- R² Score: 0.8475

Model performance for testing set
- Mean Squared Error: 66717100.6790
- R² Score: 0.8455


### Residual Analysis:
- Max residual: ELA     127138.101941
- EMI       2718.642755
- PROI         0.159152
- dtype: float64
- Min residual: ELA    -420853.018272
- EMI      -1489.071089
- PROI        -0.180871

##### Graph 
![Capturerrr](https://github.com/user-attachments/assets/8b282005-be5b-4140-a19b-8a89e8c07597)

###  Hyperparameter Tuning
Best parameters found:  {'estimator__alpha': 0.1}

### Model Validation
- Cross-validation R² scores:  [0.85155888 0.85082248 0.84980226 0.84801792 0.8467248 ]
- Mean CV R² score:  0.8493852658665786



## Deployment
you can access our app by following this link [stock-price-application-streamlit](https://stock-price-2.herokuapp.com/) or by click [stock-price-application-flask](https://stock-price-flask.herokuapp.com/)
### Streamlit
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We write a python code for our app using Streamlit; the app asks the user to enter the following data (**news data**, **Open**, **Close**).
- The output of our app will be 0 or 1 ; 0 indicates that stock price will decrease while 1 means increasing of stock price.
- The app runs on local host.
- To deploy it on the internt we have to deploy it to Heroku.

### Heroku
We deploy our Streamlit app to [ Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others. 
We prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file and setup.sh.
- setup.sh: contains setup information.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (stock_price_App_V1.py)  successfully 
- stock_price_App_V1.py: contains the python code of a Streamlit web app.
- stock_price_xg.pkl : contains our XGBClassifier model that built by modeling part.
- X_train2.npy: contains the train data of modeling part that will be used to apply PCA trnsformation to the input data of the app.

### Flask 
We also create our app   by using flask , then deployed it to Heroku . The files of this part are located into (Flask_deployment) folder. You can access the app by following this link : [stock-price-application-flask](https://stock-price-flask.herokuapp.com/)


