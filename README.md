# **PowerPulse: Household Energy Usage Forecast**

## **Project Overview**
**PowerPulse** is a machine learning project aimed at predicting household energy consumption using historical data. Accurate energy consumption predictions help in better resource planning, cost reduction, and promoting energy efficiency for both households and energy providers. This project applies data preprocessing, feature engineering, and multiple regression models to build an optimized energy usage forecast system.  

By the end of the project, actionable insights and predictive models are provided to help households monitor their energy usage patterns and assist energy providers with demand forecasting and anomaly detection.

---

## **Domain**
**Energy and Utilities**

---

## **Dataset Information**
**Dataset:** Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available.  

- **Multivariate, Time-Series Data**
- **Features**

   ![image](https://github.com/user-attachments/assets/4b69de45-e92f-45b6-9654-5eb061c12b48)
    


---

## **Business Use Cases**
- **Energy Management for Households:** Monitor energy usage, reduce bills, and promote energy-efficient habits.  
- **Demand Forecasting for Energy Providers:** Predict energy demand for better load management and pricing strategies.  
- **Anomaly Detection:** Identify irregular energy usage patterns indicating faults or unauthorized access.  
- **Smart Grid Integration:** Enable predictive analytics for real-time energy optimization.  
- **Environmental Impact:** Reduce carbon footprints and promote conservation initiatives.  

---

## **Technical Stack & Tools**
- **Languages:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
- **Models:** Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, Neural Network (MLP)  

---

## **Approach**
### 1. **Data Understanding and Exploration**
- Load dataset and examine its structure  
- Identify missing values, duplicates, and data types  

### 2. **Data Preprocessing**
- Handle missing values using appropriate strategies  
- Handle placeholder values  
- Convert data types where necessary  
- Detect and handle outliers using IQR and Z-score methods and visualize outliers using box plot 

### 3. **Feature Engineering**
- Create new features such as `Hour`, `DayOfWeek`, `Month`, `WeekOfYear`, `IsWeekend`, `IsPeakHour`, `Daily_Consumption`, `Unmetered_Energy`, `Short_Term_Avg_Power`, `Hourly_Avg_Power`, `Daily_Avg_Power`, `Power_Deviation_10min`, `Power_Anomaly_Flag`, `Season`, `TimeOfDay`, 'carbon_emission'.

### 4. **Exploratory Data Analysis (EDA)**
- Analyze trends, seasonal patterns, outliers, and skewness among features  
- Visualize correlations among variables  

### 5. **Model Selection and Training**
- Train multiple regression models on 0.1% of sample extracted from the full dataset  
- Split the dataset into training and testing sets  
- Train regression models such as:  
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
  - ElasticNet  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost  
  - Neural Network (MLP Regressor)  

### 6. **Evaluation**
- Evaluate and compare models using **MAE, MSE, RMSE, R2 Score**

---

## **Technologies Used**

| Technology            | Purpose                                                                                      |
|-----------------------|----------------------------------------------------------------------------------------------|
| **Python**            | Core programming language for data processing & analysis. [Docs](https://docs.python.org/)  |
| **Pandas**            | Data manipulation & preparation. [Docs](https://pandas.pydata.org/docs/)                    |
| **NumPy**             | Numerical computing and array manipulation. [Docs](https://numpy.org/doc/)                  |
| **Matplotlib/Seaborn**| Data visualization for exploratory data analysis (EDA). [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) |
| **Scikit-learn**      | Machine learning model development, evaluation, and preprocessing. [Docs](https://scikit-learn.org/stable/) |
| **XGBoost**           | Optimized gradient boosting library for high-performance models. [Docs](https://xgboost.readthedocs.io/) |
| **SciPy**             | Advanced statistical functions and analysis. [Docs](https://scipy.org/)                      |

---

## **Installation & Setup**
To run this project, install the required dependencies using the following command:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy ucimlrepo
```


---


## **Models Performance**


![image](https://github.com/user-attachments/assets/558b40b0-6c2b-4b55-aa72-12011801b21f)



---

## **Best Model Analysis**
1. **Neural Network (MLP)**  
   - **MAE:** 2.204464e-03  
   - **MSE:** 1.226218e-05  
   - **RMSE:** 3.501740e-03  
   - **R2 Score:** 0.999989  
   **Why it stands out:** Neural Network (MLP) achieves extremely low error metrics and almost perfect R2 score, indicating a highly accurate prediction model.

2. **Random Forest**  
   - **MAE:** 3.437224e-03  
   - **MSE:** 3.296846e-04  
   - **RMSE:** 1.815722e-02 
   - **R² Score:** 0.999710  
   **Why it stands out:** Random Forest offers near-perfect prediction with a very high R² score and low error values, making it one of the top-performing models.

3. **XGBoost**  
   - **MAE:** 1.880181e-02  
   - **MSE:** 2.138086e-03  
   - **RMSE:** 4.623945e-02  
   - **R² Score:** 0.998118
   **Why it stands out:** XGBoost balances performance with speed, achieving impressive results with relatively low error metrics.




