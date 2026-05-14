# Yes Bank Stock Price Prediction

![Yes Bank](https://img.shields.io/badge/Industry-Banking-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)

## 📌 Project Overview
Yes Bank is a well-known bank in the Indian financial domain. Since 2018, it has been in the news because of the fraud case involving former CEO Rana Kapoor. This project investigates how this scandal impacted the stock prices of the company and whether time series models or other predictive models can do justice to such volatile situations. 

The main objective of this project is to build an automated, end-to-end Machine Learning pipeline to predict the **monthly closing stock price** of Yes Bank based on its historical data.

## 📊 Dataset Description
The dataset contains monthly stock prices of Yes Bank since its inception and includes the following features:
*   **Date**: Month and year of the record.
*   **Open**: Opening stock price for the month.
*   **High**: Highest stock price during the month.
*   **Low**: Lowest stock price during the month.
*   **Close**: Closing stock price for the month (**Target Variable**).

## 🚀 Project Workflow

1.  **Exploratory Data Analysis (EDA)**: Extensive visual and statistical exploration involving 15 insightful charts. Analyzed absolute and percentage volatility, and mapped out the clear structural regime shift caused by the 2018 crisis.
2.  **Hypothesis Testing**: Formulated and tested statistical hypotheses, such as verifying the significant drop in the mean closing price post-January 2018 using a two-sample Independent t-test.
3.  **Data Preprocessing**: Chronological Train-Test Split (80-20) preserving time-series continuity (no shuffling). Features were standardized using `StandardScaler`.
4.  **Machine Learning Modeling**:
    *   **Linear Regression**: Used as a baseline model due to high linear correlation amongst features.
    *   **Ridge Regression**: Implemented with GridSearchCV to prevent overfitting.
    *   **Random Forest Regressor**: Selected as the final robust ensemble model to handle the inherent volatility, tuned via GridSearchCV.
5.  **Deployment**: The best model weights are serialized into a pickle file (`best_model_rf.pkl`) for production deployment.

## 📂 Repository Structure

| File / Folder | Description |
| --- | --- |
| `data_YesBank_StockPrices.csv` | The raw historical stock price dataset. |
| `Sample_ML_Submission_Template.ipynb` | The empty template provided for the Capstone project. |
| `Executed_Submission.ipynb` | **The Final Report**: Fully executed notebook containing all charts, code outputs, statistical justifications, and answers to project questions. |
| `best_model_rf.pkl` | The serialized Random Forest regression model, ready for deployment. |
| `app.py` | Streamlit application code for the interactive prediction web interface. |
| `requirements.txt` | Python dependencies required for deploying the Streamlit app. |

## ⚙️ How to Use

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   pip install matplotlib seaborn nbformat nbconvert
   ```
2. To view the final results, analysis, and insights, simply open `Executed_Submission.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. To test the interactive prediction web app locally, run:
   ```bash
   streamlit run app.py
   ```
   *The app can also be easily hosted on the Streamlit Community Cloud using the provided `requirements.txt` file.*

## 📈 Key Conclusions
*   The 2018 Rana Kapoor fraud scandal introduced a massive, permanent structural break in Yes Bank's stock price, establishing a new, highly volatile lower baseline.
*   The **Open**, **High**, and **Low** prices are strongly predictive of the **Close** price.
*   The Random Forest Regressor handled the non-linear market crash dynamics optimally, yielding the best RMSE. Given the business context, minimizing RMSE was prioritized to heavily penalize extreme predictive errors during volatile periods.
