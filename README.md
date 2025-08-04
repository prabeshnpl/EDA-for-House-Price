
# 🏡 House Price Prediction using Regression

This project aims to **predict house prices** based on various features in a housing dataset. It involves two key phases:
1. **Exploratory Data Analysis (EDA)** to understand and prepare the dataset
2. **Model Training** using multiple regression techniques

---

## 📁 Project Structure

```
├── dataset/   
    ├── test.csv                       # Test dataset
    └── train.csv                      # Train dataset
├── program/   
    ├── EDA.ipynb                      # Exploratory data analysis
    └── House Price Prediction.ipynb   # Preprocessing and model building
├── EDA.html  
├── House Price Prediction.html  
├── model.pkl                          # Trained model
├── README.md                          # Project documentation
```

---

## 📊 1. Exploratory Data Analysis (EDA)

The EDA process focuses on understanding the structure and characteristics of the dataset.

### Key Steps:
- Loaded and examined the shape and basic stats of the dataset
- Identified and visualized missing values
- Explored correlations between features and the target variable `SalePrice`
- Created visualizations like:
  - Heatmaps of correlation
  - Distribution plots for skewed features
  - Boxplots to see category-target relationships
- Selected meaningful features for modeling

### 🧠 Outcome:
EDA helped guide:
- Which columns to drop
- How to treat missing values
- Feature encoding and transformation decisions

---

## ⚙️ 2. Model Training and Evaluation

After EDA, we moved into feature engineering and modeling using pipelines.

### 🔧 Preprocessing:
- Dropped columns with excessive missing data
- Used `SimpleImputer` and `StandardScaler` for numeric features
- Applied `OneHotEncoder` (with `handle_unknown='ignore'`) to categorical variables
- Combined numeric and categorical pipelines with `ColumnTransformer`

### 📈 Models Used:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regressor (SVR)**
- **K-Nearest Neighbors Regressor**

Each model was evaluated using **R² score** with **cross-validation (cv=3)**.

### 🏆 Output:
A performance table showing R² scores for each model to help select the best performer.

---

### 📈 Final Model:

- **From the evaluation of each model, Gradient Boosting Regressor is selected due to high accuracy in it.**
- **Hyperparameter tuning using GridSearchCV**
- **Combined with PCA for dimension reduction**

### 🔍 Principal Component Analysis (PCA)

To reduce dimensionality and address multicollinearity, **Principal Component Analysis (PCA)** was applied after feature preprocessing. PCA helps in:

- Capturing the **most informative variance** in the data using fewer components
- Reducing the **curse of dimensionality**
- Speeding up model training time

#### Key Points:
- PCA was applied **after scaling** numeric features and one-hot encoding categorical variables.
- The explained variance ratio was used to **analyze how many components** were needed to retain >95% of the information.
- PCA-transformed data was used as input to selected model(Gradient Boosting Regressor) to compare performance with and without dimensionality reduction.

---

## ✅ Highlights

- Modular pipelines for clean preprocessing
- Robust handling of unknown categories in test data
- Multiple models benchmarked on the same pipeline
- Warnings captured for unknown categorical values
- Feature importance visualization

---

## 🚀 How to Run

1. Ensure you have the following packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run `program/ EDA.ipynb` to understand the data
3. Run `program/House Price Prediction.ipynb` to preprocess and train models

---

## 📌 Future Improvements


- Deploy best model using Flask or Streamlit
- Handle outliers more robustly
- Add logging and modular `.py` scripts

---

## 📂 Dataset

- Based on the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) dataset from Kaggle
- **Target**: `SalePrice`
- **Features**: Housing characteristics (area, type, quality, neighborhood, etc.)
