# 🏠 California Housing Price Prediction  
**A Structured Machine Learning Exploration with Linear Regression and Feature Engineering**

---

## 📌 Overview  
This project presents a systematic exploration of Linear Regression performance on the California Housing dataset using `scikit-learn`.

Rather than building a single baseline model, this repository focuses on analyzing how different modeling strategies affect performance, including feature scaling (Standardization), polynomial feature expansion (nonlinear modeling), and cross-validation for robustness evaluation.

The goal is to understand model behavior, not just generate predictions.

---

## 🎯 Objectives  
- Build a baseline Linear Regression model  
- Evaluate the impact of feature scaling and model complexity  
- Compare models using MSE, R², and cross-validation scores  
- Visualize prediction performance  

---

## 🧪 Experimental Design  

We conduct a 2 × 2 controlled experiment:

| Experiment | Scaling | Polynomial Degree | Description |
|----------|--------|------------------|------------|
| A | No | 1 | Baseline Linear Model |
| B | Yes | 1 | Scaled Linear Model |
| C | No | 2 | Nonlinear Model |
| D | Yes | 2 | Scaled Nonlinear Model |

This design enables clear comparison of the effect of feature scaling and model complexity.

---

## 📊 Dataset  

- Source: California Housing Dataset (`sklearn.datasets`)  
- Task: Regression — Predict median house value  
- Preprocessing: Filter `y < 5.0` to remove capped values and reduce noise  

---

## ⚙️ Pipeline  

Load Data  
↓  
Outlier Filtering (y < 5.0)  
↓  
Train/Test Split (train_size=0.2)  
↓  
[Optional] Polynomial Features  
↓  
[Optional] Standard Scaling  
↓  
Model Training (Linear Regression)  
↓  
Prediction  
↓  
Evaluation (MSE, R², Cross-validation)  
↓  
Visualization  

---

## 📈 Evaluation Metrics  

| Metric | Meaning | Interpretation |
|------|--------|--------------|
| MSE | Mean Squared Error | Lower is better |
| R² | Coefficient of Determination | Closer to 1 is better |
| CV Score | Cross-validation mean score | Reflects model stability |

---

## 📉 Visualization  

Each experiment produces a scatter plot:

- X-axis: True house prices  
- Y-axis: Predicted prices  
- Red line: Ideal prediction (y = x)  

The closer the points are to the diagonal, the better the model performance.

---

## 🧠 Key Insights  

- Linear Regression captures the overall trend but struggles with nonlinear patterns  
- Polynomial features increase model capacity but may introduce overfitting  
- Feature scaling improves numerical stability, especially for higher-degree features  
- Cross-validation helps distinguish real improvements from overfitting  

---

## 🗂️ Project Structure  

├── main.py              # Baseline model + visualization  
├── experiment.py        # Controlled experiments (scaling + polynomial)  
├── prediction_plot.png  # Example output  
└── README.md  

---

## 🔧 Dependencies  

pip install scikit-learn numpy matplotlib  

---

## 🚀 Usage  

Run baseline model:  
python main.py  

Run experiments:  
python experiment.py  

---

## 📌 Reproducibility  

- random_state = 42 ensures reproducibility  
- Fixed data split allows fair comparison across experiments  

---

## 🔍 Future Work  

- Introduce regularization (Ridge, Lasso)  
- Try tree-based models (Random Forest, Gradient Boosting)  
- Perform feature importance analysis  
- Optimize hyperparameters with Grid Search  

---

## 🏁 Conclusion  

This project demonstrates how linear models behave under different preprocessing and feature engineering strategies.

It provides a clear and structured understanding of model bias, variance, and the trade-off between simplicity and complexity.
