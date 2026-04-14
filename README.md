# California Housing Price Prediction

A small machine learning project based on **scikit-learn** that explores California housing price prediction with **Linear Regression**.

This repository contains:

- a **baseline model** for quick prediction and visualization
- an **experiment version** that compares different preprocessing settings and model complexities

---

## Project Goals

This project is designed to practice a complete beginner-friendly ML workflow:

- load and clean real-world data
- train a regression model
- evaluate performance with multiple metrics
- compare different experimental settings
- visualize prediction results clearly

---

## Dataset

The project uses the **California Housing Dataset** from scikit-learn.

Target: median house value

A simple filtering step is applied in the experiments:

- `y < 5.0`

This removes some high-value outliers and keeps the task more stable for learning and comparison.

---

## Files

### 1. Baseline script
This version trains a plain **LinearRegression** model and shows a single scatter plot of:

- true house prices
- predicted house prices

It is useful for understanding the basic pipeline:
data loading → splitting → training → prediction → evaluation → visualization

### 2. Experiment script
This version adds:

- **PolynomialFeatures** for model complexity control
- **StandardScaler** for normalization
- **cross_val_score** for 5-fold cross validation
- a **2×2 subplot figure** for comparison

It runs four experiments:

- no scaling + degree 1
- scaling + degree 1
- no scaling + degree 2
- scaling + degree 2

---

## Experiment Design

The second script compares two ideas:

### Feature scaling
Checks whether standardization changes model performance.

### Polynomial expansion
Checks whether increasing feature complexity helps or hurts the model.

The results show:

- degree 1 behaves normally and is stable
- degree 2 performs much worse, which suggests overfitting
- scaling has almost no effect on linear regression in this setup

---

## Visualization

The plots show **true house prices** on the x-axis and **predicted house prices** on the y-axis.

The diagonal red line represents perfect prediction:

- points closer to the line indicate better predictions
- points far away from the line indicate larger errors

In the experiment script, the results are displayed as a **2×2 grid** for easy comparison.

---

## Project Structure

```text
├── main.py                  # Baseline linear regression script
├── experiment.py            # Comparison experiment script
├── README.md                # Project description
└── true_vs_pred.png         # Optional saved plot image
