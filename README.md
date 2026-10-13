## INTRODUCTION TO DEEP LEARNING : MLP for EUR/USD Direction Prediction

Developed a **Multi-Layer Perceptron (MLP)** using TensorFlow/Keras to predict the daily direction of EUR/USD.

* **Data:** Historical Forex data (2000–2026) via `yfinance`.
* **Features:** Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) and returns.
* **Preprocessing:** Chronological 80/20 split and MinMax scaling.
* **Architecture:** Dense layers (128 → 64 → 32 → 1), ReLU, Sigmoid, and Dropout (30%).
* **Training:** Adam optimizer and binary cross-entropy loss.
* **Evaluation:** Accuracy, classification report, confusion matrix, and learning curves.

**Technologies:** Python, TensorFlow/Keras, Pandas, pandas-ta, Scikit-learn, Matplotlib, yfinance.

### Installation

```bash
pip install tensorflow yfinance pandas scikit-learn matplotlib pandas-ta
```
## INTROODUCTION TO Deep Learning 2 : Model Optimization & Backtesting

Improved the Day 1 EUR/USD MLP by implementing advanced regularization, automated training, and realistic financial backtesting.

* **Feature Engineering:** 14 technical indicators, including SMA, EMA, RSI, MACD, ATR, and returns.
* **Regularization:** Dropout, L2 regularization, and Batch Normalization to reduce overfitting.
* **Training Optimization:** EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
* **Evaluation:** Precision, Recall, F1-score, AUC-ROC, and decision threshold tuning.
* **Backtesting:** Strategy vs. Buy & Hold, incorporating 0.03% transaction costs and avoiding look-ahead bias.
* **Visualizations:** Learning curves and cumulative strategy returns.

**Technologies:** Python, TensorFlow/Keras, Pandas, Scikit-learn, yfinance, Matplotlib.

**Requirements:** Python 3.10–3.13.


## Deep Learning LSTM : Long short term Memory

Developed an **LSTM neural network** to predict the next-day EUR/USD price direction using 60-day historical sequences, with a production-ready inference pipeline.

* **Time-Series Modeling:** 60-day lookback windows transformed into 3D tensors.
* **LSTM Architecture:** Stacked recurrent layers with Dropout regularization.
* **Trading Strategy:** Prediction thresholds, trading signals, and transaction costs.
* **Backtesting:** Evaluation of strategy performance using historical data.
* **Deployment:** Saved model, scaler, and parameters for automated daily inference.

**Technologies:** Python, TensorFlow/Keras, yfinance, pandas-ta, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, JSON.

### Installation

```bash
pip install tensorflow yfinance pandas-ta scikit-learn matplotlib joblib
```



## EUR/USD Closing Price Prediction — Machine Learning : Linear Regression

Developed a Machine Learning pipeline to predict the next-day EUR/USD closing price using **Linear Regression** and historical Forex data from MetaTrader 5.

* **Data Acquisition:** Real OHLCV data via the MetaTrader 5 Python API, with simulated data as a fallback.
* **Preprocessing:** Data cleaning and feature preparation using Pandas and NumPy.
* **Modeling:** Supervised learning with Scikit-learn Linear Regression.
* **Evaluation:** RMSE, MAE, \(R^2\), and directional accuracy.
* **Objective:** Forecast next-day closing prices and evaluate prediction errors in pips.

**Technologies:** Python, MetaTrader 5, Pandas, NumPy, Scikit-learn.

### Installation

```bash
pip install MetaTrader5 pandas numpy scikit-learn
```

## EUR/USD Market Direction Classification — Machine Learning: Logistic Regression

Developed a Machine Learning pipeline to predict the next-day EUR/USD market direction (up or down) using **Logistic Regression** and historical Forex data from MetaTrader 5.

* **Data Acquisition:** Historical daily price data via the MetaTrader 5 Python API.
* **Feature Engineering:** Technical indicators such as SMA, RSI, and volatility features.
* **Preprocessing:** Data cleaning, chronological train-test split, and feature standardization using Scikit-learn.
* **Modeling:** Binary classification using Scikit-learn Logistic Regression.
* **Evaluation:** Accuracy, Precision, Recall, F1-score, confusion matrix, and ROC-AUC.
* **Objective:** Identify next-day market direction while preventing data leakage.

**Technologies:** Python, MetaTrader 5, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

### Installation

```bash
pip install MetaTrader5 pandas numpy scikit-learn matplotlib seaborn
```
## EUR/USD Market Direction Classification — Decision Tree
Developed a Machine Learning pipeline to predict the next-day EUR/USD market direction (up or down) using a **Decision Tree Classifier** and historical daily Forex data.

* **Data Acquisition:** Historical EUR/USD data from Yahoo Finance or MetaTrader 5.
* **Feature Engineering:** Technical indicators for trend, momentum, and volatility.
* **Modeling:** Decision Tree classification to capture non-linear relationships.
* **Evaluation:** Accuracy, Precision, Recall, F1-score, and confusion matrix.
* **Objective:** Predict next-day market direction using interpretable decision rules.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, yfinance, MetaTrader 5.

**Installation**

```bash
pip install yfinance MetaTrader5 scikit-learn pandas numpy
```

## EUR/USD Market Direction Classification — Random Forest

Developed a Machine Learning pipeline to classify the next-day EUR/USD market direction using a **Random Forest Classifier** trained on historical daily Forex data.

* **Data Acquisition:** Historical EUR/USD prices from Yahoo Finance or MetaTrader 5.
* **Feature Engineering:** Technical indicators to identify trends, momentum, and volatility.
* **Modeling:** Random Forest ensemble combining multiple decision trees to improve prediction stability.
* **Evaluation:** Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* **Objective:** Evaluate ensemble learning for next-day market direction prediction.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, yfinance, MetaTrader 5.

**Installation**

```bash
pip install yfinance MetaTrader5 scikit-learn pandas numpy
```
## EUR/USD Market Direction Classification — XGBoost
Developed a Machine Learning pipeline to predict the next-day EUR/USD market direction using **XGBoost**, a gradient-boosting algorithm, and historical daily Forex data.

* **Data Acquisition:** Historical EUR/USD market data from Yahoo Finance or MetaTrader 5.
* **Feature Engineering:** Technical indicators and price-based features.
* **Modeling:** XGBoost classification to learn complex non-linear relationships.
* **Evaluation:** Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* **Objective:** Assess gradient boosting for daily EUR/USD direction classification.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, XGBoost, yfinance, MetaTrader 5.

**Installation**

```bash
pip install yfinance MetaTrader5 scikit-learn xgboost pandas numpy
```
## EUR/USD Market Direction Classification — Support Vector Machine (SVM)
Developed a Machine Learning pipeline to classify the next-day EUR/USD market direction using a **Support Vector Classifier (SVC)** with an RBF kernel and historical daily Forex data.

* **Data Acquisition:** Historical EUR/USD prices from Yahoo Finance or MetaTrader 5.
* **Feature Engineering:** Technical indicators for trend, momentum, and volatility.
* **Preprocessing:** Feature standardization using `StandardScaler` within a leakage-safe pipeline.
* **Modeling:** SVM with an RBF kernel to capture non-linear decision boundaries.
* **Evaluation:** Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* **Objective:** Investigate SVM performance in next-day market direction classification.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, yfinance, MetaTrader 5.

**Installation**

```bash
pip install yfinance MetaTrader5 scikit-learn pandas numpy
```

## Rienforcement Learning for Trading

### 1. L'Environnement (`TradingEnv`)
Un environnement de trading sur mesure et simplifié qui suit le formalisme d'un processus de décision de Markov (MDP) :
*   **Espace d'États (State)** : Une fenêtre glissante de rendements récents de l'actif combinée avec la position actuelle de l'agent (0 pour Cash, 1 pour Long).
*   **Espace d'Actions (Actions)** : 
    *   `0` : Conserver la position actuelle (Hold / Wait)
    *   `1` : Acheter (Buy - si la position actuelle est Cash)
    *   `2` : Vendre (Sell - si la position actuelle est Long)
*   **Récompense (Reward)** : Le profit et perte (P&L) réel généré par la variation du prix de l'actif (valorisation Mark-to-Market), diminué des frais de transaction (`cost = 0.001` par transaction).

### 2. Algorithme 1 : Q-Learning Tabulaire
*   Les rendements continus sont convertis en catégories discrètes (Baisse, Stable, Hausse) via une fonction de numérisation (`np.digitize`).
*   Mise à jour de la table Q par l'équation de Bellman classique.

### 3. Algorithme 2 : Deep Q-Network (DQN)
*   Utilisation de réseaux de neurones profonds (`policy_net` & `target_net`) pour estimer les Q-valeurs directement à partir d'états continus.
*   Mécanisme d'**Experience Replay** (avec un buffer) pour casser la corrélation temporelle des échantillons.
*   Stratégie d'exploration $\epsilon$-greedy avec décroissance temporelle de l'exploration.

# Fine-Tuning DistilBERT for FOMC Stance Detection

This exercise demonstrates how to fine-tune `distilbert-base-uncased` to classify Federal Reserve (FOMC) sentences into 4 stances: **Hawkish**, **Dovish**, **Neutral**, or **Irrelevant**.

## Quick Specifications
* **Dataset**: `gtfintechlab/federal_reserve_system` (subset `5768`)
* **Target (`CIBLE`)**: `stance_label`
* **Model**: DistilBERT sequence classifier (66M parameters)
* **Metrics**: Accuracy & Macro F1-score

## Workflow
1. **Data Preprocessing**: Map text labels to integers and tokenize sentences up to 128 tokens.
2. **Training Setup**: Fine-tune with a learning rate of `2e-5`, batch size of 16, for up to 8 epochs with early stopping (patience of 2 on F1 metric).
3. **Evaluation**: Compare performance against a majority-class baseline.

## Key Results
* **Model Accuracy**: **56.0%**
* **Model Macro F1**: **0.570** 
* **Baseline Macro F1**: **0.132** (The model significantly outperforms the baseline)

### Metrics Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Dovish** | 0.567 | 0.630 | 0.596 | 54 |
| **Hawkish** | 0.605 | 0.535 | 0.568 | 43 |
| **Irrelevant** | 0.667 | 0.571 | 0.615 | 7 |
| **Neutral** | 0.500 | 0.500 | 0.500 | 46 |