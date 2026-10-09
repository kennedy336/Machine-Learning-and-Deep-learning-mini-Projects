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


## LSTM : Long short term Memory

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



## EUR/USD Closing Price Prediction — Linear Regression

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
