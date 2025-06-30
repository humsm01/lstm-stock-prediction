# LSTM Stock Price Prediction
A time series forecasting model built using TensorFlow/Keras to predict Microsoft’s stock closing prices based on historical trends. This project showcases the application of deep learning (LSTM) for financial market analysis, using 10+ years of stock data.

---

## Project Overview

- **Goal:** Predict future stock prices using past performance with a focus on time-based dependencies.
- **Approach:** Used Long Short-Term Memory (LSTM) networks, ideal for sequential data like financial time series.
- **Data:** [MicrosoftStock.csv] — contains historical Microsoft (MSFT) stock data including Open, Close, Volume, etc.

---

## Technologies Used

- `Python` • `TensorFlow/Keras` • `scikit-learn` • `NumPy` • `Pandas` • `Matplotlib` • `Seaborn`

---

## Results

- **Test MAE:** ~84.37
- **Test RMSE:** ~84.44
- **R² Score:** 0.8544
- **Training Time:** ~11.8 seconds

Loss curve shows strong convergence over 20 epochs.  
Final predictions visually track closely with actual values.  
See [`prediction_plot.png`](./prediction_plot.png) for the full output.

---

## Model Architecture

- `LSTM(64, return_sequences=True)` 
- `LSTM(64)`  
- `Dense(128, activation='relu')`  
- `Dropout(0.5)`  
- `Dense(1)`
