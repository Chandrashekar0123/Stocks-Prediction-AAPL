# Stock Price Prediction (AAPL)

Welcome to the **Stock Price Prediction** project, a machine learning-based model designed to forecast future stock prices of **Apple Inc. (AAPL)**. This app uses historical stock price data and advanced deep learning techniques to predict future prices, offering insights into the potential trends of the stock market.

---

## 🚨 **Problem Statement**
The stock market is volatile, and accurately predicting future stock prices is a challenging but important task. Investors and traders rely on forecasting models to make informed decisions. In this project, we aim to create a model that predicts future stock prices of **Apple Inc. (AAPL)** based on historical data. By doing so, we can provide a tool that helps investors anticipate market movements and make data-driven decisions.

---

## 🧑‍💻 **Project Description**
The project leverages a **Long Short-Term Memory (LSTM)** model to predict future stock prices. LSTM networks are well-suited for time series forecasting tasks, such as predicting stock prices, due to their ability to capture temporal dependencies. The app provides a user-friendly interface where users can input the stock symbol, start and end dates, and select the time interval for stock data analysis. The model outputs future stock price predictions, which are visualized through interactive charts.

## 🎬 Demo Video

[Watch the Stock Prediction Demo Video](https://raw.githubusercontent.com/Chandrashekar0123/Stocks-Prediction-AAPL/main/stock-prediction-video.mp4)

✅ The video walkthrough includes:
- Stock data visualization
- Prediction model explanation
- Real-time stock price forecasting
---

<img width="621" height="452" alt="image" src="https://github.com/user-attachments/assets/af1295a6-6903-4e39-982b-24a15907c1b0" />

Taking Input from User for prediction
---

## 🚀 **Features**
- **Stock Price Prediction**: Predict future stock prices for Apple Inc. (AAPL) based on past performance.
- **Visualization**: Interactive charts using Plotly to display predicted vs. actual stock prices.
- **User-Friendly Interface**: A clean and intuitive interface built with **Streamlit**.
- **Model**: LSTM-based deep learning model for stock price forecasting.
- **Data**: Historical stock data sourced from Yahoo Finance.

---

<img width="627" height="389" alt="image" src="https://github.com/user-attachments/assets/e019def5-9441-495b-a6c9-21b88f6d5bb7" />

OUTPUT predictions of Stocks

---
<img width="634" height="334" alt="image" src="https://github.com/user-attachments/assets/4cebe0d5-a9d6-420e-bdda-1f0fa6964119" />

Visual Representation of Stocks

---

## 🧑‍💻 **Installation & Setup**

To get started with this project, follow these simple steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Chandrashekar0123/Stocks-Prediction-AAPL.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd Stocks-Prediction-AAPL
    ```

3. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    ```

4. **Activate the virtual environment:**

    - On Windows:

      ```bash
      .\venv\Scripts\activate
      ```

    - On Mac/Linux:

      ```bash
      source venv/bin/activate
      ```

5. **Install required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 **How to Use**
1. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2. **Input**: Enter the stock symbol (e.g., `AAPL`), select the start and end dates, and choose the interval (Daily, Weekly, Monthly, Yearly).
3. **Prediction**: Click the "Predict Future Stock Prices" button to generate stock price predictions and view the visualized results.

---

## 📂 **Files in this Repository**
- `app.py`: Main Streamlit app to interact with the model and visualize predictions.
- `model.h5`: Trained LSTM deep learning model for stock price prediction.
- `scaler.pkl`: Scaler object used to normalize data for model input.
- `predicted_vs_actual.png`: A sample image showing predicted vs. actual stock prices.
- `requirements.txt`: List of required Python libraries for the project.

---

## ⚠️ **Disclaimer**
This is a **stock price prediction app** for educational purposes only. The predictions are based on historical data and deep learning models, but they are not guaranteed to be accurate. **Do not rely solely on this for investment decisions.**

---

<img width="277" height="496" alt="image" src="https://github.com/user-attachments/assets/85a825b4-8572-4dde-af82-59735782cc2a" />

---

## 📢 **Contributing**
Feel free to fork this repository, create issues, and submit pull requests for any enhancements or bug fixes!

---

## 📄 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### ✨ **Enjoy the stock market prediction experience!**
