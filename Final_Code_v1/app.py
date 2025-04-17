from flask import Flask, render_template, request, session, url_for
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime, timedelta
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Force Matplotlib to use the 'Agg' backend
import matplotlib.pyplot as plt
import os  # For file operations
import uuid  # For generating unique filenames

app = Flask(__name__)
app.secret_key = "your_secret_key"  # **Important!** Set a strong secret key
app.config['IMAGE_FOLDER'] = 'static/images'  # Folder to store images
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)  # Create the folder if it doesn't exist

# --- Stock Data Fetching and Model Training (Global Scope) ---
data = yf.download("INTU", period="3y")  # Fetch data once
historical_graph_filename = None  # Store filename instead of base64
if not data.empty:
    data['Days'] = (data.index - data.index.min()).days
    X = data[['Days']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    data['MA30'] = data['Close'].rolling(window=30).mean()  # Calculate 30-day moving average

    def create_historical_graph(dates, prices, ma30):
        """Creates a line graph for historical stock data, including the 30-day MA, and saves it to a file."""
        global historical_graph_filename
        plt.figure(figsize=(10, 6))
        plt.plot(dates, prices, label='Historical Data')
        plt.plot(dates, ma30, label='30-Day MA', linestyle='--')  # Plot moving average
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Intuit Stock Price - Historical')
        plt.legend()
        plt.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Generate a unique filename
        historical_graph_filename = f"{uuid.uuid4()}_historical.png"  # Unique filename for historical graph
        filepath = os.path.join(app.config['IMAGE_FOLDER'], historical_graph_filename)
        plt.savefig(filepath, format='png')
        plt.close()
        return historical_graph_filename  # Return the filename

    def create_prediction_graph(dates, prices, predicted_dates, predicted_prices, ma30):
        """Creates a line graph for historical and predicted stock data, including the 30-day MA, and saves it to a file."""
        filename = f"{uuid.uuid4()}_prediction.png"  # Unique filename for prediction graph
        filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)

        plt.figure(figsize=(10, 6))
        plt.plot(dates, prices, label='Historical Data')
        plt.plot(predicted_dates, predicted_prices, label='Predicted Data', linestyle='--')
        plt.plot(dates, ma30, label='30-Day MA', linestyle=':')  # Plot moving average (different style)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Intuit Stock Price - Historical & Predicted')
        plt.legend()
        plt.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(filepath, format='png')
        plt.close()
        return filename  # Return the filename

    historical_graph_filename = create_historical_graph(data.index, data['Close'], data['MA30'])

# --- Flask Routes ---
@app.route('/')
def index():
    global historical_graph_filename  # Access the global variable
    current_year = datetime.now().year
    historical_graph_url = url_for('static', filename=f"images/{historical_graph_filename}") if historical_graph_filename else None
    return render_template('index.html', raw_data=data.to_html(classes='data'), historical_graph_url=historical_graph_url, predicted_graph_url=None, current_year=current_year, predictions=None, model_error=None)

@app.route('/predict', methods=['POST'])
def predict():
    global historical_graph_filename  # Access the global variable
    num_days = int(request.form.get('days', 30))  # Get days from form
    predictions_data = None
    predicted_graph_filename = None

    if historical_graph_filename and not data.empty:  # Use the pre-computed data and model
        last_date = data.index[-1].to_pydatetime().date()
        start_train_date = data.index.min().to_pydatetime().date()
        future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]  # Calculate future dates
        last_train_day = (pd.Timestamp(last_date) - pd.Timestamp(start_train_date)).days
        future_days_np = np.array([[last_train_day + i] for i in range(1, num_days + 1)])  # Use i directly for days
        predicted_prices = model.predict(future_days_np)
        predictions = list(zip(future_dates, predicted_prices))
        predictions_data = [{"date": date.strftime('%Y-%m-%d'), "price": f"{price[0]:.2f}"} for date, price in predictions]
        # Fix: Check if predicted_prices is not None
        predicted_graph_filename = create_prediction_graph(data.index, data['Close'], future_dates, [p[0] for p in predicted_prices] if predicted_prices is not None else [], data['MA30'])

    current_year = datetime.now().year
    historical_graph_url = url_for('static', filename=f"images/{historical_graph_filename}") if historical_graph_filename else None
    predicted_graph_url = url_for('static', filename=f"images/{predicted_graph_filename}") if predicted_graph_filename else None
    return render_template('index.html', raw_data=data.to_html(classes='data'), historical_graph_url=historical_graph_url, predicted_graph_url=predicted_graph_url, current_year=current_year, predictions=predictions_data, model_error=None)

if __name__ == '__main__':
    app.run(debug=True)