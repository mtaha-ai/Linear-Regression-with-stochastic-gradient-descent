from linear_model import SGDRegressor
from utils import Plot
from utils import Scaler
import pandas as pd

def main():
    df = pd.read_csv('data.csv')
    X = df['Area_sqft'].values
    y = df['Price'].values
    X = X.reshape(-1, 1)
    scaler = Scaler()
    X_scaled = scaler.MinMaxScaler(X)
    y_scaled = scaler.MinMaxScaler(y)
    model = SGDRegressor()
    model.fit(X_scaled, y_scaled)
    y_pred = model.predict(X_scaled)
    Plot.plot_best_fit(X_scaled, y_scaled, y_pred)
    Plot.plot_learning_curve(model.errors_history)

main()