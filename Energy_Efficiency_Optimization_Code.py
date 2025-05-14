
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simulated Dataset
data = {
    'hour': list(range(24)),
    'temperature': np.random.randint(20, 35, size=24),
    'energy_consumption': [100 + i * 3 + np.random.randint(-10, 10) for i in range(24)]
}
df = pd.DataFrame(data)

# Features and Labels
x = df[['hour', 'temperature']]
y = df['energy_consumption']

# Model Training
model = LinearRegression()
model.fit(x, y)

# Prediction for Next Day
future_hours = pd.DataFrame({
    'hour': list(range(24)),
    'temperature': np.random.randint(20, 35, size=24)
})
predicted_energy = model.predict(future_hours)

# Optimization: Simulated 10% Reduction
optimized_energy = predicted_energy * 0.9

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(future_hours['hour'], predicted_energy, label='Predicted Usage', color='blue', marker='o')
plt.plot(future_hours['hour'], optimized_energy, label='Optimized Usage', color='green', marker='x')
plt.title("Energy Usage Prediction & Optimization")
plt.xlabel("Hour of Day")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_optimization_plot.png")
plt.show()
