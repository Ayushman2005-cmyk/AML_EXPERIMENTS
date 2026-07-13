import numpy as np
import matplotlib.pyplot as plt
print("Linear Regression Calculator")
print("Enter your data points separated by spaces (e.g., 1 2 3 4 5)")
x_input = input("Enter x values: ")
y_input = input("Enter y values: ")
try:
    x = np.array([float(i) for i in x_input.split()])
    y = np.array([float(i) for i in y_input.split()])
except ValueError:
    print("Error: Please enter only numbers separated by spaces.")
    exit()
if len(x) == 0 or len(y) == 0:
    print("Error: No data provided.")
    exit()
if len(x) != len(y):
    print(f"Error: Mismatched data. You entered {len(x)} x-values and {len(y)} y-values.")
    exit()
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_sq = np.sum(x**2)
denominator = (n * sum_x_sq - sum_x**2)
if denominator == 0:
    print("Error: Cannot calculate regression line. All x values are identical.")
    exit()
m = (n * sum_xy - sum_x * sum_y) / denominator
c = (sum_y - m * sum_x) / n
y_pred = m * x + c
mae = np.mean(np.abs(y - y_pred)) 
mse = np.mean((y - y_pred)**2)    
rmse = np.sqrt(mse)              
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
if ss_tot == 0:
    r2 = 1.0
else:
    r2 = 1 - (ss_res / ss_tot)
residuals = y - y_pred
print("\n--- Results ---")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")
print(f"Equation: y = {m:.4f}x + {c:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(x, y, color='blue', label='Data Points')
ax1.plot(x, y_pred, color='brown', linewidth=2, label=f'Regression Line\ny = {m:.2f}x + {c:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression Analysis')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.scatter(x, residuals, color='orange', label='Residuals (Errors)')
metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_xlabel('x')
ax2.set_ylabel('Error (Actual y - Predicted y)')
ax2.set_title('Evaluation: Residual Scatter Plot')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('regression_and_evaluation.png')
print("\nPlot saved as 'regression_and_evaluation.png'. Opening plot window...")
plt.show()