import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def get_input(prompt, is_num_list=False):
    val = input(prompt).strip()
    if is_num_list:
        return np.array([float(x) for x in val.split(',') if x.strip()])
    return val

def fit_model(X, y):
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    metrics = [r2_score(y, preds), np.sqrt(mse), mean_absolute_error(y, preds), mse]
    return model, preds, metrics
def print_results(model, x_names, y_name):
    eq_terms = [f"({coef:.4f} * x{idx+1})" for idx, coef in enumerate(model.coef_)]
    equation_str = f"y = {model.intercept_:.4f} + " + " + ".join(eq_terms)
    print("\n==============================")
    print("Final Regression Equation:")
    print(equation_str)
    print("==============================")
    print(f"Intercept (b0): {model.intercept_:.4f}")
    for idx, coef in enumerate(model.coef_):
        print(f"Slope for {x_names[idx]} (b{idx+1}): {coef:.4f}")
    print()
def predict_value(model, x_names, y_name):
    choice = input(f"Do you want to enter sample data for predicting {y_name}? (yes/no): ").strip().lower()
    if choice in ['yes', 'y']:
        try:
            print(f"\n--- Enter Values for Prediction ---")
            sample_dict = {f'x{idx+1}': float(input(f"Enter value for {name} (x{idx+1}): ")) for idx, name in enumerate(x_names)}
            predicted_val = model.predict(pd.DataFrame([sample_dict]))[0]
            print(f">> Predicted {y_name} (y): {predicted_val:.4f}\n")
        except ValueError:
            print("Invalid input! Please enter numeric values.\n")
    elif choice in ['no', 'n']:
        print("Exiting prediction menu. Displaying graphs...\n")
    else:
        print("Invalid Input.\n")
def plot_graphs(y_vals, predictions, y_name, metrics):
    plt.scatter(y_vals, predictions, color="#10b981", edgecolor="#065f46", linewidth=1, s=50, alpha=0.85)
    min_val = min(float(np.min(y_vals)), float(np.min(predictions)))
    max_val = max(float(np.max(y_vals)), float(np.max(predictions)))
    plt.plot([min_val, max_val], [min_val, max_val], color="#64748b", linestyle="-.", linewidth=1.5)
    plt.xlabel(f"Actual {y_name}")
    plt.ylabel(f"Predicted {y_name}")
    plt.title("Model Prediction Accuracy Evaluation")
    plt.grid(True)
    plt.show()
    metrics_names = ["R2 Score", "RMSE", "MAE", "MSE"]
    colors = ["#4f46e5", "#3b82f6", "#60a5fa", "#93c5fd"]
    bars = plt.barh(metrics_names, metrics, color=colors)
    for bar in bars:
        width = bar.get_width()
        plt.annotate(f" {width:.4f}", xy=(width, bar.get_y() + bar.get_height() / 2), ha='left', va='center', fontweight='bold')  
    plt.xlabel("Computed Values")  
    plt.ylabel("Metric Value")
    plt.title("Linear Regression Performance Metrics")
    plt.grid(True)
    plt.show()
def main():
    try:
        y_name = get_input("Enter the name of the target variable y : ")
        y_vals = get_input(f"Enter comma-separated numerical values for {y_name} (y): ", is_num_list=True)
        num_data_points = len(y_vals)
        if num_data_points < 2:
            print("\nError: Please enter at least 2 or more data points.")
            return
        num_features = int(input("\nHow many feature variables (x) do you want to enter? "))
        x_data = {}
        x_names = []
        print("\nPlease enter comma-separated numerical values for each feature:")
        for i in range(num_features):
            feat_name = get_input(f"Enter the name of feature x{i+1} : ")
            x_names.append(feat_name)
            feat_vals = get_input(f"Enter values for {feat_name} (x{i+1}): ", is_num_list=True)
            if len(feat_vals) != num_data_points:
                print(f"\nError: '{feat_name}' has {len(feat_vals)} data points, but {y_name} has {num_data_points}.")
                print("\nExecution stopped due to mismatched data point lengths.")
                return
            x_data[f'x{i+1}'] = feat_vals
        df = pd.DataFrame(x_data)
        model, preds, metrics = fit_model(df, y_vals)
        print_results(model, x_names, y_name)
        predict_value(model, x_names, y_name)
        plot_graphs(y_vals, preds, y_name, metrics)
    except ValueError as e:
        print(f"\nError: Invalid input. {e}")
if __name__ == "__main__":
    main()