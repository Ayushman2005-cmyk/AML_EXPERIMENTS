from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    y_name = request.form.get('y_name', 'y').strip()
    y_input = request.form.get('y_values', '')
    
    try:
        y_vals = np.array([float(val) for val in y_input.split(",") if val.strip()])
        num_data_points = len(y_vals)
    except ValueError:
        return "Error: Invalid numerical values for y.", 400

    x_data = {}
    x_names = []
    x_values_raw = []
    
    idx = 0
    while f'x_name_{idx}' in request.form:
        feat_name = request.form.get(f'x_name_{idx}').strip()
        feat_input = request.form.get(f'x_values_{idx}', '')
        
        if not feat_input.strip():
            idx += 1
            continue
            
        try:
            feat_vals = np.array([float(val) for val in feat_input.split(",") if val.strip()])
            if len(feat_vals) != num_data_points:
                return f"Error: '{feat_name}' has {len(feat_vals)} data points, but '{y_name}' has {num_data_points}.", 400
            
            x_names.append(feat_name)
            x_values_raw.append(feat_input)
            x_data[f'x{idx+1}'] = feat_vals
        except ValueError:
            return f"Error: Invalid numerical values inside feature '{feat_name}'.", 400
        idx += 1

    # Model Calculation - Scikit-Learn Multiple Linear Regression
    data = pd.DataFrame(x_data)
    model = LinearRegression()
    model.fit(data, y_vals)
    predictions = model.predict(data)

    mse = mean_squared_error(y_vals, predictions)
    mae = mean_absolute_error(y_vals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_vals, predictions)

    intercept = model.intercept_
    coefficients = model.coef_
    equation_terms = [f"({coef:.4f} * x{i+1})" for i, coef in enumerate(coefficients)]
    equation_str = f"y = {intercept:.4f} + " + " + ".join(equation_terms)
    slopes = {x_names[i]: f"{coef:.4f}" for i, coef in enumerate(coefficients)}

    plt.figure(figsize=(6, 4.2), dpi=150)
    plt.scatter(y_vals, predictions, color="#6366f1", edgecolor="#4338ca", linewidth=1.2, s=70, alpha=0.85, label="Actual vs Predicted")
    min_val = min(float(np.min(y_vals)), float(np.min(predictions)))
    max_val = max(float(np.max(y_vals)), float(np.max(predictions)))
    plt.plot([min_val, max_val], [min_val, max_val], color="#06b6d4", linestyle="--", linewidth=2, label="Ideal Fit Line")
    plt.xlabel(f"Actual {y_name}", fontsize=10, fontweight='bold', color="#1e293b")
    plt.ylabel(f"Predicted {y_name}", fontsize=10, fontweight='bold', color="#1e293b")
    plt.title("Model Prediction Accuracy Evaluation", fontsize=11, fontweight='bold', color="#0f172a", pad=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    plt.tight_layout()
    
    # Ensure static directory exists
    os.makedirs(app.static_folder, exist_ok=True)

    plot1_filename = 'accuracy_plot.png'
    plot1_path = os.path.join(app.static_folder, plot1_filename)
    plt.savefig(plot1_path, format="png", bbox_inches='tight', dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.2), dpi=150)
    metrics_names = ["R2 Score", "RMSE", "MAE", "MSE"]
    metrics_vals = [r2, rmse, mae, mse]
    colors = ["#ec4899", "#8b5cf6", "#0284c7", "#10b981"]
    bars = plt.barh(metrics_names, metrics_vals, color=colors, height=0.55, edgecolor="none")
    for bar in bars:
        width = bar.get_width()
        plt.annotate(f" {width:.4f}", xy=(width, bar.get_y() + bar.get_height() / 2), ha='left', va='center', fontweight='bold', color="#0f172a")    
    plt.xlabel("Computed Value", fontsize=10, fontweight='bold', color="#1e293b")
    plt.ylabel("Metric Name", fontsize=10, fontweight='bold', color="#1e293b")
    plt.title("Linear Regression Performance Metrics", fontsize=11, fontweight='bold', color="#0f172a", pad=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    plot2_filename = 'metrics_plot.png'
    plot2_path = os.path.join(app.static_folder, plot2_filename)
    plt.savefig(plot2_path, format="png", bbox_inches='tight', dpi=150)
    plt.close()

    x_raw_str = ";".join(x_values_raw)
    timestamp = int(time.time())

    return render_template('result.html', 
                           equation=equation_str, 
                           intercept=f"{intercept:.4f}", 
                           slopes=slopes, 
                           y_name=y_name,
                           y_input=y_input,
                           x_names=x_names,
                           x_raw_str=x_raw_str,
                           plot1_filename=plot1_filename, 
                           plot2_filename=plot2_filename,
                           timestamp=timestamp,
                           mse=f"{mse:.4f}",
                           mae=f"{mae:.4f}",
                           rmse=f"{rmse:.4f}",
                           r2=f"{r2:.4f}")

@app.route('/predict', methods=['POST'])
def predict():
    y_input = request.form['y_input']
    x_raw_str = request.form['x_raw_str']
    y_name = request.form['y_name']

    y_vals = np.array([float(val) for val in y_input.split(",") if val.strip()])
    feature_strings = x_raw_str.split(";")
    
    x_data = {}
    for i, f_str in enumerate(feature_strings):
        x_data[f'x{i+1}'] = [float(val) for val in f_str.split(",") if val.strip()]
        
    model = LinearRegression()
    model.fit(pd.DataFrame(x_data), y_vals)

    try:
        sample_dict = {}
        for idx in range(len(feature_strings)):
            val = float(request.form[f'pred_x_{idx}'])
            sample_dict[f'x{idx+1}'] = val

        predicted_val = model.predict(pd.DataFrame([sample_dict]))[0]
        return f"<div style='font-family: system-ui; text-align: center; padding: 60px 20px; background: #0f172a; color: white; min-height: 100vh; display: flex; align-items: center; justify-content: center;'><div style='background: rgba(30,41,59,0.8); border: 1px solid rgba(255,255,255,0.1); padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); max-width: 500px; width: 100%;'><h2 style='font-size: 1.6rem; margin-bottom: 20px; color: #38bdf8;'>Predicted {y_name} (y): <span style='color: #10b981; font-family: monospace; font-weight: 800;'>{predicted_val:.4f}</span></h2><a href='javascript:history.back()' style='background: #6366f1; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: bold; display: inline-block;'>← Go Back</a></div></div>"
    except Exception:
        return "Invalid input numbers entered for prediction.", 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)