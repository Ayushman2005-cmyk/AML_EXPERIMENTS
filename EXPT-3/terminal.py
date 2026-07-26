import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
def train_and_evaluate(X, y, title):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    p = m.predict(X_te)
    mse, mae, r2 = metrics.mean_squared_error(y_te, p), metrics.mean_absolute_error(y_te, p), metrics.r2_score(y_te, p)
    rmse = np.sqrt(mse)
    eq = f"y = {m.intercept_:.4f}" + "".join([f" + {c:.4f} * {col}" for c, col in zip(m.coef_, X.columns)])
    print(f"\n--- {title} ---\nEq: {eq}\nMSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return m, X_te, y_te, p, [mae, mse, rmse, r2]
def plot_chart(ctype, x, y, title, xl, yl, p=None):
    plt.figure()
    if ctype == 'scatter_line':
        plt.scatter(x, y, color='b', label='Actual')
        plt.plot(x, p, color='r', lw=2, label='Regression')
    elif ctype == 'actual_vs_pred':
        plt.scatter(x, y, color='purple', label='Predictions')
        plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='Ideal Fit')
    elif ctype == 'metrics_bar':
        bars = plt.barh(["MAE", "MSE", "RMSE", "R2"], x, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
        for b in bars: plt.annotate(f"{b.get_width():.4f}", xy=(b.get_width(), b.get_y() + b.get_height()/2), va="center")
    if ctype != 'metrics_bar': plt.legend()
    plt.title(title); plt.xlabel(xl); plt.ylabel(yl); plt.grid(True); plt.show()
df = pd.read_csv('https://raw.githubusercontent.com/Ayushman2005-cmyk/AML_EXPERIMENTS/main/EXPT-3/studentGradeDataSet.csv')
print(f"Shape: {df.shape}\n\nNulls:\n{df.isnull().sum()}\n\nStats:\n{df.describe()}")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm').set_title('Correlation Heatmap'); plt.show()
df.plot(kind='box', subplots=True, figsize=(15, 4)); plt.tight_layout(); plt.show()
X, y = df[['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4']], df['SEM 5']
best = X.corrwith(y).abs().idxmax()
print(f"\nTop correlated feature with SEM 5 is: '{best}'")
_, X_te_s, y_te_s, p_s, m_s = train_and_evaluate(df[[best]], y, 'Simple Linear Regression')
plot_chart('scatter_line', X_te_s, y_te_s, f'Simple LR: {best} vs SEM 5', best, 'SEM 5', p_s)
plot_chart('metrics_bar', m_s, None, 'Simple LR Metrics', 'Value', 'Metric')
m_m, _, y_te_m, p_m, m_m_mets = train_and_evaluate(X, y, 'Multiple Linear Regression')
plot_chart('actual_vs_pred', y_te_m, p_m, 'Multiple LR: Actual vs Predicted', 'Actual', 'Predicted')
plot_chart('metrics_bar', m_m_mets, None, 'Multiple LR Metrics', 'Value', 'Metric')
print("\n--- Enter Marks to Predict SEM 5 ---")
u_inp = pd.DataFrame([[float(input(f"Enter SEM {i} marks: ")) for i in range(1, 5)]], columns=X.columns)
print(f"\nThe predicted 5th Semester mark is: {m_m.predict(u_inp)[0]:.2f}")