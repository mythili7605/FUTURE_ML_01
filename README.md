# Sales & Demand Forecasting System (Vanilla JS)

An interactive, premium Sales & Demand Forecasting Dashboard built from scratch using vanilla HTML, CSS, and JavaScript. The system processes raw retail transaction CSV logs, aggregates them, runs feature engineering, trains custom-built machine learning models, compares their accuracy metrics, and projects future sales demand.

## 🚀 How to Run Locally

### Prerequisites
- [Node.js](https://nodejs.org/) installed on your machine.

### Installation & Launch
1. Open terminal and navigate to the project directory:
   ```bash
   cd task1
   ```
2. Install the lightweight development server:
   ```bash
   npm install
   ```
3. Launch the local dev server:
   ```bash
   npm run dev
   ```
4. Open the displayed URL in your browser (typically `http://localhost:5173`).

---

## 🛠️ Architecture & Project Structure

The project has zero complex framework dependencies and runs entirely in the client's browser. It uses **Chart.js** for animated charts, **PapaParse** for CSV loading, and **Lucide Icons** for vector graphic icons.

```
task1/
├── index.html            # Single Page Application layout
├── styles.css            # Dark mode glassmorphic CSS styling
├── js/
│   ├── mathUtils.js      # Statistical metrics (MAE, RMSE, MAPE)
│   ├── dataProcessor.js  # CSV loading, date parsing, aggregations, features
│   ├── models.js         # Custom ML algorithms written from scratch
│   └── app.js            # Controller connecting UI elements and charts
├── train (1).csv         # Workspace retail dataset
├── package.json          # Node configurations
└── README.md             # Documentation
```

---

## 🧠 Machine Learning Models Built from Scratch

To satisfy the "from scratch" requirement, the models in [models.js](file:///c:/Users/Busi/OneDrive/Desktop/task1/js/models.js) are implemented purely in JavaScript:

### 1. Multiple Linear Regression (Gradient Descent)
Fits a linear combination of features to predict sales:
$$\hat{y} = w_0 + w_1 x_{trend} + w_2 x_{\sin} + w_3 x_{\cos} + w_4 x_{dow}$$

- **Feature Engineering**:
  - *Time Trend*: A normalized linear index tracking chronological sequence.
  - *Cyclical Month Encoding*: Month ($m \in [1, 12]$) is projected onto sine and cosine circles:
    $$x_{\sin} = \sin\left(\frac{2\pi m}{12}\right), \quad x_{\cos} = \cos\left(\frac{2\pi m}{12}\right)$$
    This ensures that December ($12$) and January ($1$) are recognized as temporally adjacent, allowing the model to capture year-end seasonality.
  - *Day of Week*: Scaled day index ($0$ to $6$).
- **Optimized via Gradient Descent**: Weights are updated iteratively to minimize Mean Squared Error (MSE). The training loss history is recorded and plotted.

### 2. Holt-Winters Triple Exponential Smoothing
An additive time-series forecasting model that splits data into level, trend, and seasonal indices. It is ideal for sales data with strong cyclical patterns:
- **Level**: $L_t = \alpha(Y_t - S_{t-p}) + (1-\alpha)(L_{t-1} + T_{t-1})$
- **Trend**: $T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$
- **Seasonal**: $S_t = \gamma(Y_t - L_t) + (1-\gamma)S_{t-p}$
- **Forecast**: $\hat{Y}_{t+m} = L_t + m T_t + S_{t+m-p \pmod p}$

Where $p$ is the seasonal period (auto-detected: 7 for daily, 12 for monthly, 4 for weekly).

### 3. Decision Tree Regressor
A non-parametric model that recursively splits features (trend index, month, day of month, day of week) to minimize sample variance:
- Evaluates split thresholds by maximizing **Variance Reduction**:
  $$\text{Reduction} = \text{Var}_{\text{parent}} - \left( \frac{n_{\text{left}}}{n} \text{Var}_{\text{left}} + \frac{n_{\text{right}}}{n} \text{Var}_{\text{right}} \right)$$
- Fits to the specified `maxDepth` and predicts step outcomes based on decision thresholds.

### 4. Simple Moving Average
A baseline forecasting model that averages the last $k$ values recursively for future forecasts, establishing a benchmark for checking the accuracy of complex models.

---

## 📊 Features & UI Elements

- **Obsidian Dark Mode**: Glassmorphism UI cards utilizing `backdrop-filter: blur(12px)` and subtle glowing indicators.
- **Dynamic Aggregation Selector**: Instantly toggle data views between Daily, Weekly, and Monthly sales.
- **Granular Filters**: Filter the dataset by **Product Category**, **Sub-Category**, or **Region** before training, allowing localized demand forecasting.
- **Interactive Evaluation**: Inspect MAE, RMSE, and MAPE metrics side-by-side to choose the best model.
- **CSV Data Export**: Export future forecast projections to a downloadable CSV file.
