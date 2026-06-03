/**
 * Custom Machine Learning models implemented from scratch in JavaScript.
 */

// ============================================================================
// 1. BASELINE MOVING AVERAGE MODEL
// ============================================================================
export class MovingAverageModel {
  constructor(windowSize = 7) {
    this.windowSize = windowSize;
    this.history = [];
  }

  fit(X, y) {
    // For moving average, we just save the training target values
    this.history = [...y];
  }

  predict(X) {
    // X is an array of feature objects (including dates, etc.)
    // We predict sequentially, using a sliding window.
    const predictions = [];
    const tempHistory = [...this.history];

    for (let i = 0; i < X.length; i++) {
      // Take average of last `windowSize` values
      const start = Math.max(0, tempHistory.length - this.windowSize);
      const window = tempHistory.slice(start);
      const average = window.reduce((sum, val) => sum + val, 0) / (window.length || 1);
      
      predictions.push(average);
      tempHistory.push(average); // Feedback loop for multi-step forecast
    }
    return predictions;
  }
}

// ============================================================================
// 2. MULTIPLE LINEAR REGRESSION (GRADIENT DESCENT)
// ============================================================================
export class LinearRegressionModel {
  constructor(learningRate = 0.05, epochs = 500) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.weights = []; // [bias, w_trend, w_month_sin, w_month_cos, w_dayofweek]
    this.lossHistory = [];
    this.trendMin = 0;
    this.trendMax = 1;
  }

  /**
   * Helper to build feature vector
   * @param {Object} item 
   * @returns {number[]} 5-dim feature vector
   */
  _getFeatures(item, index) {
    // Scale trend index
    const scaledTrend = (index - this.trendMin) / (this.trendMax - this.trendMin || 1);
    
    // Cyclical Month: month is 1-12
    const m = item.month || 1;
    const monthSin = Math.sin((2 * Math.PI * m) / 12);
    const monthCos = Math.cos((2 * Math.PI * m) / 12);
    
    // Day of week: 0-6
    const dow = item.dayofweek || 0;
    const dayOfWeekScaled = dow / 6; // scale to [0, 1]

    return [1, scaledTrend, monthSin, monthCos, dayOfWeekScaled];
  }

  fit(X, y) {
    // Establish trend scale from training indexes
    this.trendMin = 0;
    this.trendMax = X.length - 1 || 1;

    // Feature matrix
    const features = X.map((item, idx) => this._getFeatures(item, idx));
    const nSamples = features.length;
    const nFeatures = features[0].length;

    // Initialize weights to small random numbers
    this.weights = Array(nFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.lossHistory = [];

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalSquareError = 0;
      const gradients = Array(nFeatures).fill(0);

      // Compute gradients
      for (let i = 0; i < nSamples; i++) {
        const xi = features[i];
        const yi = y[i];
        
        // Predict
        let pred = 0;
        for (let j = 0; j < nFeatures; j++) {
          pred += weights_dot_product(this.weights, xi);
        }
        
        const error = pred - yi;
        totalSquareError += error * error;

        for (let j = 0; j < nFeatures; j++) {
          gradients[j] += error * xi[j];
        }
      }

      // Update weights
      for (let j = 0; j < nFeatures; j++) {
        this.weights[j] -= (this.learningRate * gradients[j]) / nSamples;
      }

      const mse = totalSquareError / nSamples;
      this.lossHistory.push(mse);
    }
  }

  predict(X, startIdx = 0) {
    return X.map((item, idx) => {
      // Note: startIdx lets us map future indexes beyond training range
      const xi = this._getFeatures(item, startIdx + idx);
      return weights_dot_product(this.weights, xi);
    });
  }
}

function weights_dot_product(w, x) {
  let sum = 0;
  for (let i = 0; i < w.length; i++) {
    sum += w[i] * x[i];
  }
  return sum;
}

// ============================================================================
// 3. HOLT-WINTERS TRIPLE EXPONENTIAL SMOOTHING (ADDITIVE)
// ============================================================================
export class HoltWintersModel {
  /**
   * @param {number} alpha - Level smoothing
   * @param {number} beta - Trend smoothing
   * @param {number} gamma - Seasonal smoothing
   * @param {number} period - Seasonal cycle (7 for daily, 12 for monthly)
   */
  constructor(alpha = 0.2, beta = 0.1, gamma = 0.3, period = 7) {
    this.alpha = alpha;
    this.beta = beta;
    this.gamma = gamma;
    this.period = period;
    
    // Fit state
    this.level = 0;
    this.trend = 0;
    this.seasonals = [];
  }

  fit(X, y) {
    const p = this.period;
    const n = y.length;
    
    if (n < 2 * p) {
      // Fallback if dataset is too small for Holt-Winters seasonality
      // Simple exponential smoothing
      this.level = y.reduce((a, b) => a + b, 0) / n;
      this.trend = 0;
      this.seasonals = Array(p).fill(0);
      return;
    }

    // 1. Initialize Level (average of first season)
    let initLevel = 0;
    for (let i = 0; i < p; i++) {
      initLevel += y[i];
    }
    initLevel /= p;

    // 2. Initialize Trend (average slope between corresponding points in first and second seasons)
    let initTrend = 0;
    for (let i = 0; i < p; i++) {
      initTrend += (y[i + p] - y[i]) / p;
    }
    initTrend /= p;

    // 3. Initialize Seasonals
    const initSeasonals = Array(p).fill(0);
    const numSeasons = Math.floor(n / p);
    const seasonAverages = [];
    
    for (let s = 0; s < numSeasons; s++) {
      let sum = 0;
      for (let i = 0; i < p; i++) {
        sum += y[s * p + i];
      }
      seasonAverages.push(sum / p);
    }

    for (let i = 0; i < p; i++) {
      let val = 0;
      for (let s = 0; s < numSeasons; s++) {
        val += y[s * p + i] - seasonAverages[s];
      }
      initSeasonals[i] = val / numSeasons;
    }

    // Run smoothing updates over training data
    let levels = [initLevel];
    let trends = [initTrend];
    let seasonals = [...initSeasonals];

    for (let i = 0; i < n; i++) {
      const val = y[i];
      const prevL = levels[levels.length - 1];
      const prevT = trends[trends.length - 1];
      const prevS = seasonals[i]; // Corresponding seasonal from prior cycle

      // Additive HW formulas
      const L = this.alpha * (val - prevS) + (1 - this.alpha) * (prevL + prevT);
      const T = this.beta * (L - prevL) + (1 - this.beta) * prevT;
      const S = this.gamma * (val - L) + (1 - this.gamma) * prevS;

      levels.push(L);
      trends.push(T);
      seasonals.push(S); // Append new seasonal calculation for future cycles
    }

    // Save terminal state for predictions
    this.level = levels[levels.length - 1];
    this.trend = trends[trends.length - 1];
    // Keep only the last `period` seasonal indices
    this.seasonals = seasonals.slice(seasonals.length - p);
  }

  predict(X) {
    const predictions = [];
    const p = this.period;
    
    for (let m = 1; m <= X.length; m++) {
      // Forecast m steps ahead: L + m*T + S_(t + m - p)
      const seasonalIndex = (m - 1) % p;
      const seasonalFactor = this.seasonals[seasonalIndex] || 0;
      const pred = this.level + m * this.trend + seasonalFactor;
      predictions.push(Math.max(0, pred)); // sales can't be negative
    }
    return predictions;
  }
}

// ============================================================================
// 4. DECISION TREE REGRESSOR FROM SCRATCH
// ============================================================================
class DecisionTreeNode {
  constructor(feature = null, threshold = null, left = null, right = null, value = null) {
    this.feature = feature;       // Name of the splitting feature
    this.threshold = threshold;   // Splitting threshold value
    this.left = left;             // Left subtree
    this.right = right;           // Right subtree
    this.value = value;           // Target prediction if it's a leaf node
  }

  isLeaf() {
    return this.value !== null;
  }
}

export class DecisionTreeRegressor {
  constructor(maxDepth = 5, minSamplesSplit = 10) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.root = null;
  }

  fit(X, y) {
    // Add sequential trend index to feature objects to capture time
    const data = X.map((item, idx) => ({
      ...item,
      trendIndex: idx,
      target: y[idx]
    }));

    const features = ['trendIndex', 'month', 'day', 'dayofweek'];
    this.root = this._buildTree(data, features, 0);
  }

  _buildTree(data, features, depth) {
    const nSamples = data.length;

    // Check stop criteria
    if (depth >= this.maxDepth || nSamples < this.minSamplesSplit || this._variance(data) < 1e-4) {
      const leafValue = data.reduce((sum, item) => sum + item.target, 0) / (nSamples || 1);
      return new DecisionTreeNode(null, null, null, null, leafValue);
    }

    let bestFeature = null;
    let bestThreshold = null;
    let bestVarianceReduction = -1;
    let bestLeftData = [];
    let bestRightData = [];

    const parentVariance = this._variance(data);

    // Scan features for best split
    for (const feature of features) {
      // Get all unique values as candidate thresholds
      const values = data.map(item => item[feature]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
      
      // Try midpoints of adjacent unique values
      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        
        const left = [];
        const right = [];
        for (const item of data) {
          if (item[feature] <= threshold) {
            left.push(item);
          } else {
            right.push(item);
          }
        }

        if (left.length === 0 || right.length === 0) continue;

        // Calculate variance reduction
        const leftVar = this._variance(left);
        const rightVar = this._variance(right);
        const weightLeft = left.length / nSamples;
        const weightRight = right.length / nSamples;
        const childrenVariance = weightLeft * leftVar + weightRight * rightVar;
        const varianceReduction = parentVariance - childrenVariance;

        if (varianceReduction > bestVarianceReduction) {
          bestVarianceReduction = varianceReduction;
          bestFeature = feature;
          bestThreshold = threshold;
          bestLeftData = left;
          bestRightData = right;
        }
      }
    }

    // If no split reduces variance, return a leaf
    if (bestVarianceReduction <= 0) {
      const leafValue = data.reduce((sum, item) => sum + item.target, 0) / nSamples;
      return new DecisionTreeNode(null, null, null, null, leafValue);
    }

    // Build children
    const leftChild = this._buildTree(bestLeftData, features, depth + 1);
    const rightChild = this._buildTree(bestRightData, features, depth + 1);

    return new DecisionTreeNode(bestFeature, bestThreshold, leftChild, rightChild, null);
  }

  _variance(data) {
    if (data.length === 0) return 0;
    const mean = data.reduce((sum, item) => sum + item.target, 0) / data.length;
    let sumSqDiff = 0;
    for (let i = 0; i < data.length; i++) {
      sumSqDiff += Math.pow(data[i].target - mean, 2);
    }
    return sumSqDiff / data.length;
  }

  predict(X, startIdx = 0) {
    return X.map((item, idx) => {
      // Map back to feature object format expected by split rules
      const featureObj = {
        ...item,
        trendIndex: startIdx + idx
      };
      return this._predictItem(this.root, featureObj);
    });
  }

  _predictItem(node, item) {
    if (node.isLeaf()) {
      return node.value;
    }
    const val = item[node.feature];
    if (val <= node.threshold) {
      return this._predictItem(node.left, item);
    } else {
      return this._predictItem(node.right, item);
    }
  }
}
