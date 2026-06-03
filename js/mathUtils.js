/**
 * Statistical and mathematical utilities for evaluation metrics and data operations.
 */

/**
 * Calculates the Mean Absolute Error (MAE) between actual and predicted values.
 * MAE = (1 / n) * sum(|actual_i - predicted_i|)
 * @param {number[]} actual 
 * @param {number[]} predicted 
 * @returns {number}
 */
export function calculateMAE(actual, predicted) {
  if (!actual.length || actual.length !== predicted.length) return 0;
  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    sum += Math.abs(actual[i] - predicted[i]);
  }
  return sum / actual.length;
}

/**
 * Calculates the Root Mean Squared Error (RMSE) between actual and predicted values.
 * RMSE = sqrt((1 / n) * sum((actual_i - predicted_i)^2))
 * @param {number[]} actual 
 * @param {number[]} predicted 
 * @returns {number}
 */
export function calculateRMSE(actual, predicted) {
  if (!actual.length || actual.length !== predicted.length) return 0;
  let sumOfSquares = 0;
  for (let i = 0; i < actual.length; i++) {
    sumOfSquares += Math.pow(actual[i] - predicted[i], 2);
  }
  return Math.sqrt(sumOfSquares / actual.length);
}

/**
 * Calculates the Mean Absolute Percentage Error (MAPE) between actual and predicted values.
 * MAPE = (100% / n) * sum(|(actual_i - predicted_i) / actual_i|)
 * @param {number[]} actual 
 * @param {number[]} predicted 
 * @returns {number}
 */
export function calculateMAPE(actual, predicted) {
  if (!actual.length || actual.length !== predicted.length) return 0;
  let sumPercent = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    const act = actual[i];
    if (Math.abs(act) > 1e-5) { // Avoid division by zero
      sumPercent += Math.abs((act - predicted[i]) / act);
      count++;
    }
  }
  return count > 0 ? (sumPercent / count) * 100 : 0;
}

/**
 * Basic statistical summary helper.
 * @param {number[]} values 
 * @returns {{mean: number, std: number, min: number, max: number, sum: number}}
 */
export function getStats(values) {
  if (!values.length) return { mean: 0, std: 0, min: 0, max: 0, sum: 0 };
  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  let sqDiffSum = 0;
  for (let i = 0; i < values.length; i++) {
    sqDiffSum += Math.pow(values[i] - mean, 2);
  }
  const variance = sqDiffSum / values.length;
  const std = Math.sqrt(variance);

  return { mean, std, min, max, sum };
}

/**
 * Min-max scaler for normalizing feature inputs.
 * @param {number[]} values 
 * @returns {{scaled: number[], min: number, max: number}}
 */
export function minMaxScale(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const scaled = values.map(v => (v - min) / range);
  return { scaled, min, max };
}
