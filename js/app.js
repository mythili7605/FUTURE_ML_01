import { parseCSV, preprocessData, aggregateData, splitTrainTest, generateFutureDates } from './dataProcessor.js';
import { calculateMAE, calculateRMSE, calculateMAPE, getStats } from './mathUtils.js';
import { MovingAverageModel, LinearRegressionModel, HoltWintersModel, DecisionTreeRegressor } from './models.js';

// Application State
let rawTransactions = [];
let preprocessedTransactions = [];
let aggregatedSeries = [];
let trainSeries = [];
let testSeries = [];
let charts = {}; // references to Chart.js charts
let trainedModels = {};
let testPredictions = {};
let futureForecasts = {};
let selectedModelName = 'HoltWinters';

// DOM Selectors
const loadPreloadedBtn = document.getElementById('load-preloaded-btn');
const welcomeLoadPreloadedBtn = document.getElementById('welcome-load-preloaded');
const uploadCsvBtn = document.getElementById('upload-csv-btn');
const csvFileInput = document.getElementById('csv-file-input');
const dragDropZone = document.getElementById('drag-drop-zone');
const trainModelsBtn = document.getElementById('train-models-btn');
const exportCsvBtn = document.getElementById('export-csv-btn');

// Filters and Configuration Selectors
const filterCategory = document.getElementById('filter-category');
const filterSubcategory = document.getElementById('filter-subcategory');
const filterRegion = document.getElementById('filter-region');
const aggregationLevel = document.getElementById('aggregation-level');
const trainSplit = document.getElementById('train-split');
const forecastPeriods = document.getElementById('forecast-periods');

// Sliders and Value Indicators
const lrSlider = document.getElementById('lr-slider');
const lrVal = document.getElementById('lr-val');
const epochsSlider = document.getElementById('epochs-slider');
const epochsVal = document.getElementById('epochs-val');
const alphaSlider = document.getElementById('alpha-slider');
const alphaVal = document.getElementById('alpha-val');
const betaSlider = document.getElementById('beta-slider');
const betaVal = document.getElementById('beta-val');
const gammaSlider = document.getElementById('gamma-slider');
const gammaVal = document.getElementById('gamma-val');
const depthSlider = document.getElementById('depth-slider');
const depthVal = document.getElementById('depth-val');

// Tabs
const tabLinks = document.querySelectorAll('.tab-link');
const tabPanes = document.querySelectorAll('.tab-pane');

// Loader Overlay
const globalLoader = document.getElementById('global-loader');
const loaderTitle = document.getElementById('loader-title');
const loaderSubtitle = document.getElementById('loader-subtitle');

// Status bar
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// KPI fields
const kpiTotalSales = document.getElementById('kpi-total-sales');
const kpiTotalOrders = document.getElementById('kpi-total-orders');
const kpiTimeframe = document.getElementById('kpi-timeframe');
const kpiMissingVals = document.getElementById('kpi-missing-vals');

// Forecast KPI fields
const forecastTotalKpi = document.getElementById('forecast-total-kpi');
const forecastBestModelKpi = document.getElementById('forecast-best-model-kpi');
const forecastHorizonKpi = document.getElementById('forecast-horizon-kpi');

// Table Bodies
const metricsTableBody = document.getElementById('metrics-table-body');
const forecastTableBody = document.getElementById('forecast-table-body');

// ----------------------------------------------------------------------------
// INITIALIZE PAGE
// ----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  initSliders();
  lucide.createIcons(); // render vector icons
});

function setupEventListeners() {
  // Navigation Tabs switching
  tabLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      const targetId = link.getAttribute('data-target');
      
      tabLinks.forEach(l => l.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      
      link.classList.add('active');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Load Preloaded dataset
  loadPreloadedBtn.addEventListener('click', () => loadPreloadedDataset());
  welcomeLoadPreloadedBtn.addEventListener('click', () => loadPreloadedDataset());

  // Upload custom dataset
  uploadCsvBtn.addEventListener('click', () => csvFileInput.click());
  csvFileInput.addEventListener('change', handleFileSelect);

  // Drag and Drop files
  dragDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dragDropZone.classList.add('dragover');
  });
  dragDropZone.addEventListener('dragleave', () => {
    dragDropZone.classList.remove('dragover');
  });
  dragDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.csv')) {
        readFile(file);
      } else {
        alert("Please drop a valid .csv file.");
      }
    }
  });

  // Filter Changes
  filterCategory.addEventListener('change', () => {
    populateSubCategories();
    refreshAggregation();
  });
  filterSubcategory.addEventListener('change', refreshAggregation);
  filterRegion.addEventListener('change', refreshAggregation);
  aggregationLevel.addEventListener('change', () => {
    // Auto adjust forecast period default if level changes
    const level = aggregationLevel.value;
    if (level === 'monthly') forecastPeriods.value = 12;
    else if (level === 'weekly') forecastPeriods.value = 8;
    else forecastPeriods.value = 30;
    
    refreshAggregation();
  });

  // Train Button
  trainModelsBtn.addEventListener('click', handleTrainAndForecast);

  // Export CSV
  exportCsvBtn.addEventListener('click', handleExportCSV);
}

function initSliders() {
  // Bind inputs to their label values
  lrSlider.addEventListener('input', () => lrVal.textContent = lrSlider.value);
  epochsSlider.addEventListener('input', () => epochsVal.textContent = epochsSlider.value);
  alphaSlider.addEventListener('input', () => alphaVal.textContent = alphaSlider.value);
  betaSlider.addEventListener('input', () => betaVal.textContent = betaSlider.value);
  gammaSlider.addEventListener('input', () => gammaVal.textContent = gammaSlider.value);
  depthSlider.addEventListener('input', () => depthVal.textContent = depthSlider.value);
}

// ----------------------------------------------------------------------------
// DATA LOADER PROCESSORS
// ----------------------------------------------------------------------------
function showLoader(title, subtitle) {
  loaderTitle.textContent = title;
  loaderSubtitle.textContent = subtitle;
  globalLoader.classList.add('active');
}

function hideLoader() {
  globalLoader.classList.remove('active');
}

async function loadPreloadedDataset() {
  showLoader("Fetching Preloaded Data", "Retrieving train (1).csv from workspace...");
  try {
    const response = await fetch('/train (1).csv');
    if (!response.ok) {
      throw new Error(`Failed to retrieve file, status ${response.status}`);
    }
    const text = await response.text();
    processCSVText(text, "train (1).csv");
  } catch (err) {
    hideLoader();
    alert("Could not load local dataset directly via network fetch (this is normal when running files directly via file:// schema due to browser security restrictions). Please click 'Upload Custom CSV' and choose the 'train (1).csv' file from the desktop folder.");
    console.error(err);
  }
}

function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    readFile(e.target.files[0]);
  }
}

function readFile(file) {
  showLoader("Loading File", `Reading ${file.name}...`);
  const reader = new FileReader();
  reader.onload = function(evt) {
    processCSVText(evt.target.result, file.name);
  };
  reader.readAsText(file);
}

async function processCSVText(csvText, filename) {
  loaderTitle.textContent = "Parsing CSV File";
  loaderSubtitle.textContent = "Formatting cells...";
  
  try {
    rawTransactions = await parseCSV(csvText);
    if (rawTransactions.length === 0) {
      throw new Error("Parsed dataset is empty.");
    }
    
    loaderTitle.textContent = "Data Preprocessing";
    loaderSubtitle.textContent = "Cleaning and sorting data entries...";
    
    const prep = preprocessData(rawTransactions);
    preprocessedTransactions = prep.cleanedData;
    
    // Set Status
    statusDot.classList.add('loaded');
    statusText.textContent = `${filename} (${preprocessedTransactions.length.toLocaleString()} rows)`;
    
    // KPIs
    const totalSales = preprocessedTransactions.reduce((sum, item) => sum + item.sales, 0);
    kpiTotalSales.textContent = `$${totalSales.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    kpiTotalOrders.textContent = preprocessedTransactions.length.toLocaleString();
    
    const minDate = preprocessedTransactions[0].date.toLocaleDateString();
    const maxDate = preprocessedTransactions[preprocessedTransactions.length - 1].date.toLocaleDateString();
    kpiTimeframe.textContent = `${minDate} - ${maxDate}`;
    kpiMissingVals.textContent = `${prep.missingStats.nullSales + prep.missingStats.nullDates}`;

    // Enable Sidebar Controls
    filterCategory.disabled = false;
    filterSubcategory.disabled = false;
    filterRegion.disabled = false;
    trainModelsBtn.disabled = false;

    // Enable navigation tabs
    tabLinks.forEach(link => link.disabled = false);

    // Build filter option dropdown lists
    populateFilterDropdowns();

    // Trigger EDA aggregation and view switch
    refreshAggregation();
    
    // Switch to Overview Pane
    hideLoader();
    switchTab('pane-overview');

  } catch (error) {
    hideLoader();
    alert(`CSV Processing failed: ${error.message}`);
    console.error(error);
  }
}

function switchTab(targetPaneId) {
  tabLinks.forEach(link => {
    const target = link.getAttribute('data-target');
    if (target === targetPaneId) {
      link.classList.add('active');
      link.disabled = false;
    } else {
      link.classList.remove('active');
    }
  });

  tabPanes.forEach(pane => {
    if (pane.id === targetPaneId) {
      pane.classList.add('active');
    } else {
      pane.classList.remove('active');
    }
  });
}

function populateFilterDropdowns() {
  const categories = new Set();
  const regions = new Set();
  
  preprocessedTransactions.forEach(item => {
    categories.add(item.category);
    regions.add(item.region);
  });

  // Populate Categories
  filterCategory.innerHTML = '<option value="All">All Categories</option>';
  [...categories].sort().forEach(cat => {
    filterCategory.innerHTML += `<option value="${cat}">${cat}</option>`;
  });

  // Populate Regions
  filterRegion.innerHTML = '<option value="All">All Regions</option>';
  [...regions].sort().forEach(reg => {
    filterRegion.innerHTML += `<option value="${reg}">${reg}</option>`;
  });

  populateSubCategories();
}

function populateSubCategories() {
  const catFilter = filterCategory.value;
  const subCategories = new Set();
  
  preprocessedTransactions.forEach(item => {
    if (catFilter === 'All' || item.category === catFilter) {
      subCategories.add(item.subCategory);
    }
  });

  filterSubcategory.innerHTML = '<option value="All">All Sub-Categories</option>';
  [...subCategories].sort().forEach(sub => {
    filterSubcategory.innerHTML += `<option value="${sub}">${sub}</option>`;
  });
}

// ----------------------------------------------------------------------------
// DATA AGGREGATION & EDA GRAPHING
// ----------------------------------------------------------------------------
function refreshAggregation() {
  const filters = {
    category: filterCategory.value,
    subCategory: filterSubcategory.value,
    region: filterRegion.value
  };
  const level = aggregationLevel.value;

  aggregatedSeries = aggregateData(preprocessedTransactions, filters, level);
  
  updateOverviewFilterBadges(filters);
  renderHistoricalChart();
  renderDistributionCharts();
}

function updateOverviewFilterBadges(filters) {
  const badgeContainer = document.getElementById('overview-filter-badges');
  badgeContainer.innerHTML = '';
  
  Object.entries(filters).forEach(([key, val]) => {
    if (val && val !== 'All') {
      const badge = document.createElement('span');
      badge.className = 'filter-tag';
      badge.innerHTML = `
        <strong>${key.charAt(0).toUpperCase() + key.slice(1)}:</strong> ${val}
        <span class="remove-btn" data-key="filter-${key.toLowerCase()}">&times;</span>
      `;
      badgeContainer.appendChild(badge);
    }
  });

  // Bind clear buttons on badges
  badgeContainer.querySelectorAll('.remove-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const selectorId = btn.getAttribute('data-key');
      const el = document.getElementById(selectorId);
      if (el) {
        el.value = 'All';
        if (selectorId === 'filter-category') {
          populateSubCategories();
        }
        refreshAggregation();
      }
    });
  });
}

function destroyChart(name) {
  if (charts[name]) {
    charts[name].destroy();
    delete charts[name];
  }
}

function renderHistoricalChart() {
  destroyChart('historical');
  
  const ctx = document.getElementById('chart-historical').getContext('2d');
  
  const labels = aggregatedSeries.map(item => item.dateStr);
  const data = aggregatedSeries.map(item => item.sales);

  charts['historical'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Sales Revenue ($)',
        data: data,
        borderColor: '#7c3aed',
        backgroundColor: 'rgba(124, 58, 237, 0.05)',
        borderWidth: 2,
        fill: true,
        tension: 0.15,
        pointRadius: labels.length > 100 ? 0 : 3,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af', font: { size: 10 } }
        }
      }
    }
  });
}

function renderDistributionCharts() {
  // Category distribution
  destroyChart('category');
  
  const catSums = {};
  const regSums = {};
  
  preprocessedTransactions.forEach(item => {
    catSums[item.category] = (catSums[item.category] || 0) + item.sales;
    regSums[item.region] = (regSums[item.region] || 0) + item.sales;
  });

  const catCtx = document.getElementById('chart-category').getContext('2d');
  charts['category'] = new Chart(catCtx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(catSums),
      datasets: [{
        data: Object.values(catSums),
        backgroundColor: ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'],
        borderWidth: 1,
        borderColor: 'rgba(3, 7, 18, 0.5)'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#f3f4f6', font: { size: 10 } }
        }
      }
    }
  });

  // Region distribution
  destroyChart('region');
  const regCtx = document.getElementById('chart-region').getContext('2d');
  charts['region'] = new Chart(regCtx, {
    type: 'bar',
    data: {
      labels: Object.keys(regSums),
      datasets: [{
        label: 'Sales Revenue ($)',
        data: Object.values(regSums),
        backgroundColor: 'rgba(6, 182, 212, 0.65)',
        borderColor: '#06b6d4',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af' }
        }
      }
    }
  });
}

// ----------------------------------------------------------------------------
// MACHINE LEARNING MODEL TRAINING & VALIDATION
// ----------------------------------------------------------------------------
function handleTrainAndForecast() {
  if (aggregatedSeries.length < 15) {
    alert("Insufficient data points for forecasting. Please adjust aggregation level to daily/weekly, or clear your filters to increase dataset size.");
    return;
  }

  showLoader("Training ML Models", "Running models on historical data...");

  // Delay execution slightly so the UI loader shows up
  setTimeout(() => {
    try {
      const splitRatio = parseFloat(trainSplit.value);
      const split = splitTrainTest(aggregatedSeries, splitRatio);
      trainSeries = split.train;
      testSeries = split.test;

      const level = aggregationLevel.value;
      let period = 7; // daily
      if (level === 'monthly') period = 12;
      else if (level === 'weekly') period = 4; // seasonal cycle of 4 weeks in a month

      const horizon = parseInt(forecastPeriods.value) || 12;

      // Extract raw target values
      const yTrain = trainSeries.map(item => item.sales);
      const yTest = testSeries.map(item => item.sales);

      // Model 1: Baseline Moving Average
      const maModel = new MovingAverageModel(level === 'monthly' ? 3 : 7);
      maModel.fit(trainSeries, yTrain);
      const maPreds = maModel.predict(testSeries);

      // Model 2: Multiple Linear Regression (Gradient Descent)
      const lrLr = parseFloat(lrSlider.value);
      const lrEpochs = parseInt(epochsSlider.value);
      const lrModel = new LinearRegressionModel(lrLr, lrEpochs);
      lrModel.fit(trainSeries, yTrain);
      const lrPreds = lrModel.predict(testSeries, trainSeries.length);

      // Model 3: Holt-Winters (Triple Smoothing)
      const hwA = parseFloat(alphaSlider.value);
      const hwB = parseFloat(betaSlider.value);
      const hwG = parseFloat(gammaSlider.value);
      const hwModel = new HoltWintersModel(hwA, hwB, hwG, period);
      hwModel.fit(trainSeries, yTrain);
      const hwPreds = hwModel.predict(testSeries);

      // Model 4: Decision Tree Regressor
      const dtDepth = parseInt(depthSlider.value);
      const dtModel = new DecisionTreeRegressor(dtDepth, 5);
      dtModel.fit(trainSeries, yTrain);
      const dtPreds = dtModel.predict(testSeries, trainSeries.length);

      // Save predictions
      testPredictions = {
        'MovingAverage': maPreds,
        'LinearRegression': lrPreds,
        'HoltWinters': hwPreds,
        'DecisionTree': dtPreds
      };

      // Compute Evaluation Metrics
      const metrics = {};
      Object.keys(testPredictions).forEach(name => {
        metrics[name] = {
          mae: calculateMAE(yTest, testPredictions[name]),
          rmse: calculateRMSE(yTest, testPredictions[name]),
          mape: calculateMAPE(yTest, testPredictions[name])
        };
      });

      // Find Best Model (lowest MAE)
      let bestModel = 'HoltWinters';
      let minMae = Infinity;
      Object.keys(metrics).forEach(name => {
        if (metrics[name].mae < minMae) {
          minMae = metrics[name].mae;
          bestModel = name;
        }
      });
      selectedModelName = bestModel;

      // Fit models on full dataset for final future forecasting
      const yAll = aggregatedSeries.map(item => item.sales);
      
      const maFinal = new MovingAverageModel(level === 'monthly' ? 3 : 7);
      maFinal.fit(aggregatedSeries, yAll);

      const lrFinal = new LinearRegressionModel(lrLr, lrEpochs);
      lrFinal.fit(aggregatedSeries, yAll);

      const hwFinal = new HoltWintersModel(hwA, hwB, hwG, period);
      hwFinal.fit(aggregatedSeries, yAll);

      const dtFinal = new DecisionTreeRegressor(dtDepth, 5);
      dtFinal.fit(aggregatedSeries, yAll);

      // Generate future feature indexes
      const lastDate = aggregatedSeries[aggregatedSeries.length - 1].date;
      const futureDates = generateFutureDates(lastDate, horizon, level);

      // Predict future
      futureForecasts = {
        'MovingAverage': maFinal.predict(futureDates),
        'LinearRegression': lrFinal.predict(futureDates, aggregatedSeries.length),
        'HoltWinters': hwFinal.predict(futureDates),
        'DecisionTree': dtFinal.predict(futureDates, aggregatedSeries.length),
        'dates': futureDates
      };

      // Display evaluation metrics in UI
      renderMetricsTable(metrics, bestModel);

      // Plot LR Loss curve
      renderLossChart(lrModel.lossHistory);

      // Plot Validation Predictions overlap
      renderValidationChart(testSeries.map(item => item.dateStr), yTest);

      // Update Forecast KPIs & Plot Projections
      renderForecastView(bestModel, horizon);

      // Render Seasonality charts
      renderSeasonalityAnalysisCharts();

      hideLoader();
      switchTab('pane-trainer');

    } catch (err) {
      hideLoader();
      alert(`Model training error: ${err.message}`);
      console.error(err);
    }
  }, 100);
}

function renderMetricsTable(metrics, bestModelName) {
  metricsTableBody.innerHTML = '';
  
  const niceNames = {
    'MovingAverage': 'Simple Moving Average (Baseline)',
    'LinearRegression': 'Multiple Linear Regression',
    'HoltWinters': 'Holt-Winters Exponential Smoothing',
    'DecisionTree': 'Decision Tree Regressor'
  };

  Object.entries(metrics).forEach(([name, vals]) => {
    const isBest = name === bestModelName;
    const highlightClass = isBest ? 'class="best-metric-highlight"' : '';
    const badge = isBest ? '<span class="status-badge success">Best Model</span>' : '<span class="status-badge warning">Fit</span>';
    
    const row = document.createElement('tr');
    if (isBest) row.style.background = 'rgba(16, 185, 129, 0.04)';
    
    row.innerHTML = `
      <td ${isBest ? 'style="font-weight:600; color:var(--accent);"' : ''}>${niceNames[name]}</td>
      <td ${highlightClass}>${vals.mae.toFixed(2)}</td>
      <td>${vals.rmse.toFixed(2)}</td>
      <td>${vals.mape.toFixed(2)}%</td>
      <td>${badge}</td>
    `;
    metricsTableBody.appendChild(row);
  });
}

function renderLossChart(lossHistory) {
  destroyChart('loss');
  const ctx = document.getElementById('chart-training-loss').getContext('2d');
  
  // Downsample if loss history is very large to keep chart fast
  let labels = lossHistory.map((_, i) => i + 1);
  let data = lossHistory;
  if (lossHistory.length > 200) {
    const step = Math.ceil(lossHistory.length / 100);
    labels = labels.filter((_, i) => i % step === 0);
    data = data.filter((_, i) => i % step === 0);
  }

  charts['loss'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean Squared Error (MSE)',
        data: data,
        borderColor: '#a78bfa',
        borderWidth: 2,
        fill: false,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: 'Epochs', color: '#9ca3af' },
          grid: { display: false },
          ticks: { color: '#9ca3af' }
        },
        y: {
          title: { display: true, text: 'Loss', color: '#9ca3af' },
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af' }
        }
      }
    }
  });
}

function renderValidationChart(dates, actuals) {
  destroyChart('validation');
  const ctx = document.getElementById('chart-validation-predictions').getContext('2d');

  charts['validation'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Actual Sales',
          data: actuals,
          borderColor: '#f3f4f6',
          borderWidth: 2.5,
          pointRadius: 3,
          fill: false
        },
        {
          label: 'Holt-Winters',
          data: testPredictions['HoltWinters'],
          borderColor: '#10b981',
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [3, 3],
          fill: false
        },
        {
          label: 'Linear Regression',
          data: testPredictions['LinearRegression'],
          borderColor: '#3b82f6',
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [3, 3],
          fill: false
        },
        {
          label: 'Decision Tree',
          data: testPredictions['DecisionTree'],
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [3, 3],
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af' }
        }
      }
    }
  });
}

// ----------------------------------------------------------------------------
// FORECAST PROJECTION VIEWS
// ----------------------------------------------------------------------------
function renderForecastView(bestModelName, horizon) {
  const futureDates = futureForecasts.dates;
  const bestForecast = futureForecasts[bestModelName];
  
  // Forecast KPIs
  const totalForecastedSales = bestForecast.reduce((sum, v) => sum + v, 0);
  forecastTotalKpi.textContent = `$${totalForecastedSales.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  
  const niceNames = {
    'MovingAverage': 'Moving Average',
    'LinearRegression': 'Linear Regression',
    'HoltWinters': 'Holt-Winters',
    'DecisionTree': 'Decision Tree'
  };
  forecastBestModelKpi.textContent = niceNames[bestModelName];
  forecastHorizonKpi.textContent = `${horizon} ${aggregationLevel.value === 'monthly' ? 'Months' : aggregationLevel.value === 'weekly' ? 'Weeks' : 'Days'}`;

  // 1. Chart Projections
  destroyChart('forecast');
  const ctx = document.getElementById('chart-forecast').getContext('2d');
  
  // Merge labels and datasets
  // Show last 30 historical points to keep graph uncluttered, plus future points
  const showHistPoints = Math.min(aggregatedSeries.length, 40);
  const historicalSubset = aggregatedSeries.slice(aggregatedSeries.length - showHistPoints);
  
  const labels = [
    ...historicalSubset.map(item => item.dateStr),
    ...futureDates.map(item => item.dateStr)
  ];

  // Align datasets
  const actualsData = [...historicalSubset.map(item => item.sales)];
  // Future predictions fill the remainder
  const futureBestData = Array(historicalSubset.length - 1).fill(null);
  // overlap final actual element for gap-free line drawing
  futureBestData.push(historicalSubset[historicalSubset.length - 1].sales);
  futureBestData.push(...bestForecast);

  const futureLrData = Array(historicalSubset.length - 1).fill(null);
  futureLrData.push(historicalSubset[historicalSubset.length - 1].sales);
  futureLrData.push(...futureForecasts['LinearRegression']);

  const futureHwData = Array(historicalSubset.length - 1).fill(null);
  futureHwData.push(historicalSubset[historicalSubset.length - 1].sales);
  futureHwData.push(...futureForecasts['HoltWinters']);

  const futureDtData = Array(historicalSubset.length - 1).fill(null);
  futureDtData.push(historicalSubset[historicalSubset.length - 1].sales);
  futureDtData.push(...futureForecasts['DecisionTree']);

  charts['forecast'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Historical Sales',
          data: actualsData,
          borderColor: '#7c3aed',
          backgroundColor: 'rgba(124, 58, 237, 0.05)',
          borderWidth: 2,
          fill: true,
          pointRadius: historicalSubset.length > 50 ? 0 : 3
        },
        {
          label: `${niceNames[bestModelName]} Forecast (Best)`,
          data: futureBestData,
          borderColor: '#10b981',
          borderWidth: 3,
          borderDash: [5, 5],
          backgroundColor: 'rgba(16, 185, 129, 0.05)',
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 8
        },
        {
          label: 'Linear Reg Forecast',
          data: futureLrData,
          borderColor: 'rgba(59, 130, 246, 0.4)',
          borderWidth: 1.5,
          borderDash: [2, 2],
          fill: false,
          pointRadius: 0
        },
        {
          label: 'Holt-Winters Forecast',
          data: futureHwData,
          borderColor: 'rgba(16, 185, 129, 0.4)',
          borderWidth: 1.5,
          borderDash: [2, 2],
          fill: false,
          pointRadius: 0
        },
        {
          label: 'Decision Tree Forecast',
          data: futureDtData,
          borderColor: 'rgba(245, 158, 11, 0.4)',
          borderWidth: 1.5,
          borderDash: [2, 2],
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.03)' },
          ticks: { color: '#9ca3af' }
        }
      }
    }
  });

  // 2. Forecast breakdown table
  forecastTableBody.innerHTML = '';
  for (let i = 0; i < futureDates.length; i++) {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td style="font-weight: 500;">${futureDates[i].dateStr}</td>
      <td style="color: var(--accent); font-weight: 600;">$${bestForecast[i].toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
      <td>$${futureForecasts['LinearRegression'][i].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
      <td>$${futureForecasts['HoltWinters'][i].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
      <td>$${futureForecasts['DecisionTree'][i].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
    `;
    forecastTableBody.appendChild(row);
  }
}

// ----------------------------------------------------------------------------
// FEATURE VISUALIZATION & SEASONALITY ANALYSIS
// ----------------------------------------------------------------------------
function renderSeasonalityAnalysisCharts() {
  destroyChart('monthSeasonality');
  destroyChart('dayofweekSeasonality');

  // Month Seasonality: Calculate average sales per month (1-12)
  const monthSums = Array(12).fill(0);
  const monthCounts = Array(12).fill(0);
  
  aggregatedSeries.forEach(item => {
    const mIdx = item.month - 1;
    monthSums[mIdx] += item.sales;
    monthCounts[mIdx] += 1;
  });

  const monthAverages = monthSums.map((sum, i) => monthCounts[i] > 0 ? sum / monthCounts[i] : 0);
  const monthLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  const monthCtx = document.getElementById('chart-month-seasonality').getContext('2d');
  charts['monthSeasonality'] = new Chart(monthCtx, {
    type: 'bar',
    data: {
      labels: monthLabels,
      datasets: [{
        label: 'Average Sales ($)',
        data: monthAverages,
        backgroundColor: 'rgba(124, 58, 237, 0.75)',
        borderColor: '#7c3aed',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#9ca3af' } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.03)' }, ticks: { color: '#9ca3af' } }
      }
    }
  });

  // Day of Week Seasonality: Calculate average sales per day (Sun-Sat)
  const dowSums = Array(7).fill(0);
  const dowCounts = Array(7).fill(0);

  aggregatedSeries.forEach(item => {
    const dIdx = item.dayofweek;
    dowSums[dIdx] += item.sales;
    dowCounts[dIdx] += 1;
  });

  const dowAverages = dowSums.map((sum, i) => dowCounts[i] > 0 ? sum / dowCounts[i] : 0);
  const dowLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  const dowCtx = document.getElementById('chart-dayofweek-seasonality').getContext('2d');
  charts['dayofweekSeasonality'] = new Chart(dowCtx, {
    type: 'bar',
    data: {
      labels: dowLabels,
      datasets: [{
        label: 'Average Sales ($)',
        data: dowAverages,
        backgroundColor: 'rgba(6, 182, 212, 0.75)',
        borderColor: '#06b6d4',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#9ca3af' } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.03)' }, ticks: { color: '#9ca3af' } }
      }
    }
  });
}

// ----------------------------------------------------------------------------
// EXPORT PREDICTIONS TO CSV FILE
// ----------------------------------------------------------------------------
function handleExportCSV() {
  if (!futureForecasts.dates || futureForecasts.dates.length === 0) {
    alert("No forecast predictions available to export.");
    return;
  }

  const futureDates = futureForecasts.dates;
  const maFore = futureForecasts['MovingAverage'];
  const lrFore = futureForecasts['LinearRegression'];
  const hwFore = futureForecasts['HoltWinters'];
  const dtFore = futureForecasts['DecisionTree'];

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Date,Moving Average,Linear Regression,Holt Winters,Decision Tree\n";

  for (let i = 0; i < futureDates.length; i++) {
    const row = [
      futureDates[i].dateStr,
      maFore[i].toFixed(4),
      lrFore[i].toFixed(4),
      hwFore[i].toFixed(4),
      dtFore[i].toFixed(4)
    ].join(",");
    csvContent += row + "\n";
  }

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `sales_forecast_${aggregationLevel.value}.csv`);
  document.body.appendChild(link);
  
  link.click();
  document.body.removeChild(link);
}
