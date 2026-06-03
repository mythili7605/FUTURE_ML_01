/**
 * Data Processing, Preprocessing, Filtering, Aggregation, and Feature Engineering.
 */

/**
 * Parses CSV text, using PapaParse if available, or falling back to a custom parser.
 * @param {string} csvText 
 * @returns {Promise<any[]>} List of objects representing CSV rows
 */
export function parseCSV(csvText) {
  return new Promise((resolve) => {
    // Check if PapaParse is loaded in the global scope
    if (typeof window !== 'undefined' && window.Papa) {
      window.Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          resolve(results.data);
        },
        error: (err) => {
          console.error("PapaParse error, using fallback parser:", err);
          resolve(fallbackParseCSV(csvText));
        }
      });
    } else {
      resolve(fallbackParseCSV(csvText));
    }
  });
}

/**
 * Fallback CSV parser in pure JS when PapaParse is not loaded.
 * Handles quoted fields containing commas.
 */
function fallbackParseCSV(text) {
  const lines = text.split(/\r?\n/);
  if (lines.length === 0) return [];
  
  // Extract headers
  const headers = splitCSVLine(lines[0]);
  const data = [];
  
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const values = splitCSVLine(line);
    const row = {};
    
    for (let j = 0; j < headers.length; j++) {
      const headerName = headers[j];
      let val = values[j] !== undefined ? values[j] : null;
      
      // Basic type conversions
      if (val !== null) {
        if (!isNaN(val) && val.trim() !== '') {
          val = Number(val);
        } else if (val.toLowerCase() === 'true') {
          val = true;
        } else if (val.toLowerCase() === 'false') {
          val = false;
        }
      }
      row[headerName] = val;
    }
    data.push(row);
  }
  return data;
}

function splitCSVLine(line) {
  const result = [];
  let inQuotes = false;
  let currentVal = '';
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(currentVal.replace(/^"|"$/g, '').trim());
      currentVal = '';
    } else {
      currentVal += char;
    }
  }
  result.push(currentVal.replace(/^"|"$/g, '').trim());
  return result;
}

/**
 * Parses standard date formats, specifically handling DD/MM/YYYY.
 * @param {any} val 
 * @returns {Date|null}
 */
export function parseDate(val) {
  if (!val) return null;
  const str = String(val).trim();
  
  // Handle DD/MM/YYYY or DD-MM-YYYY format
  const dparts = str.split(/[-/]/);
  if (dparts.length === 3) {
    if (dparts[0].length === 4) {
      // YYYY-MM-DD
      return new Date(parseInt(dparts[0]), parseInt(dparts[1]) - 1, parseInt(dparts[2]));
    }
    // Assume DD/MM/YYYY
    const day = parseInt(dparts[0]);
    const month = parseInt(dparts[1]);
    const year = parseInt(dparts[2]);
    if (!isNaN(day) && !isNaN(month) && !isNaN(year)) {
      return new Date(year, month - 1, day);
    }
  }
  
  const parsed = new Date(str);
  return isNaN(parsed.getTime()) ? null : parsed;
}

/**
 * Maps CSV headers to target system fields case-insensitively.
 * @param {any[]} rawData 
 * @returns {{cleanedData: any[], missingStats: {nullSales: number, nullDates: number}}}
 */
export function preprocessData(rawData) {
  if (!rawData || rawData.length === 0) {
    return { cleanedData: [], missingStats: { nullSales: 0, nullDates: 0 } };
  }

  // Find column keys
  const sample = rawData[0];
  const keys = Object.keys(sample);
  
  const dateKey = keys.find(k => /date/i.test(k) && !/ship/i.test(k)) || keys.find(k => /date/i.test(k)) || 'Date';
  const salesKey = keys.find(k => /sales/i.test(k) || /revenue/i.test(k) || /amount/i.test(k)) || 'Sales';
  const categoryKey = keys.find(k => k.toLowerCase() === 'category') || 'Category';
  const subCategoryKey = keys.find(k => /sub-category/i.test(k) || /subcategory/i.test(k)) || 'Sub-Category';
  const regionKey = keys.find(k => /region/i.test(k)) || 'Region';
  const productKey = keys.find(k => /product.*name/i.test(k) || /product/i.test(k)) || 'Product Name';

  let nullSales = 0;
  let nullDates = 0;

  const cleanedData = [];

  for (let i = 0; i < rawData.length; i++) {
    const row = rawData[i];
    const rawSales = row[salesKey];
    const rawDate = row[dateKey];
    
    const date = parseDate(rawDate);
    const sales = typeof rawSales === 'number' ? rawSales : parseFloat(String(rawSales).replace(/[$,]/g, ''));

    if (date === null) {
      nullDates++;
      continue;
    }
    if (isNaN(sales) || sales === null || sales === undefined) {
      nullSales++;
      continue;
    }

    cleanedData.push({
      date,
      sales,
      category: row[categoryKey] ? String(row[categoryKey]).trim() : 'Unknown',
      subCategory: row[subCategoryKey] ? String(row[subCategoryKey]).trim() : 'Unknown',
      region: row[regionKey] ? String(row[regionKey]).trim() : 'Unknown',
      productName: row[productKey] ? String(row[productKey]).trim() : 'Unknown'
    });
  }

  // Sort chronologically
  cleanedData.sort((a, b) => a.date - b.date);

  return {
    cleanedData,
    missingStats: { nullSales, nullDates }
  };
}

/**
 * Filter and aggregate transaction data by day, week, or month.
 * @param {any[]} cleanTransactions 
 * @param {Object} filters - e.g. { category: 'Furniture', region: 'West' }
 * @param {'daily'|'weekly'|'monthly'} aggregationLevel 
 * @returns {any[]} Time-series items with { date: Date, dateStr: string, sales: number, month, day, dayofweek, year }
 */
export function aggregateData(cleanTransactions, filters = {}, aggregationLevel = 'daily') {
  // Apply filters
  let filtered = cleanTransactions;
  
  if (filters.category && filters.category !== 'All') {
    filtered = filtered.filter(item => item.category === filters.category);
  }
  if (filters.subCategory && filters.subCategory !== 'All') {
    filtered = filtered.filter(item => item.subCategory === filters.subCategory);
  }
  if (filters.region && filters.region !== 'All') {
    filtered = filtered.filter(item => item.region === filters.region);
  }

  if (filtered.length === 0) return [];

  // Grouping keys
  const groups = {};

  filtered.forEach(item => {
    let key;
    let dateVal;
    const d = item.date;

    if (aggregationLevel === 'monthly') {
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, '0');
      key = `${year}-${month}`;
      // Represent month with the first day of that month
      dateVal = new Date(year, d.getMonth(), 1);
    } else if (aggregationLevel === 'weekly') {
      // Find starting Sunday of the week
      const day = d.getDay();
      const diff = d.getDate() - day;
      const sunday = new Date(d.getFullYear(), d.getMonth(), diff);
      const year = sunday.getFullYear();
      const month = String(sunday.getMonth() + 1).padStart(2, '0');
      const dateNum = String(sunday.getDate()).padStart(2, '0');
      key = `${year}-${month}-${dateNum}`;
      dateVal = sunday;
    } else { // daily
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, '0');
      const dateNum = String(d.getDate()).padStart(2, '0');
      key = `${year}-${month}-${dateNum}`;
      dateVal = new Date(year, d.getMonth(), d.getDate());
    }

    if (!groups[key]) {
      groups[key] = {
        date: dateVal,
        dateStr: key,
        sales: 0
      };
    }
    groups[key].sales += item.sales;
  });

  // Convert to sorted list and engineer features
  const timeSeries = Object.values(groups).sort((a, b) => a.date - b.date);

  timeSeries.forEach(item => {
    const d = item.date;
    item.year = d.getFullYear();
    item.month = d.getMonth() + 1;
    item.day = d.getDate();
    item.dayofweek = d.getDay();
  });

  return timeSeries;
}

/**
 * Splits sequential timeseries into train and test sets.
 * @param {any[]} timeSeries 
 * @param {number} trainRatio - e.g. 0.8 
 * @returns {{train: any[], test: any[]}}
 */
export function splitTrainTest(timeSeries, trainRatio = 0.8) {
  const trainSize = Math.floor(timeSeries.length * trainRatio);
  return {
    train: timeSeries.slice(0, trainSize),
    test: timeSeries.slice(trainSize)
  };
}

/**
 * Generates future date items with engineered features for forecasting.
 * @param {Date} lastDate 
 * @param {number} periods - number of steps (days/weeks/months)
 * @param {'daily'|'weekly'|'monthly'} aggregationLevel 
 * @returns {any[]} List of feature objects for forecasting period
 */
export function generateFutureDates(lastDate, periods, aggregationLevel) {
  const futureItems = [];
  let currentDate = new Date(lastDate);

  for (let i = 1; i <= periods; i++) {
    let nextDate;
    if (aggregationLevel === 'monthly') {
      nextDate = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1);
    } else if (aggregationLevel === 'weekly') {
      nextDate = new Date(currentDate.getTime() + 7 * 24 * 60 * 60 * 1000);
    } else { // daily
      nextDate = new Date(currentDate.getTime() + 24 * 60 * 60 * 1000);
    }

    const year = nextDate.getFullYear();
    const month = nextDate.getMonth() + 1;
    const day = nextDate.getDate();
    const dayofweek = nextDate.getDay();

    let dateStr;
    if (aggregationLevel === 'monthly') {
      dateStr = `${year}-${String(month).padStart(2, '0')}`;
    } else {
      dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    }

    futureItems.push({
      date: nextDate,
      dateStr,
      year,
      month,
      day,
      dayofweek
    });
    currentDate = nextDate;
  }

  return futureItems;
}
