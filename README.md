
# AI-Driven Demand Forecasting & Inventory Optimization

## Overview

An AI-driven demand forecasting and inventory optimization application developed using Python and Streamlit. The system analyzes historical sales data to forecast future demand and provide inventory recommendations that support better stock management and reduce the risk of overstocking or stockouts.

## Problem Statement

Businesses often struggle to maintain optimal inventory levels due to fluctuating demand and inaccurate forecasting. Excess inventory increases holding costs, while insufficient stock can lead to lost sales and customer dissatisfaction.

## Proposed Solution

This project uses time-series forecasting and inventory management techniques to analyze historical sales data, predict future demand, and generate inventory recommendations.

## Features

* Historical sales data analysis
* Interactive sales dashboard
* Demand forecasting using Prophet
* Forecast visualization and results table
* Inventory optimization recommendations
* Safety stock calculation
* Reorder point estimation
* Recommended order quantity
* Stock sufficiency analysis
* Downloadable forecast results
* CSV dataset support

## Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Prophet

## Project Structure

```text
ai_demand_forecasting/
│
├── app_fixed.py
├── models.py
├── data_generator.py
├── requirements.txt
├── SETUP_GUIDE.txt
├── sales_data.csv
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SYEDA-RAFATHUR-RAHMAN/Demand-forecasting
cd ai_demand_forecasting
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app_fixed.py
```

The application will open in your browser.

## Dataset

The project includes a sample sales dataset (`sales_data.csv`). Users can also upload their own compatible sales CSV files.

## Expected Outcomes

* Improved demand forecasting
* Better inventory planning
* Reduced risk of stockouts and overstocking
* Data-driven replenishment decisions

## Project Information

**Project Type:** Final-Year Engineering Project
**Domain:** Data Analytics and Demand Forecasting
**Institution:** Ghousia College of Engineering
**University:** Visvesvaraya Technological University (VTU)

## Future Enhancements

* Integration with real-time sales data
* Advanced forecasting model comparison
* Automated inventory alerts
* Integration with business inventory systems

## Disclaimer

Forecasts and inventory recommendations are estimates based on historical data and the assumptions used by the application. Actual demand may vary.

