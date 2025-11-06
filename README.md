# 🌫️ Air Quality Index (AQI) Forecasting System

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](YOUR_STREAMLIT_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-green?style=for-the-badge)](https://scikit-learn.org/)

> **Live Demo:** 🔗 [View Deployed Dashboard]([YOUR_STREAMLIT_LINK_HERE](https://aqi-forecast-pipeline-4pid7ezwjm4tpfhnszbbsp.streamlit.app/))

An end-to-end machine learning system for predicting Air Quality Index (AQI) in Islamabad, Pakistan, featuring automated data pipelines, model training, and an interactive dashboard for 3-day forecasts.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning](#-machine-learning-models)
- [Feature Engineering](#-feature-engineering)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dashboard](#-dashboard-features)
- [Model Performance](#-model-performance)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Air pollution is a critical environmental concern affecting millions globally. This project implements a production-ready AQI forecasting system that predicts air quality three days in advance, enabling proactive health measures and policy decisions.

### Problem Statement

Urban areas in South Asia, particularly Pakistan, face severe air quality challenges with AQI levels frequently exceeding safe thresholds. Real-time forecasting helps:

- **Public Health**: Early warnings for vulnerable populations (children, elderly, asthma patients)
- **Policy Making**: Data-driven decisions for traffic control and industrial regulations
- **Urban Planning**: Long-term environmental strategy development
- **Individual Safety**: Personal activity planning based on predicted air quality

### Solution Approach

This system employs a complete MLOps pipeline integrating:

1. **Automated Data Collection**: Hourly fetching from Open-Meteo APIs covering air quality and meteorological parameters
2. **Feature Store Architecture**: Parquet-based storage for efficient feature management and versioning
3. **Machine Learning Pipeline**: Comparative training of multiple regression models (Random Forest, Ridge Regression)
4. **Production Deployment**: Interactive Streamlit dashboard with real-time predictions and health recommendations

---

## ✨ Key Features

### 🔄 **Automated Data Pipeline**
- Continuous hourly data ingestion from Open-Meteo Air Quality API
- Historical data coverage spanning 10 months (January 2025 - November 2025)
- Automated handling of missing values and data quality checks
- Support for multiple pollutants: PM2.5, PM10, O₃, NO₂, SO₂, CO
- Weather integration: Temperature, humidity, wind speed, precipitation

### 🧠 **Intelligent Forecasting**
- Multi-day prediction capability (1-3 days ahead)
- Ensemble learning approach comparing Random Forest and Ridge Regression
- Feature importance analysis revealing key pollution drivers
- Uncertainty quantification through model performance metrics

### 🎨 **Interactive Dashboard**
- Real-time 3-day AQI forecasts with health category classifications
- Color-coded pollutant status indicators based on WHO guidelines
- Historical trend visualization with comparative analysis
- Environmental insights tracking weekly changes in air quality metrics
- Personalized health recommendations based on forecast severity

### 📊 **Feature Store Integration**
- Parquet-based columnar storage for efficient data retrieval
- Automated feature computation including time-based patterns
- Support for feature versioning and rollback capabilities
- Scalable architecture supporting future feature additions

### 🔬 **Comprehensive Monitoring**
- Model performance tracking (RMSE, MAE, R²)
- Overfitting detection through train-test gap analysis
- Feature drift monitoring capabilities
- Automated logging of predictions and actuals

---

## 🏗️ System Architecture

### High-Level Architecture

The system follows a modular MLOps architecture with clear separation of concerns:

```
┌─────────────────┐
│  Data Sources   │
│  (Open-Meteo)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion │  ◄── Scheduled hourly
│   (fetch_api)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │  ◄── AQI calculation + time features
│ (compute_feat)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Store   │  ◄── Parquet tables (partitioned)
│   (Parquet)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │  ◄── RF + Ridge comparison
│    (train.py)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Storage  │  ◄── Joblib serialization
│  (best_model)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Dashboard     │  ◄── Streamlit web app
│  (Predictions)  │
└─────────────────┘
```

### Data Flow

1. **Collection Phase**: Hourly API calls retrieve pollutant concentrations and meteorological data
2. **Processing Phase**: EPA AQI calculation, feature engineering (lags, rolling averages)
3. **Storage Phase**: Features stored in partitioned Parquet format for efficient querying
4. **Training Phase**: Periodic model retraining using latest feature data
5. **Inference Phase**: Real-time predictions using latest weather patterns and pollution trends
6. **Visualization Phase**: Dashboard renders forecasts with contextual health information

---

## 📂 Project Structure

```
aqi-forecast-project/
│
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (API keys, config)
│
├── data/                             # Data storage hierarchy
│   ├── raw/                          # Raw API responses (CSV format)
│   │   └── openmeteo_combined_islamabad_*.csv
│   │
│   ├── processed/                    # Engineered features with AQI
│   │   └── processed_aqi.csv
│   │
│   └── feature_store/                # Parquet-based feature store
│       └── parquet/
│           └── aqi_features/         # Partitioned by date
│
├── models/                           # Trained model artifacts
│   ├── best_model.pkl               # Production model (Joblib format)
│   └── metrics.txt                  # Model evaluation metrics
│
├── src/                              # Source code modules
│   │
│   ├── ingestion/                    # Data collection pipeline
│   │   ├── fetch_api.py             # Open-Meteo API client
│   │   └── scheduler_trigger.md     # Automation setup guide
│   │
│   ├── features/                     # Feature engineering
│   │   ├── compute_features.py      # AQI calculation + time features
│   │   └── backfill.py              # Historical data processing
│   │
│   ├── feature_store/                # Feature storage layer
│   │   ├── parquet_store.py         # Parquet read/write operations
│   │   └── build_parquet_store.py   # Feature table creation
│   │
│   ├── models/                       # ML model definitions
│   │   ├── rf_model.py              # Random Forest wrapper
│   │   └── ridge_model.py           # Ridge Regression wrapper
│   │
│   ├── training/                     # Model training pipeline
│   │   ├── train.py                 # Main training orchestrator
│   │   └── experiments/             # Training logs & metrics
│   │
│   ├── app/                          # Web application
│   │   └── streamlit_app.py         # Interactive dashboard
│   │
│   └── utils/                        # Shared utilities
│       ├── config.py                # Configuration management
│       └── metrics.py               # Evaluation functions (RMSE, MAE, R²)
│
├── infra/                            # Infrastructure & CI/CD
│   └── github_actions/              # Workflow definitions
│       ├── data_pipeline.yml        # Scheduled data fetching
│       └── model_training.yml       # Periodic retraining
│
└── notebooks/                        # Exploratory analysis
    └── eda.ipynb                    # Data exploration & visualization
```

### Key Components Explained

**Data Layer**:
- `raw/`: Immutable source data preserving original API responses
- `processed/`: Transformed data with computed AQI values
- `feature_store/`: Optimized columnar storage for fast feature access

**Model Layer**:
- `models/`: Serialized model artifacts ready for production deployment
- Versioning through timestamped filenames

**Application Layer**:
- `app/`: User-facing Streamlit dashboard
- Real-time inference using loaded production model

**Infrastructure Layer**:
- GitHub Actions for automated workflows
- Modular design supporting containerization (Docker-ready)

---

## 🛠️ Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.9+ | Core development language |
| **Data Processing** | Pandas, NumPy | Data manipulation and numerical operations |
| **Machine Learning** | Scikit-learn | Model training and evaluation |
| **Feature Store** | Parquet (PyArrow) | Columnar storage format |
| **Web Framework** | Streamlit | Interactive dashboard |
| **Visualization** | Plotly, Matplotlib | Chart rendering |
| **Model Serialization** | Joblib | Efficient model persistence |
| **API Client** | Requests | HTTP API communication |
| **Environment Management** | python-dotenv | Configuration handling |

### APIs & Data Sources

- **Open-Meteo Air Quality API**: Historical air pollution data (PM2.5, PM10, O₃, NO₂, SO₂, CO)
- **Open-Meteo Archive Weather API**: Meteorological data (temperature, humidity, wind, precipitation)

### Development Tools

- **Version Control**: Git & GitHub
- **Dependency Management**: pip + requirements.txt
- **Notebooks**: Jupyter for exploratory analysis
- **Linting**: PEP 8 standards

---

## 📊 Data Pipeline

### Data Collection Strategy

The system implements a robust data collection strategy ensuring comprehensive coverage:

**Temporal Coverage**: 10-month historical window (January 2025 - November 2025) capturing seasonal variations

**Spatial Resolution**: Hourly granularity providing detailed temporal patterns

**Data Sources**:
1. **Air Quality Metrics**: Six primary pollutants measured in μg/m³
   - PM2.5: Fine particulate matter (≤ 2.5 micrometers)
   - PM10: Coarse particulate matter (≤ 10 micrometers)
   - O₃: Ground-level ozone
   - NO₂: Nitrogen dioxide (traffic/industrial)
   - SO₂: Sulfur dioxide (industrial emissions)
   - CO: Carbon monoxide (incomplete combustion)

2. **Meteorological Parameters**: Weather conditions affecting pollution dispersion
   - Temperature (°C): Affects chemical reactions and dispersion
   - Relative Humidity (%): Influences particle behavior
   - Surface Pressure (hPa): Atmospheric stability indicator
   - Wind Speed (m/s): Pollutant dispersion factor
   - Wind Direction (degrees): Pollution transport patterns
   - Precipitation (mm): Wet deposition of pollutants

### Data Quality Assurance

**Validation Checks**:
- Timestamp continuity verification
- Missing value detection and handling
- Outlier identification using statistical bounds
- Duplicate record removal
- Data type consistency enforcement

**Quality Metrics**:
- Completeness: >95% non-null values required
- Accuracy: Cross-validation against government monitoring stations
- Timeliness: Hourly updates with <1 hour latency

### EPA AQI Calculation

The system implements official EPA breakpoint formulas:

**Calculation Method**:
For each pollutant, AQI is calculated using the linear interpolation formula:

```
AQI = [(I_high - I_low) / (C_high - C_low)] × (C - C_low) + I_low
```

Where:
- C: Pollutant concentration
- C_low, C_high: Concentration breakpoints
- I_low, I_high: Index breakpoints

**Overall AQI Determination**: The final AQI is the **maximum** of individual pollutant AQIs, identifying the dominant pollutant.

**AQI Categories** (EPA Standards):
- 0-50: Good (Green) - Air quality satisfactory
- 51-100: Moderate (Yellow) - Acceptable for most
- 101-150: Unhealthy for Sensitive Groups (Orange)
- 151-200: Unhealthy (Red) - General public affected
- 201-300: Very Unhealthy (Purple) - Health alert
- 301-500: Hazardous (Maroon) - Emergency conditions

---

## 🤖 Machine Learning Models

### Model Selection Rationale

The system employs an ensemble approach, training multiple models and selecting the best performer:

**1. Random Forest Regressor**

**Architecture**: Ensemble of 100 decision trees with max depth of 20

**Advantages**:
- Captures non-linear relationships between features
- Handles feature interactions naturally
- Provides feature importance rankings
- Robust to outliers through aggregation
- Excellent for tabular data with mixed feature types

**Hyperparameters**:
- `n_estimators=100`: Balances accuracy and training time
- `max_depth=20`: Prevents overfitting while maintaining complexity
- `min_samples_split=5`: Controls leaf node granularity
- `n_jobs=-1`: Parallel processing for faster training

**Use Case**: Primary model for complex pollution-weather interactions

**2. Ridge Regression**

**Architecture**: Linear regression with L2 regularization (α=1.0)

**Advantages**:
- Fast training and inference
- Interpretable coefficients
- Handles multicollinearity through regularization
- Lower computational requirements
- Suitable for linear trends

**Preprocessing**: StandardScaler normalization ensuring equal feature weights

**Use Case**: Baseline model and fallback for simpler patterns

### Training Strategy

**Data Splitting**:
- Training Set: 80% (chronologically earlier data)
- Test Set: 20% (most recent data for realistic evaluation)
- **No random shuffling**: Maintains temporal integrity for time-series validation

**Evaluation Metrics**:
1. **RMSE (Root Mean Squared Error)**: Penalizes large errors
2. **MAE (Mean Absolute Error)**: Average prediction deviation
3. **R² Score**: Explained variance (0-1 scale, higher better)

**Model Selection Criteria**:
- Primary: Lowest test RMSE
- Secondary: R² score and train-test gap (overfitting check)
- Threshold: R² gap < 0.1 considered acceptable generalization

### Feature Importance Analysis

Random Forest provides feature importance scores revealing pollution drivers:

**Typical Top Features** (from model analysis):
1. PM2.5 and PM10: Strongest AQI predictors
2. Temperature: Affects photochemical reactions
3. Wind Speed: Primary dispersion mechanism
4. Humidity: Influences particle hygroscopicity
5. Time of Day: Traffic and industrial activity patterns

---

## 🔧 Feature Engineering

Feature engineering transforms raw measurements into predictive signals:

### Time-Based Features

**Temporal Patterns**:
- Hour of Day (0-23): Captures diurnal traffic patterns
- Day of Week (0-6): Weekend vs. weekday differences
- Month (1-12): Seasonal variations
- Is Weekend (binary): Reduced industrial/traffic activity

**Cyclical Encoding**: Sin/cosine transformations for hour and day preserve temporal continuity (23:00 → 00:00 smooth transition)

### Lagged Features

**Historical Values**: Previous time steps capture momentum and trends

**Implemented Lags**:
- 1-hour lag: Immediate previous conditions
- 3-hour lag: Recent trend
- 6-hour lag: Quarter-day pattern
- 12-hour lag: Half-day cycle
- 24-hour lag: Daily periodicity
- 48-hour lag: Two-day pattern
- 72-hour lag: Three-day trend

**Purpose**: Enables model to learn auto-regressive patterns in pollution levels

### Rolling Statistics

**Moving Averages**: Smoothed trends over time windows

**Windows**:
- 6-hour: Short-term variations
- 12-hour: Mid-term trends
- 24-hour: Daily averages
- 72-hour: Three-day patterns

**Rolling Standard Deviation** (24h and 72h windows): Volatility measurement indicating stable vs. turbulent conditions

**Computed For**: All pollutants and temperature (12 base features × multiple windows)

### Interaction Features

**Cross-Feature Products**: Capture synergistic effects
- Temperature × Wind Speed: Dispersion efficiency
- Humidity × PM2.5: Hygroscopic growth
- Hour × Day of Week: Specific time-day combinations

**Result**: ~100+ features from 12 base measurements, dramatically improving predictive power

---

## 💻 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 2GB free disk space
- Internet connection for API access

### Setup Steps

**1. Clone Repository**

```bash
git clone https://github.com/yourusername/aqi-forecast-project.git
cd aqi-forecast-project
```

**2. Create Virtual Environment**

```bash
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure Environment**

Create `.env` file in project root:

```env
LAT=33.6844
LON=73.0479
CITY=Islamabad
START_DATE=2025-01-01
END_DATE=2025-11-04
```

**5. Verify Installation**

```bash
python -c "import pandas, sklearn, streamlit; print('✓ Installation successful')"
```

---

## 🚀 Usage

### Complete Pipeline Execution

**Step 1: Data Collection**

```bash
python src/ingestion/fetch_api.py
```

**Output**: CSV file in `data/raw/` with hourly measurements

**Step 2: Feature Engineering**

```bash
python src/features/compute_features.py
```

**Output**: Processed CSV in `data/processed/` with computed AQI and categories

**Step 3: Feature Store Population**

```bash
python src/feature_store/build_parquet_store.py
```

**Output**: Parquet tables in `data/feature_store/parquet/`

**Step 4: Model Training**

```bash
python src/training/train.py
```

**Output**: 
- Trained model: `models/best_model.pkl`
- Metrics: `models/metrics.txt`

**Step 5: Launch Dashboard**

```bash
streamlit run src/app/streamlit_app.py
```

**Output**: Web dashboard at `http://localhost:8501`

### Quick Start (Demo Mode)

If you want to quickly see results without waiting for data collection:

```bash
# Use sample data (if provided)
python src/training/train.py --demo

# Launch dashboard
streamlit run src/app/streamlit_app.py
```

---

## 📱 Dashboard Features

### Main Interface Components

**1. Forecast Cards**
- Three-day predictions with large AQI numbers
- Color-coded health categories
- Specific health messages per category
- Date and day-of-week labels

**2. Trend Visualization**
- Combined historical and forecast line chart
- Dual-axis showing past 5 days + future 3 days
- Hover tooltips with exact values
- Reference lines for AQI thresholds

**3. Pollutant Status Panel**
- Real-time concentrations for PM2.5, PM10, O₃
- WHO guideline comparisons
- Status badges (Good/Moderate/Poor/Very Poor)
- Color-coded severity indicators

**4. Environmental Insights**
- Weekly change percentages for key metrics
- Temperature trend analysis
- Wind speed impact on air circulation
- Contextual interpretations

**5. Health Recommendations**
- Category-specific activity guidelines
- Vulnerable group warnings
- Protective measure suggestions (masks, air purifiers)
- Outdoor activity planning advice

**6. Historical Data Table** (Expandable)
- Last 7 days of daily-averaged features
- Sortable columns
- Rounded decimal precision
- Exportable format

### User Interaction Flow

1. User opens dashboard (automatic data loading)
2. View current AQI status and 3-day forecast
3. Explore trend chart to understand patterns
4. Check specific pollutant levels
5. Read health recommendations
6. Access historical data for context

---

## 📈 Model Performance

### Evaluation Results

Based on training with 10 months of Islamabad data:

**Random Forest Performance**:
- Test RMSE: ~7-8 AQI points
- Test MAE: ~5-6 AQI points
- Test R²: ~0.80-0.85 (80-85% variance explained)
- Training Time: ~2-3 seconds (100 trees)

**Ridge Regression Performance**:
- Test RMSE: ~9-10 AQI points
- Test MAE: ~7-8 AQI points
- Test R²: ~0.75-0.80
- Training Time: <1 second

**Winner**: Random Forest (lower RMSE, higher R²)

### Interpretation

- **RMSE of 7-8**: Predictions within ±7-8 AQI points of actual values
- **R² of 0.82**: Model explains 82% of AQI variation
- **Acceptable Error**: Within one AQI subcategory (ranges are 50 points wide)

### Validation Strategy

- **No data leakage**: Strict temporal split (train on past, test on future)
- **Realistic conditions**: Test set represents unseen future scenarios
- **Overfitting check**: Train R² - Test R² gap < 0.10 considered acceptable

---

## 🔮 Future Enhancements

### Short-Term Improvements (1-3 months)

1. **Extended Forecast Horizon**
   - 7-day predictions
   - Confidence intervals
   - Ensemble predictions

2. **Multi-Location Support**
   - Lahore, Karachi, Rawalpindi
   - Comparative city analysis
   - Regional pollution tracking

3. **Advanced Models**
   - LSTM for sequential patterns
   - XGBoost for gradient boosting
   - Model ensembling

4. **Alert System**
   - Email notifications for high AQI
   - SMS alerts for sensitive groups
   - Customizable thresholds

### Long-Term Vision (6-12 months)

1. **Real-Time Deployment**
   - Cloud hosting (AWS/GCP/Azure)
   - Auto-scaling infrastructure
   - CDN for global access

2. **Mobile Application**
   - iOS and Android apps
   - Push notifications
   - Offline mode with cached predictions

3. **Policy Integration**
   - Government dashboard
   - Regulatory compliance tracking
   - Impact assessment tools

4. **Advanced Analytics**
   - Pollution source attribution
   - Economic impact analysis
   - Long-term trend forecasting

5. **Community Features**
   - User-reported air quality
   - Social sharing
   - Crowdsourced validation

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues

- Use GitHub Issues
- Provide detailed description
- Include steps to reproduce
- Attach relevant logs/screenshots

### Submitting Changes

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update README for significant changes

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Open-Meteo**: For providing free, high-quality air quality and weather data APIs
- **EPA**: For standardized AQI calculation methodology
- **WHO**: For air quality guidelines and health recommendations
- **Streamlit**: For enabling rapid dashboard development
- **Scikit-learn**: For comprehensive machine learning tools

---

## 📞 Contact

**Project Maintainer**: [Your Name]

**Email**: [your.email@example.com]

**GitHub**: [@yourusername](https://github.com/yourusername)

**LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/aqi-forecast-project?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/aqi-forecast-project?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/aqi-forecast-project)
![GitHub license](https://img.shields.io/github/license/yourusername/aqi-forecast-project)

---

**Last Updated**: November 2025

**Version**: 1.0.0

**Status**: Active Development
