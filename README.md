# FoW-Sim — Future of Work Scenario Simulator

**BSc Honours Project: Predictive Analysis and Simulation of the Future of Work Using AI and Machine Learning**

A comprehensive AI-powered workforce prediction system that analyzes labor market trends, automation risk, and economic indicators to forecast employment patterns across 25 countries over 5, 10, and 20-year horizons.

## Features

- **Data Pipeline**: World Bank API + OECD integration + external datasets (automation risk, remote work trends, AI adoption, skills gap)
- **ML Models**: 12 algorithms including Random Forest, XGBoost, LightGBM, TensorFlow Neural Networks, LSTM
- **Time-Series Forecasting**: ARIMA, ETS, and ensemble methods
- **Scenario Simulation**: 9 future scenarios (Rapid AI, Remote Work Revolution, Green Economy, etc.)
- **Ethics Framework**: IEEE AI Ethics compliance with fairness metrics
- **Interactive Dashboard**: Streamlit with Plotly visualizations
- **REST API**: FastAPI backend

---

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

---

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd fow-sim
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install -U pip

# Install package in development mode with all dependencies
pip install -e ".[dev]"
```

### Step 4: Verify Installation
```bash
# Check installed packages
pip list

# Verify fowsim is installed
python -c "import fowsim; print('FoW-Sim installed successfully!')"
```

---

## Testing

### Run All Tests
```bash
# Run all 37 tests with verbose output
python -m pytest tests/ -v

# Run tests with short traceback
python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/test_models.py -v
python -m pytest tests/test_scenarios.py -v
python -m pytest tests/test_external_data.py -v
python -m pytest tests/test_data_validation.py -v
python -m pytest tests/test_validate.py -v
```

### Test Coverage
```bash
# Run tests with coverage report
python -m pytest tests/ --cov=fowsim --cov-report=html
```

### Type Checking
```bash
# Run mypy type checker
python -m mypy src/fowsim --ignore-missing-imports
```

### Linting
```bash
# Run ruff linter
python -m ruff check src/fowsim
```

---

## Running the Application

### CLI Commands

The CLI provides access to all functionality:

```bash
# Show all available commands
python -m fowsim.cli --help
```

#### Build Dataset (World Bank + External Data)
```bash
# Build panel dataset from 2000 to 2024
python -m fowsim.cli build-data --start-year 2000 --end-year 2024
```

#### Train Models
```bash
# Train forecasting models with backtesting
python -m fowsim.cli train --horizons 5 10 20

# Train with baseline models (ARIMA, ETS)
python -m fowsim.cli train --horizons 5 10 20 --include-baseline
```

#### Run Scenario Simulation
```bash
# Simulate future scenario for a country
python -m fowsim.cli simulate --country USA --scenario rapid_ai --horizon 10
python -m fowsim.cli simulate --country GBR --scenario remote_work_revolution --horizon 20
python -m fowsim.cli simulate --country DEU --scenario green_economy --horizon 5
```

#### Evaluate Ethics & Fairness
```bash
# Run IEEE Ethics Framework evaluation
python -m fowsim.cli evaluate-ethics
```

#### Compare Models
```bash
# Compare model performance
python -m fowsim.cli compare-models
```

#### Full Pipeline
```bash
# Run complete pipeline (data → train → evaluate)
python -m fowsim.cli full-pipeline --start-year 2000 --end-year 2024 --horizons 5 10 20
```

---

### Streamlit Dashboard

```bash
# Run the interactive dashboard
streamlit run src/fowsim/ui/streamlit_app.py

# Run in headless mode (no browser auto-open)
streamlit run src/fowsim/ui/streamlit_app.py --server.headless true

# Run on specific port
streamlit run src/fowsim/ui/streamlit_app.py --server.port 8502
```

Access the dashboard at: **http://localhost:8501**

Dashboard Pages:
- **Home**: Project overview and key metrics
- **Data Explorer**: Explore panel data by country and indicator
- **Model Performance**: View backtesting results and model comparison
- **Scenario Analysis**: Interactive scenario simulation
- **Forecast Viewer**: View predictions for different horizons

---

### FastAPI REST API

```bash
# Run the API server
python -m fowsim.cli run-api

# Run with auto-reload for development
python -m fowsim.cli run-api --reload

# Run on specific port
python -m fowsim.cli run-api --port 8000
```

Access the API at:
- **API Root**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

API Endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `GET /data/summary` - Data summary
- `GET /data/countries` - List of countries
- `GET /data/indicators` - List of indicators
- `GET /data/country/{country}` - Country data
- `GET /scenarios` - Available scenarios
- `POST /simulate` - Run simulation

---

## Project Structure

```
fow-sim/
├── data/
│   ├── raw/                    # External datasets
│   │   ├── automation_risk.csv
│   │   ├── remote_work_trends.csv
│   │   ├── ai_adoption_index.csv
│   │   └── skills_gap_data.csv
│   ├── interim/                # Intermediate data
│   └── processed/              # Final processed data
│       ├── panel.parquet
│       ├── forecasts.parquet
│       └── backtest_metrics.csv
├── src/fowsim/
│   ├── config/
│   │   ├── settings.py         # Configuration
│   │   └── indicators.yaml     # Indicator definitions
│   ├── data/
│   │   ├── pipeline.py         # Data pipeline
│   │   ├── ingest_worldbank.py # World Bank API
│   │   ├── build_features.py   # Feature engineering
│   │   └── validate.py         # Data validation
│   ├── models/
│   │   ├── ml_models.py        # ML model definitions
│   │   ├── tensorflow_models.py # Neural networks
│   │   ├── train.py            # Training logic
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── baselines.py        # ARIMA, ETS
│   │   ├── clustering.py       # Country clustering
│   │   └── ethics.py           # IEEE Ethics Framework
│   ├── simulation/
│   │   ├── scenarios.py        # 9 future scenarios
│   │   └── simulator.py        # Simulation engine
│   ├── ui/
│   │   ├── streamlit_app.py    # Main dashboard
│   │   └── pages/              # Dashboard pages
│   ├── api/
│   │   └── app.py              # FastAPI application
│   └── cli.py                  # Command-line interface
├── tests/                      # Test suite (37 tests)
├── notebooks/                  # Jupyter notebooks
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Verification Commands

Run these commands to verify the project is working correctly:

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 2. Run all tests (should pass 37/37)
python -m pytest tests/ -v

# 3. Verify imports
python -c "from fowsim.data.pipeline import build_panel_dataset; from fowsim.models.ml_models import make_models; from fowsim.models.tensorflow_models import build_dense_nn; from fowsim.models.ethics import evaluate_fairness; from fowsim.simulation.scenarios import scenario_registry; from fowsim.api.app import app; print('All imports OK')"

# 4. Check CLI
python -m fowsim.cli --help

# 5. Verify external datasets exist
python -c "import pandas as pd; print('automation_risk.csv:', len(pd.read_csv('data/raw/automation_risk.csv')), 'rows'); print('remote_work_trends.csv:', len(pd.read_csv('data/raw/remote_work_trends.csv')), 'rows'); print('ai_adoption_index.csv:', len(pd.read_csv('data/raw/ai_adoption_index.csv')), 'rows'); print('skills_gap_data.csv:', len(pd.read_csv('data/raw/skills_gap_data.csv')), 'rows')"

# 6. Check scenarios
python -c "from fowsim.simulation.scenarios import scenario_registry; scenarios = scenario_registry(); print(f'{len(scenarios)} scenarios available:', list(scenarios.keys()))"

# 7. Check ML models
python -c "from fowsim.models.ml_models import make_models; models = make_models(); print(f'{len(models)} ML models available:', list(models.keys()))"

# 8. Run Streamlit (opens browser)
streamlit run src/fowsim/ui/streamlit_app.py
```

---

## Available Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | Current trends continue |
| `rapid_ai` | Accelerated AI adoption |
| `remote_work_revolution` | Widespread remote work |
| `green_economy` | Green jobs transition |
| `skills_crisis` | Widening skills gap |
| `universal_basic_income` | UBI implementation |
| `global_collaboration` | International cooperation |
| `tech_regulation` | Heavy tech regulation |
| `pandemic_recovery` | Post-pandemic recovery |

---

## Supported Countries (25)

**Developed**: USA, GBR, DEU, JPN, CAN, AUS, FRA, ITA

**Emerging**: CHN, IND, BRA, RUS, MEX, IDN, TUR, KOR

**South Asia**: PAK, BGD, LKA, NPL

**Middle East**: SAU, ARE

**Africa**: ZAF, NGA, EGY

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/panel.parquet` | Processed panel data with features |
| `data/processed/forecasts.parquet` | Model predictions |
| `data/processed/backtest_metrics.csv` | Model evaluation metrics |

---

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. **Missing dependencies**: Reinstall package
   ```bash
   pip install -e ".[dev]"
   ```

3. **TensorFlow warnings**: These can be safely ignored (GPU not required)

4. **Streamlit not found**: Install streamlit
   ```bash
   pip install streamlit
   ```

---

## References

- World Bank Open Data API
- OECD Statistics
- IEEE Ethically Aligned Design
- Frey & Osborne (2017) - Automation Risk
- McKinsey Global Institute - Future of Work Reports

---

## Author

BSc Honours Project - Semester 7

---

## License

This project is for educational purposes.
