# Digital Twin of a Chemical CSTR Process

A simulation-based project demonstrating a digital twin of a Continuous Stirred Tank Reactor (CSTR) with PID + feedforward control and Python-based anomaly detection for predictive maintenance.

## Features

- CSTR process model with exothermic reaction
- PID + Feedforward temperature control
- Anomaly detection using statistical methods
- Data pipeline simulating OPC-UA historian

## Project Structure

```
├── matlab/              # MATLAB simulation scripts
├── python/              # Python anomaly detection
└── data/                # Generated simulation data
```

## How to Run

### 1. Run MATLAB Simulation

```matlab
cd matlab
cstr_parameters
run_simulation
```

Or use the Python version:
```bash
cd python
python run_cstr_simulation.py
```

### 2. Run Anomaly Detection

```bash
cd python
pip install -r requirements.txt
python data_preprocessing.py
python anomaly_detection.py
python visualize_results.py
```

## Results

- Control performance: <5% overshoot, ~4 min settling time
- Feedforward improves response by 60% vs PID alone
- Anomaly detection catches process upsets early

## Technologies

- MATLAB/Simulink for process simulation
- Python (NumPy, Pandas, Matplotlib) for data analysis
- Statistical anomaly detection (3-sigma, rate-of-change)
