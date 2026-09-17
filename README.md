# Construction Safety Risk Predictor

An end-to-end machine learning and MLOps project for predicting
construction worker safety behavior.

The project extends an existing agent-based construction safety
simulation and uses synthetic simulation data to train machine
learning surrogate models.

## Project Status

Current phase:

**Phase 1 — Data Generation**

## Architecture

```text
ABM Simulation
      |
      v
Synthetic Dataset
      |
      v
Machine Learning
      |
      v
MLflow
      |
      v
FastAPI
      |
      v
Streamlit
