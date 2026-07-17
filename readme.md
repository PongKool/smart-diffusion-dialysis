# Machine Learning-Powered Diffusion Dialysis System

This diffusion dialysis system is designed to treat wastewater from the metal industry. The system is autonomous, IoT-controlled, and utilizes machine learning to assess and predict membrane health.


## 🌟 Overview

This project provides a complete pipeline from synthetic data generation to machine learning model training and a fully interactive web dashboard. It utilizes various predictive models to monitor the efficiency and maintenance needs of diffusion dialysis membranes, such as predicting fouling, acid recovery rates, and estimating the remaining days until the membrane requires cleaning.

## 📁 Repository Structure

* **`dashboard.py`**: The main Streamlit application for the interactive user dashboard.
* **`train_model.py`**: Script used to train the machine learning models.
* **`generate_data.py`** / **`generate_good_data.py`**: Scripts for generating synthetic membrane and temperature-corrected data for training and testing.
* **`generate_data_for_training.py`**: Preprocessing script to convert raw `sensors.csv` data into the standardized format required for training.
* **`generate_sensors.py`**: Script to simulate raw sensor output data.
* **`*.pkl` files**: Pre-trained machine learning models:
    * `recovery_model.pkl`: Predicts the acid recovery performance.
    * `fouling_model.pkl`: Assesses the current fouling state of the membrane.
    * `days_to_cleaning_model.pkl`: Predicts how many days are left before a cleaning cycle is necessary.
* **`sensors.csv`**: The raw sensor log file.
* **`diffusion_dialysis_input_data.csv`** & **`diffusion_dialysis_input_data2.csv`**: Standardized datasets used as primary inputs for model training.
* **`wake_apps.py`** & **`.github/workflows`**: Automation scripts and GitHub Actions to keep the Streamlit Cloud application awake and running.
* **`.devcontainer/`**: Configuration for standardizing the development environment.
* **`requirements.txt`**: Python dependencies required to run the project.
