# Land and Ocean Temperature Prediction

This Jupyter notebook, `landOceanTempPrediction.ipynb`, implements a machine learning model to predict the Land and Ocean Average Temperature based on historical weather data.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Project Overview

The goal of this project is to build and train a `RandomForestRegressor` model to predict the `LandAndOceanAverageTemperature`. The notebook includes steps for:

- Loading and initial exploration of the dataset.
- Data cleaning and preprocessing, including handling missing values and feature engineering (extracting month and year).
- Splitting the data into training and testing sets.
- Training a `RandomForestRegressor` model.
- Evaluating the model's performance.
- A section to take user input for predictions.

## Data

The notebook expects a CSV file named `weather_data.csv` to be present in the same directory as the notebook. This file should contain historical temperature data, including columns such as:

- `dt` (date)
- `LandAverageTemperature`
- `LandMaxTemperature`
- `LandMinTemperature`
- `LandAndOceanAverageTemperature`
- And their respective Uncertainty columns (which are dropped during preprocessing).

## Installation

To run this notebook, you need to have Python and the required libraries installed. It's recommended to use a virtual environment.

1. Clone the repository (if applicable) or download the notebook and data file.
2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
3. Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
4. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Install Jupyter (if you don't have it):
    ```bash
    pip install jupyter
    ```

## Usage

1. Ensure `weather_data.csv` is in the same directory as the notebook.
2. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3. Open `landOceanTempPrediction.ipynb` in your web browser.
4. Run all cells sequentially.
5. The notebook will print information about the dataset, clean the data, train the model, and at the end, prompt you to enter values for `LandAverageTemperature`, `LandMaxTemperature`, and `LandMinTemperature` to get a prediction.

## Dependencies

The following Python libraries are used in this project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `seaborn`: For statistical data visualization.
- `matplotlib`: For creating static, interactive, and animated visualizations.
- `scikit-learn`: For machine learning functionalities, including `RandomForestRegressor` and `mean_squared_error`.