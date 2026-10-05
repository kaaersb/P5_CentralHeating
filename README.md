# P5_CentralHeating# P5 Central Heating

A data-driven anomaly detection pipeline for identifying unusual behaviour in central heating systems using heat-meter data and machine learning.

The project processes large-scale heating data, trains an **Isolation Forest** model to identify anomalous measurements, and analyses the detected anomalies to identify heat meters with potentially abnormal behaviour.

## Overview

The project consists of three main stages:

1. **Data processing** – Clean and prepare the raw heat-meter data.
2. **Anomaly detection** – Train an Isolation Forest model and classify observations as normal or anomalous.
3. **Anomaly analysis** – Aggregate anomalies by heat meter and generate diagnostic visualisations.

The anomaly detection model uses the following features:

* `Delta_T_24h` – 24-hour rolling change in temperature difference
* `Flow_24h` – 24-hour rolling flow
* `T_outdoor` – Outdoor temperature

The Isolation Forest uses a contamination rate of `0.05`, meaning the model is configured to expect approximately 5% of observations to be anomalous. A fixed random seed is used to make the results reproducible.

---

## Project Structure

```text
P5_CentralHeating/
│
├── Data/
│   ├── ...                         # Raw data
│   └── Processed_Data/
│       ├── processed_heat_data.csv
│       └── processed_heat_data_with_anomalies.csv
│
├── Data_processing/
│   └── ...                         # Data cleaning and preprocessing
│
├── Models/
│   └── isolation_forest_model.pkl # Trained Isolation Forest
│
├── Outputs/
│   ├── worst_meters_ranking.csv
│   └── anomaly_scatter_plot.png
│
├── Analyze_Anomalies.py
├── Model_Training.py
├── LICENSE
└── README.md
```

The repository is organised around processed data, machine-learning models, and generated analysis outputs.

---

## Method

### Isolation Forest

An **Isolation Forest** is used for unsupervised anomaly detection.

Unlike supervised classification, the model does not require manually labelled examples of faulty heating systems. Instead, it learns the structure of the data and identifies observations that are easier to isolate from the rest of the dataset.

The model is configured as:

```python
IsolationForest(
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)
```

The model is trained using a sample of up to **2,000,000 observations** to limit memory usage. Predictions are then generated over the dataset in chunks.

### Anomaly labels

The Isolation Forest produces two labels:

| Label | Meaning               |
| ----: | --------------------- |
|   `1` | Normal observation    |
|  `-1` | Anomalous observation |

An anomaly does not necessarily mean that a heat meter is faulty. It indicates that the observation differs significantly from the patterns learned by the model and should therefore be investigated further.

---

## Data Pipeline

The general workflow is:

```text
Raw Heat-Meter Data
        │
        ▼
┌─────────────────────┐
│ Data Processing     │
│ & Feature Creation  │
└──────────┬──────────┘
           │
           ▼
processed_heat_data.csv
           │
           ▼
┌─────────────────────┐
│ Isolation Forest    │
│ Model Training      │
└──────────┬──────────┘
           │
           ├──────────────► isolation_forest_model.pkl
           │
           ▼
processed_heat_data_with_anomalies.csv
           │
           ▼
┌─────────────────────┐
│ Anomaly Analysis    │
└──────────┬──────────┘
           │
           ├──────────────► worst_meters_ranking.csv
           │
           └──────────────► anomaly_scatter_plot.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kaaersb/P5_CentralHeating.git
cd P5_CentralHeating
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn joblib tqdm matplotlib seaborn
```

---

## Usage

### 1. Process the data

Run the data-processing pipeline first.

The processing stage should produce:

```text
Data/Processed_Data/processed_heat_data.csv
```

This file is used as the input to the machine-learning stage.

---

### 2. Train the anomaly detection model

Run:

```bash
python Model_Training.py
```

The script:

1. Loads the processed heat-meter dataset.
2. Reads the data in chunks to reduce memory pressure.
3. Selects the features used for anomaly detection.
4. Samples up to 2 million observations for model training.
5. Trains an Isolation Forest.
6. Predicts anomalies across the complete dataset.
7. Saves the trained model.
8. Exports the dataset with anomaly labels.

The trained model is saved to:

```text
Models/isolation_forest_model.pkl
```

The labelled dataset is saved to:

```text
Data/Processed_Data/processed_heat_data_with_anomalies.csv
```

The training script is designed for large datasets and uses chunked processing during loading, prediction, and output.

---

### 3. Analyse detected anomalies

After model training, run:

```bash
python Analyze_Anomalies.py
```

The analysis script:

* Filters observations classified as anomalies.
* Groups anomalies by `MeterID`.
* Counts the number of anomalies for each meter.
* Calculates the percentage of anomalous readings.
* Ranks meters by anomaly count.
* Generates a scatter plot comparing outdoor temperature with 24-hour temperature change.

The results are saved to:

```text
Outputs/
├── worst_meters_ranking.csv
└── anomaly_scatter_plot.png
```

---

## Output

### Meter ranking

`worst_meters_ranking.csv` contains information about the anomaly behaviour of individual heat meters.

The main columns are:

| Column               | Description                                        |
| -------------------- | -------------------------------------------------- |
| `MeterID`            | Unique heat-meter identifier                       |
| `Anomaly_Count`      | Number of detected anomalies                       |
| `Total_Readings`     | Total number of observations                       |
| `Anomaly_Percentage` | Percentage of observations classified as anomalous |

The meters are sorted by anomaly count, making it possible to identify installations that may require further investigation.

### Anomaly scatter plot

`anomaly_scatter_plot.png` visualises the relationship between:

* Outdoor temperature (`T_outdoor`)
* 24-hour rolling temperature difference (`Delta_T_24h`)
* Anomaly status

A sample of up to 50,000 observations is used for the visualisation to keep the plot manageable.

---

## Computational Considerations

The dataset can contain tens of millions of observations, so memory management is an important part of the pipeline.

The training script uses:

* Chunked CSV loading
* Random sampling for model training
* Chunked anomaly prediction
* Chunked CSV output
* Multi-core processing through `n_jobs=-1`

The model itself is trained on a maximum of 2 million randomly selected observations rather than the complete dataset. This provides a practical compromise between model training time, memory consumption, and statistical coverage.

---

## Reproducibility

Random operations use:

```text
random_state = 42
```

This is used both when sampling the training data and when configuring the Isolation Forest, allowing the same dataset and environment to produce reproducible results.

---

## Limitations

The anomaly detection results should be interpreted as **indicators of unusual behaviour rather than proof of a malfunction**.

In particular:

* Isolation Forest is an unsupervised method and does not know whether an anomaly represents an actual technical fault.
* The `5%` contamination parameter is a modelling assumption.
* The model only considers three features.
* Anomalies may be caused by legitimate operating conditions, sensor errors, data quality issues, or actual heating-system problems.
* Further domain-specific analysis is required before concluding that a specific meter is faulty.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.