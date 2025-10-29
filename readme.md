# Dynamic Predictive Resilience Platform (DPRP)

## Project Overview: Transforming NexGen Logistics

The Dynamic Predictive Resilience Platform (DPRP) is an advanced analytics solution developed for NexGen Logistics. Its core function is to transition operations from reactive failure management to a **proactive, predictive, and resilient** model.

The platform integrates Machine Learning (ML) prediction with Multi-Objective Optimization to forecast delivery risks and simulate the optimal, cost-effective intervention before customer experience is impacted. This directly addresses NexGen's mandate to **reduce operational costs by 15-20%** and become an innovation leader in logistics.

## Problem Solved: The Resilience Gap

The fundamental challenge at NexGen is a lack of **operational agility** to effectively mitigate **cascading failures** caused by external uncertainty (traffic, weather, fleet status).

**The DPRP solves this by:**

1.  **Predicting** the **Probability of Delay ($\text{P\_Delay}$)** for every order in transit using ML.
2.  **Optimizing** the corrective action (Internal Re-route vs. External Offload) based on a unified score that minimizes the combined factor of **cost, time-to-delivery, and environmental impact ($\text{CO}_2$)**.

## Key Features of the Streamlit Application

The DPRP is an interactive web prototype built with Python and Streamlit, featuring two primary interfaces:

### 1\. Intervention Simulator (The Fixer)

This interface is the system's core functionality, enabling analysts to run real-time risk simulations.

  * **Risk Ranking:** Displays a dynamic list of orders sorted by their $\text{P\_Delay}$.
  * **Optimization Engine:** Simulates the two corrective strategies—**Internal Re-route** (using the most cost-efficient fleet asset, calculated via **Operational Cost per KM**) and **External Offload** (using the best historical carrier).
  * **Actionable Recommendation:** Provides a clear decision, projected cost change, and projected risk reduction, enabling proactive customer service.

### 2\. Resilience Dashboard

Provides a comprehensive view of operational health and historical cost leakage.

  * **Key Performance Indicators (KPIs):** Displays the overall Delay Rate, Average Customer Rating, and average Cost Leakage.
  * **Root Cause Analysis:** Visualizations link delays and customer issues (from feedback data) to specific operational failures.
  * **Benchmarking:** Compares NexGen's internal fleet performance against third-party carrier benchmarks.

## Technical Implementation & Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Data Processing** | Python (Pandas, NumPy) | Loading, cleaning, and merging the 7 distinct datasets; engineering derived metrics. |
| **Prediction Engine** | Scikit-learn (Logistic Regression) | Training the model to generate the core $\text{P\_Delay}$ and **Risk-Adjusted Delivery Time ($\text{RADT}$)**. |
| **Optimization Logic** | Custom Python Logic | Implementing the multi-objective scoring function to simulate and rank corrective actions. |
| **User Interface** | Streamlit | Building the required interactive, web-based prototype with dynamic filters and visualizations (Altair). |

## Installation and Setup

To run the DPRP application locally, follow these steps.

### Prerequisites

Ensure you have Python installed (3.7+ is recommended).

### 1\. File Structure

Ensure the `dprp_app.py` file and all **7 original CSV files** (`orders.csv`, `delivery_performance.csv`, `vehicle_fleet.csv`, etc.) are located in the same directory.

### 2\. Install Dependencies

You can create a `requirements.txt` file (as listed in the deliverable) and install the necessary libraries:

```
pandas
streamlit
scikit-learn
matplotlib
seaborn
altair
numpy
```

Install them using pip:

```bash
pip install -r requirements.txt
```

### 3\. Run the Application

Execute the mandatory Streamlit command in your terminal from the project directory:

```bash
streamlit run dprp_app.py
```

The application will launch in your default web browser, typically at `http://localhost:8501`.