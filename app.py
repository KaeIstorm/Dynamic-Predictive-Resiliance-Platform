import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import altair as alt

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NexGen DPRP: Dynamic Predictive Resilience Platform")

# --- GLOBAL VARIABLES & DATA LOADING (Re-run all analysis steps) ---
@st.cache_data
def load_and_prepare_data():
    # 1. Load Core Data
    df_orders = pd.read_csv('Dataset/orders.csv')
    df_delivery = pd.read_csv('Dataset/delivery_performance.csv')
    df_routes = pd.read_csv('Dataset/routes_distance.csv')
    df_cost = pd.read_csv('Dataset/cost_breakdown.csv')
    df_feedback = pd.read_csv('Dataset/customer_feedback.csv')
    df_fleet = pd.read_csv('Dataset/vehicle_fleet.csv')

    # 2. Merge Core Order Data (Inner Join for completed orders)
    df_master = pd.merge(df_orders, df_delivery, on='Order_ID', how='inner')
    df_master = pd.merge(df_master, df_routes, on='Order_ID', how='inner')
    df_master = pd.merge(df_master, df_cost, on='Order_ID', how='inner')
    df_master = pd.merge(df_master, df_feedback, on='Order_ID', how='inner')

    # 3. Core Feature Engineering
    df_master['Delay_Days'] = df_master['Actual_Delivery_Days'] - df_master['Promised_Delivery_Days']
    df_master['Is_Delayed'] = (df_master['Delay_Days'] > 0).astype(int)
    df_master['Total_Op_Cost'] = df_master[['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']].sum(axis=1)
    df_master['Is_Bad_Review'] = (df_master['Rating'] <= 2).astype(int)

    # 4. Prepare Fleet & Carrier Data for Optimization
    df_fleet['Op_Cost_per_KM_INR'] = (df_fleet['Age_Years'] * 1.5 + df_fleet['CO2_Emissions_Kg_per_KM'] * 20) / df_fleet['Fuel_Efficiency_KM_per_L']

    df_delivery['Delay_Days'] = df_delivery['Actual_Delivery_Days'] - df_delivery['Promised_Delivery_Days']
    carrier_metrics = df_delivery.groupby('Carrier').agg(
        Avg_Cost=('Delivery_Cost_INR', 'mean'),
        Avg_Delay=('Delay_Days', 'mean'),
        Count=('Order_ID', 'count')
    ).reset_index()

    return df_master, df_fleet, carrier_metrics

# --- 5. PREDICTIVE MODEL TRAINING (Must be outside the optimization function) ---
def train_predictive_model(df):
    features = ['Priority', 'Order_Value_INR', 'Distance_KM', 'Fuel_Consumption_L',
                'Traffic_Delay_Minutes', 'Weather_Impact', 'Carrier', 'Special_Handling']
    target = 'Is_Delayed'

    # Filter features to those available in the master dataset
    available_features = [f for f in features if f in df.columns]

    X = df[available_features]
    y = df[target]
    
    # Encoding
    X = pd.get_dummies(X, columns=['Priority', 'Weather_Impact', 'Carrier', 'Special_Handling'], drop_first=True)
    
    # Retrain model on all data to get P_Delay for all 83 records
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    
    # Calculate P_Delay
    df['P_Delay'] = model.predict_proba(X)[:, 1]

    # Calculate RADT
    max_delay_factor = df['Delay_Days'].quantile(0.95) if df['Delay_Days'].quantile(0.95) > 0 else 1
    df['RADT'] = df['Promised_Delivery_Days'] + (df['P_Delay'] * max_delay_factor)
    
    return df, max_delay_factor

# --- 6. OPTIMIZATION LOGIC FUNCTION (The Fixer) ---
def simulate_optimization(order, df_fleet, carrier_metrics, max_delay_factor):
    DELAY_COST_WEIGHT = 100
    TIME_REDUCTION_FACTOR = 0.5
    
    # Auxiliary Data
    best_external_carrier = carrier_metrics.loc[carrier_metrics['Avg_Delay'].idxmin()]
    cheapest_vehicle = df_fleet[df_fleet['Status'] == 'Available'].sort_values('Op_Cost_per_KM_INR').iloc[0]

    # Base Metrics
    base_p_delay = order['P_Delay']
    route_dist = order['Distance_KM']
    base_cost = order['Delivery_Cost_INR']

    # --- Option 1: Internal Re-route (Minimize OpCost/CO2) ---
    re_route_cost = order['Toll_Charges_INR'] + (route_dist * cheapest_vehicle['Op_Cost_per_KM_INR'])
    re_route_p_delay = base_p_delay * TIME_REDUCTION_FACTOR
    re_route_score = re_route_cost + (DELAY_COST_WEIGHT * re_route_p_delay * max_delay_factor)

    # --- Option 2: External Offload (Minimize Delay/Cost) ---
    offload_cost = best_external_carrier['Avg_Cost']
    offload_p_delay = (base_p_delay + (best_external_carrier['Avg_Delay'] / max_delay_factor)) / 2
    offload_score = offload_cost + (DELAY_COST_WEIGHT * offload_p_delay * max_delay_factor)

    # --- Decision ---
    if re_route_score < offload_score:
        result = {
            'Action': 'Internal Re-route',
            'New_Cost': re_route_cost,
            'New_P_Delay': re_route_p_delay,
            'Vehicle': cheapest_vehicle['Vehicle_ID'],
            'Details': f"Re-route to cheapest available asset ({cheapest_vehicle['Vehicle_Type']}). Op. Cost: {cheapest_vehicle['Op_Cost_per_KM_INR']:.2f} INR/KM."
        }
    else:
        result = {
            'Action': f'External Offload to {best_external_carrier["Carrier"]}',
            'New_Cost': offload_cost,
            'New_P_Delay': offload_p_delay,
            'Vehicle': 'N/A',
            'Details': f"Offload to best partner ({best_external_carrier['Carrier']}). Avg Delay: {best_external_carrier['Avg_Delay']:.2f} days."
        }

    result['Cost_Change'] = result['New_Cost'] - base_cost
    result['P_Delay_Reduction'] = base_p_delay - result['New_P_Delay']
    return result

# --- STREAMLIT APP LAYOUT ---
df_master, df_fleet, carrier_metrics = load_and_prepare_data()
df_master, max_delay_factor = train_predictive_model(df_master.copy()) # Use a copy for manipulation

# --- TITLE ---
st.title("🛡️ Dynamic Predictive Resilience Platform (DPRP)")
st.caption("NexGen Logistics: From Reactive to Predictive Operations. Data-driven tool for intervention simulation.")

# --- SIDEBAR (Interactivity/Filters) ---
st.sidebar.header("Data Filters")
selected_priority = st.sidebar.multiselect('Filter by Priority', df_master['Priority'].unique(), default=df_master['Priority'].unique())
selected_segment = st.sidebar.multiselect('Filter by Customer Segment', df_master['Customer_Segment'].unique(), default=df_master['Customer_Segment'].unique())
min_p_delay = st.sidebar.slider("Min Probability of Delay (P_Delay)", 0.0, 1.0, 0.4)
df_filtered = df_master[
    (df_master['Priority'].isin(selected_priority)) &
    (df_master['Customer_Segment'].isin(selected_segment)) &
    (df_master['P_Delay'] >= min_p_delay)
]
st.sidebar.markdown(f"**Filtered Orders:** {len(df_filtered)}")


# --- MAIN DASHBOARD TABS ---
tab1, tab2 = st.tabs(["🚀 Intervention Simulator (The Fixer)", "📊 Resilience Dashboard"])

# ----------------------------------------------------
# TAB 1: INTERVENTION SIMULATOR
# ----------------------------------------------------
with tab1:
    st.header("Proactive Intervention Simulator")
    st.markdown("Select a high-risk order below to run the multi-objective optimization engine and determine the best corrective action (Internal Re-route vs. External Offload).")

    # Order Selection (Interactivity Requirement)
    high_risk_orders_list = df_filtered.sort_values('P_Delay', ascending=False).head(20)
    order_id = st.selectbox(
        'Select a High-Risk Order ID (Sorted by P_Delay)',
        options=high_risk_orders_list['Order_ID'].tolist(),
        format_func=lambda x: f"{x} (P_Delay: {high_risk_orders_list[high_risk_orders_list['Order_ID'] == x]['P_Delay'].iloc[0]:.2f})"
    )
    
    if order_id:
        selected_order = df_master[df_master['Order_ID'] == order_id].iloc[0]
        st.subheader(f"Order Details: {order_id}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current P(Delay)", f"{selected_order['P_Delay']:.2f}")
        col2.metric("Base Delivery Cost", f"INR {selected_order['Delivery_Cost_INR']:.2f}")
        col3.metric("Promised Days", f"{selected_order['Promised_Delivery_Days']} days")
        
        st.markdown("---")
        st.subheader("Optimization Engine Recommendation")
        
        # Run the Optimization Logic
        optimization_result = simulate_optimization(selected_order, df_fleet, carrier_metrics, max_delay_factor)
        
        # Display Results
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Recommended Action", optimization_result['Action'])
        colB.metric("Projected Cost", f"INR {optimization_result['New_Cost']:.2f}", f"{optimization_result['Cost_Change']:.2f} INR")
        colC.metric("Projected P(Delay)", f"{optimization_result['New_P_Delay']:.2f}", f"-{optimization_result['P_Delay_Reduction']:.2f} P(Delay)")
        colD.metric("Proactive CX Message", "Immediate Notification to Customer", "Risk Averted")

        st.info(f"**Justification:** {optimization_result['Details']}")

# ----------------------------------------------------
# TAB 2: RESILIENCE DASHBOARD
# ----------------------------------------------------
with tab2:
    st.header("Core Operational Resilience Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders Analyzed", len(df_master))
    col2.metric("Delay Rate (Is_Delayed=1)", f"{df_master['Is_Delayed'].mean() * 100:.1f}%")
    col3.metric("Average Customer Rating", f"{df_master['Rating'].mean():.2f} / 5")
    col4.metric("Cost Leakage (Avg Op. Cost)", f"INR {df_master['Total_Op_Cost'].mean():.2f}")
    
    st.markdown("---")

    # --- CHART 1: Root Cause Analysis (Bar Chart) ---
    st.subheader("Root Cause Analysis: Delays & Issues")
    issue_counts = df_master.groupby('Issue_Category')['Is_Delayed'].sum().sort_values(ascending=False).reset_index()
    
    chart1 = alt.Chart(issue_counts).mark_bar().encode(
        x=alt.X('Issue_Category', sort='-y', title='Primary Issue Category'),
        y=alt.Y('Is_Delayed', title='Total Delayed Orders'),
        tooltip=['Issue_Category', 'Is_Delayed']
    ).properties(title='Delayed Orders by Primary Customer Issue Category').interactive()
    st.altair_chart(chart1, use_container_width=True)

    # --- CHART 2: Risk Profile Scatter Plot (Scatter Plot) ---
    st.subheader("Risk & Cost Profile (Outlier Identification)")
    chart2 = alt.Chart(df_master).mark_circle().encode(
        x=alt.X('Distance_KM', title='Route Distance (KM)'),
        y=alt.Y('Delivery_Cost_INR', title='Delivery Cost (INR)'),
        color='P_Delay', # Color by Probability of Delay
        size=alt.Size('P_Delay', legend=None),
        tooltip=['Order_ID', 'Route', 'Distance_KM', 'Delivery_Cost_INR', 'P_Delay']
    ).properties(title='Delivery Cost vs. Distance, Colored by P(Delay) Risk').interactive()
    st.altair_chart(chart2, use_container_width=True)

    # --- CHART 3: Performance by Carrier (Grouped Bar Chart) ---
    st.subheader("Carrier Performance vs. NexGen Fleet")
    carrier_performance = df_master.groupby('Carrier').agg(
        Avg_Delay_Days=('Delay_Days', 'mean'),
        Count=('Order_ID', 'count')
    ).reset_index().sort_values('Avg_Delay_Days', ascending=False)
    
    chart3 = alt.Chart(carrier_performance).mark_bar().encode(
        x=alt.X('Carrier', sort='-y', title='Carrier'),
        y=alt.Y('Avg_Delay_Days', title='Average Delay Days'),
        tooltip=['Carrier', 'Avg_Delay_Days', 'Count']
    ).properties(title='Average Delay Days by Carrier').interactive()
    st.altair_chart(chart3, use_container_width=True)

    # --- CHART 4: Traffic Impact Distribution (Histogram) ---
    st.subheader("Traffic Delay Distribution")
    chart4 = alt.Chart(df_master).mark_bar().encode(
        alt.X("Traffic_Delay_Minutes", bin=True, title='Traffic Delay (Minutes)'),
        alt.Y('count()', title='Number of Orders'),
        tooltip=[alt.Tooltip("Traffic_Delay_Minutes", bin=True), 'count()']
    ).properties(title='Distribution of Traffic Delays').interactive()
    st.altair_chart(chart4, use_container_width=True)
    
    # --- Download Functionality (Requirement) ---
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(df_master)
    st.download_button(
        label="Download Full DPRP Modeling Data",
        data=csv,
        file_name='dprp_analysis_data.csv',
        mime='text/csv',
    )