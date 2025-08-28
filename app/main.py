import streamlit as st
import joblib
import pandas as pd 
import altair as alt
import plotly.express as px
import numpy as np
import gdown
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import pandas as pd


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != "category":
            if col_type == 'int64':
                df[col] = df[col].astype('int32')
            elif col_type == 'float64':
                df[col] = df[col].astype('float32')
    return df 


def get_clean_data():
    df = pd.read_csv('data/sales_train.csv')
    calendar = pd.read_csv('data/calendar.csv')
    inventory = pd.read_csv('data/inventory.csv')

    reduce_memory_usage(df)
    reduce_memory_usage(calendar)
    reduce_memory_usage(inventory)
    train_1 = df.merge(inventory, on=['unique_id','warehouse'], how='left')
    train_1 = train_1.merge(calendar, on=['date', 'warehouse'], how='left')
    train_1['school_holiday'] = train_1['winter_school_holidays'] + train_1['school_holidays']
    train_1['discounts'] = train_1['type_0_discount'] + train_1['type_1_discount'] + train_1['type_2_discount'] + train_1['type_3_discount'] + train_1['type_4_discount'] + train_1['type_5_discount'] + train_1['type_6_discount']
    train_1.rename(columns={'L1_category_name_en': 'category'}, inplace=True)
    train_1.drop(['L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en', 'winter_school_holidays', 'school_holidays', 'type_6_discount', 'type_5_discount', 'type_4_discount', 'type_3_discount', 'type_1_discount', 'type_2_discount', 'type_0_discount'], axis=1, inplace=True)
    train_1['holiday_name'] = train_1['holiday_name'].fillna('no_hol')
    median = train_1['total_orders'].median()
    median1 = train_1['sales'].median()
    train_1.fillna({'total_orders': median, 'sales': median1}, inplace=True)
    train = train_1.copy()
    train['date'] = pd.to_datetime(train['date'])
    train['year'] = train['date'].dt.year
    train['month'] = train['date'].dt.month
    train['quarter'] = train['date'].dt.quarter
    train['day'] = train['date'].dt.day
    train.set_index('date', inplace=True)
    train.sort_index(inplace=True)
    target_col = "sales"

    # Split early
    val_split = train[train.year == 2023]
    train_split = train[train.year != 2023]

    # Separate X and y
    X_train_raw = train_split.drop(columns=[target_col])
    y_train = train_split[target_col]

    X_valid_raw = val_split.drop(columns=[target_col])
    y_valid = val_split[target_col]
    future_dates = pd.date_range(start='2025-07-01', end='2026-12-31')
    calendar_2 = pd.DataFrame({'date': future_dates})
    calendar_2['year'] = calendar_2['date'].dt.year
    calendar_2['month'] = calendar_2['date'].dt.month
    calendar_2['day'] = calendar_2['date'].dt.day
    calendar_2['quarter'] = calendar_2['date'].dt.quarter
    calendar_2[['warehouse', 'total_orders', 'sell_price_main', 'availability', 'category', 'holiday', 'discounts',
                'unique_id', 'product_unique_id', 'name', 'holiday_name', 'shops_closed', 'school_holiday']] = \
    train[['warehouse', 'total_orders', 'sell_price_main', 'availability', 'category', 'holiday', 'discounts',
        'unique_id', 'product_unique_id', 'name', 'holiday_name', 'shops_closed', 'school_holiday']]\
        .reset_index(drop=True).iloc[:len(calendar_2)]

    df_test = calendar_2.copy()
    df_test.set_index('date', inplace=True)
    df_test.sort_index(inplace=True)
    df_test = df_test[X_train_raw.columns]
    return df_test


def add_sidebar(step): 
    st.sidebar.header("Predictors")

    data = get_clean_data()

    date_keys = ["year", "month", "quarter", "day"]
    categorical_cols = ['warehouse', 'name', 'category', 'holiday_name']

    # Create category maps for categorical variables
    category_maps = {
        col: {label: idx for idx, label in enumerate(data[col].astype('category').cat.categories)}
        for col in categorical_cols
    }

    slider_labels = [
        ("Warehouse", "warehouse"),
        ("Total Orders", "total_orders"),
        ("Sell Price Main", "sell_price_main"),
        ("Availability", "availability"),
        ("Category", "category"),
        ("Holiday", "holiday"),
        ("Discounts", "discounts"),
        ("Unique ID", "unique_id"),
        ("Product Unique ID", "product_unique_id"),
        ("Product Name", "name"),
        ("Holiday Name", "holiday_name"),
        ("Shops Closed", "shops_closed"),
        ("School Holiday", "school_holiday"),
        ("Year", "year"),
        ("Month", "month"),
        ("Quarter", "quarter"),
        ("Day", "day")
    ]

    input_dict = {}

    for label, key in slider_labels:
        sidebar_key = f"{key}_{step}"

        col_data = data[key]

        if key in categorical_cols:
            selected_label = st.sidebar.selectbox(label, list(category_maps[key].keys()), key=sidebar_key)
            input_dict[key] = selected_label  # ✅ keep string


        elif key in date_keys:
            min_val = int(col_data.min())
            max_val = 2025 if key == "year" else int(col_data.max())
            mean_val = int(col_data.mean())

            if min_val == max_val:
                max_val = min_val + 1

            input_dict[key] = st.sidebar.number_input(
                label,
                value=mean_val,
                min_value=min_val,
                max_value=max_val,
                key=sidebar_key
            )

        else:
            unique_vals = col_data.unique()

            if set(unique_vals).issubset({0, 1}):
                # Binary column: use checkbox or selectbox
                checked = st.sidebar.checkbox(label, value=bool(col_data.mode()[0]), key=sidebar_key)
                input_dict[key] = int(checked)
            else:
                min_val = 0.0
                max_val = float(col_data.max())
                mean_val = float(col_data.mean())

                if max_val <= min_val:
                    max_val = min_val + 1.0

                if not (min_val <= mean_val <= max_val):
                    mean_val = min_val

                input_dict[key] = st.sidebar.slider(
                    label,
                    min_val,
                    max_val,
                    mean_val,
                    key=sidebar_key
                )

    return input_dict



import pandas as pd
import plotly.express as px

def view_interactive_visualization(input_data, category_maps):
    # Columns to include in the chart
    chart_columns = ["warehouse", "total_orders", "sell_price_main", "category", "year"]

    # Load dataset
    df = get_clean_data()

    # Decode categorical inputs
    decoded_inputs = {}
    for col in ["warehouse", "category"]:
        decoded_inputs[col] = input_data[col]

    # Numeric inputs
    decoded_inputs.update({
        "total_orders": input_data["total_orders"],
        "sell_price_main": input_data["sell_price_main"],
        "year": input_data["year"]
    })

    chart_data = []

    for col in chart_columns:
        if col in ["warehouse", "category"]:
            # Use frequency count as max value
            freq = df[col].value_counts()
            max_val = freq.max()

            # Use selected category count (frequency in dataset)
            selected_label = decoded_inputs[col]
            selected_val = freq.get(selected_label, 0)

            chart_data.append({
                "Feature": col,
                "Value": selected_val if selected_val else max_val,  # default to max if no selection
                "Label": selected_label
            })

        else:
            # Numeric columns
            max_val = df[col].max()
            selected_val = decoded_inputs[col]

            chart_data.append({
                "Feature": col,
                "Value": selected_val if selected_val else max_val,
                "Label": f"Selected {col}"
            })

    chart_df = pd.DataFrame(chart_data)

    # Create single-bar chart
    fig = px.bar(
        chart_df,
        x="Feature",
        y="Value",
        text="Label",
        hover_data=["Label", "Value"],
        title="Dynamic Feature Values",
        template="plotly_white"
    )

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def add_predictions(input_data):
    model = joblib.load(open("model/random_forest_model.pkl", "rb"))
    scaler = joblib.load(open("model/preprocessor.pkl", "rb"))

    input_df = pd.DataFrame([input_data])
    
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0] 
    prediction_rounded = int(round(prediction))  

    st.subheader("Units Of Sales Made")
    st.write(f"Sales Prediction: {prediction_rounded} units")

def main():
    st.set_page_config(
        page_title="Sales_Predictor",
        page_icon="Grocery-store",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    step = 1
    data = get_clean_data()

    categorical_cols = ['warehouse', 'name', 'category', 'holiday_name']
    category_maps = {
        col: {label: idx for idx, label in enumerate(data[col].astype('category').cat.categories)}
        for col in categorical_cols
    }

    input_data = add_sidebar(step)

    with st.container():
        st.title("Sales Predictor")
        st.subheader("Change atmost 3 columns at a time to avoid lags or crash! enjoy ;)")
        st.write("forecasts future sales for Rohik store using past data and trends")

    col1, col2 = st.columns([4, 1])

    with col1:
        view_interactive_visualization(input_data, category_maps)

    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
