
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

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
    return train



def rmsle(y_test, y_preds):
    """Calculates root mean squared log error between predictions and true labels."""
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few levels
def show_scores(model, X_train, y_train, X_valid, y_valid):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {
        "Training MAE": mean_absolute_error(y_train, train_preds),
        "Valid MAE": mean_absolute_error(y_valid, val_preds),
        "Training RMSLE": rmsle(y_train, train_preds),
        "Valid RMSLE": rmsle(y_valid, val_preds),
        "Training R^2": r2_score(y_train, train_preds),
        "Valid R^2": r2_score(y_valid, val_preds)
    }
    return scores

def create_model(data):
    target_col = "sales"

    # Split early
    val_split = data[data.year == 2024]
    train_split = data[data.year != 2024]

    # Separate X and y
    X_train_raw = train_split.drop(columns=[target_col])
    y_train = train_split[target_col]

    X_valid_raw = val_split.drop(columns=[target_col])
    y_valid = val_split[target_col]

    # Identify columns
    categorical_cols = X_train_raw.select_dtypes(include='object').columns.tolist()
    numerical_cols = X_train_raw.select_dtypes(exclude='object').columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    X_train = preprocessor.fit_transform(X_train_raw)
    X_valid = preprocessor.transform(X_valid_raw)

    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "| y_valid:", y_valid.shape)

    print("now fitting...")
    ideal_model = RandomForestRegressor(
        n_estimators=40,
        min_samples_leaf=1,
        min_samples_split=14,
        max_features=0.5,
        n_jobs=-1,
        max_samples=700000,
        random_state=42
    )

    # Fit the model
    ideal_model.fit(X_train, y_train)

    # Evaluate the model
    scores = show_scores(ideal_model, X_train, y_train, X_valid, y_valid)
    print(scores)

    # Return model and preprocessor for future use
    return ideal_model, preprocessor




def main():
    data = get_clean_data()
    print(data.info())

    # Unpack the returned model and preprocessor
    ideal_model, preprocessor = create_model(data)

    # Save model and preprocessor
    joblib.dump(ideal_model, 'random_forest_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    print("Model and preprocessor saved successfully!")

if __name__ == '__main__':
    main()