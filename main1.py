import pandas as pd
import numpy as np


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != "category":
            if col_type == 'int64':
                df[col] = df[col].astype('int32')
            elif col_type == 'float64':
                df[col] = df[col].astype('float32')
    return df 


df = pd.read_csv('data/sales_train.csv')
calendar = pd.read_csv('data/calendar.csv')
inventory = pd.read_csv('data/inventory.csv')
print(df.columns)         # Your main dataset
print(calendar.columns)   # Calendar dataset
print(inventory.columns)
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
numerical_cols = df_test.select_dtypes(include=['number']).columns.tolist()

# Categorical columns (object, category)
categorical_cols = df_test.select_dtypes(include=['object', 'category']).columns.tolist()

print(df.columns)         # Your main dataset
print(calendar.columns)   # Calendar dataset
print(inventory.columns) 
#print("Numerical columns:", numerical_cols)
#print("Categorical columns:", categorical_cols)
#print(df_test.info())
#print(df_test['discounts'].value_counts())
#print(calendar['holiday_name'].head(10))
#print(calendar['holiday_name'].isna().mean())  # % of NaNs

