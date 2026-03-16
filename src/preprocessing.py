import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess(filepath='data/retail_price_dataset.csv'):
    df = pd.read_csv(filepath)

    # Encode categoricals
    le_product = LabelEncoder()
    le_category = LabelEncoder()
    df['product_id_enc'] = le_product.fit_transform(df['product_id'].astype(str))
    df['product_category_enc'] = le_category.fit_transform(df['product_category_name'].astype(str))

    # Competitor features
    df['avg_comp_price'] = df[['comp_1', 'comp_2', 'comp_3']].mean(axis=1)
    df['min_comp_price'] = df[['comp_1', 'comp_2', 'comp_3']].min(axis=1)
    df['avg_comp_score'] = df[['ps1', 'ps2', 'ps3']].mean(axis=1)
    df['price_vs_avg_comp'] = df['unit_price'] - df['avg_comp_price']
    df['price_vs_min_comp'] = df['unit_price'] - df['min_comp_price']

    # Interaction features
    df['price_score'] = df['unit_price'] * df['product_score']
    df['price_volume'] = df['unit_price'] * df['volume']

    feature_cols = [
        'unit_price', 'product_name_lenght', 'product_description_lenght',
        'product_photos_qty', 'product_weight_g', 'product_score', 'customers',
        'weekday', 'weekend', 'holiday', 'month', 'year', 'volume',
        'avg_comp_price', 'min_comp_price', 'avg_comp_score',
        'price_vs_avg_comp', 'price_vs_min_comp', 'price_score', 'price_volume',
        'product_id_enc', 'product_category_enc'
    ]

    df_model = df[feature_cols + ['qty']].copy()
    df_model.dropna(inplace=True)

    # Remove extreme qty outliers
    q99 = df_model['qty'].quantile(0.99)
    df_model = df_model[df_model['qty'] <= q99]

    # Save encoders info for app use
    product_map = dict(zip(df['product_id'].astype(str), df['product_id_enc']))
    category_map = dict(zip(df['product_category_name'].astype(str), df['product_category_enc']))

    return df_model, feature_cols, product_map, category_map, df

if __name__ == '__main__':
    df_model, feature_cols, product_map, category_map, df_raw = load_and_preprocess()
    df_model.to_csv('data/processed_data.csv', index=False)
    print(f"Saved processed_data.csv — shape: {df_model.shape}")
    print("Features:", feature_cols)
