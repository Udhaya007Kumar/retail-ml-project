# ============================================================
# STEP 2: FEATURE ENGINEERING  ⚙️
# ============================================================
# என்ன பண்றோம்:
#   - 6 clean tables-ஐ ஒரே big table-ஆ merge பண்றோம்
#   - புது useful columns உருவாக்குறோம்
#   - Text columns → Numbers-ஆ மாத்துறோம் (ML-க்கு)
#   - Regression target  : Sales (number predict)
#   - Classification target: Sales_Category (High/Low predict)
#   - Final dataset save பண்றோம்
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder   # Text → Number
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: PATHS
# ============================================================

PROCESSED_PATH = "data/processed/"
FINAL_PATH     = "data/processed/"

print("=" * 60)
print("⚙️  FEATURE ENGINEERING STARTED")
print("=" * 60)


# ============================================================
# PART 2: CLEAN DATA LOAD பண்றோம்
# ============================================================

def load_cleaned_data():
    """
    Step 1-ல save பண்ணிய clean files படிக்கிறோம்.
    """
    print("\n📂 Loading cleaned data...")

    data = {}
    data['sales']      = pd.read_csv(PROCESSED_PATH + "sales_cleaned.csv")
    data['products']   = pd.read_csv(PROCESSED_PATH + "products_cleaned.csv")
    data['stores']     = pd.read_csv(PROCESSED_PATH + "stores_cleaned.csv")
    data['inventory']  = pd.read_csv(PROCESSED_PATH + "inventory_cleaned.csv")
    data['promotions'] = pd.read_csv(PROCESSED_PATH + "promotions_cleaned.csv")
    data['holidays']   = pd.read_csv(PROCESSED_PATH + "holidays_cleaned.csv")

    for name, df in data.items():
        print(f"   ✅ {name:12s} → {df.shape[0]:,} rows, {df.shape[1]} columns")

    return data


# ============================================================
# PART 3: TABLES MERGE பண்றோம்
# ============================================================
# merge() என்னன்னா: 2 tables-ஐ common column வச்சு join பண்றது
# SQL-ல இதை JOIN னு சொல்வாங்க
#
# Example:
#   sales table:    Sale_ID, Product_ID, Sales
#   products table: Product_ID, Category, Price
#   merge பண்ணா:    Sale_ID, Product_ID, Sales, Category, Price
# ============================================================

def merge_all_tables(data):
    """
    6 tables-ஐ ஒரே DataFrame-ஆ merge பண்றோம்.
    """
    print("\n🔗 Merging all tables...")

    # Base: sales table (50,000 rows — இதுதான் main table)
    df = data['sales'].copy()
    print(f"   Base (sales)     : {len(df):,} rows")

    # --- Merge 1: Sales + Products ---
    # Product_ID common column — இதை வச்சு join பண்றோம்
    # how='left' → sales-ல உள்ள எல்லா rows keep பண்றோம்
    df = df.merge(
        data['products'][['Product_ID', 'Category', 'Brand',
                          'Price', 'Profit_Margin', 'Price_Range']],
        on='Product_ID',
        how='left'
    )
    print(f"   + Products merged : {len(df):,} rows, {len(df.columns)} cols")

    # --- Merge 2: + Stores ---
    df = df.merge(
        data['stores'][['Store_ID', 'City', 'State', 'Region',
                        'Store_Type', 'Store_Age_Years']],
        on='Store_ID',
        how='left'
    )
    print(f"   + Stores merged  : {len(df):,} rows, {len(df.columns)} cols")

    # --- Merge 3: + Holidays ---
    # Date column-ஐ வச்சு join — இந்த date holiday-ஆ இல்லையான்னு தெரியும்
    # sales Date → date only (time remove பண்றோம்)
    df['Date_only'] = pd.to_datetime(df['Date']).dt.date.astype(str)
    data['holidays']['Date_only'] = pd.to_datetime(
        data['holidays']['Date']).dt.date.astype(str)

    df = df.merge(
        data['holidays'][['Date_only', 'Is_Holiday', 'Holiday_Type']],
        on='Date_only',
        how='left'
    )

    # Holiday இல்லா rows-ல NaN வரும் → 0 போடுறோம்
    df['Is_Holiday']   = df['Is_Holiday'].fillna(0).astype(int)
    df['Holiday_Type'] = df['Holiday_Type'].fillna('No Holiday')
    print(f"   + Holidays merged: {len(df):,} rows, {len(df.columns)} cols")

    # --- Merge 4: + Promotions ---
    # Product-க்கு promotion இருக்கான்னு check பண்றோம்
    # ஒரு product-க்கு multiple promotions இருக்கலாம்
    # → max discount எடுக்குறோம்
    promo_agg = data['promotions'].groupby('Product_ID').agg(
        Has_Promotion=('Promo_ID', 'count'),
        Max_Discount=('Discount_Percentage', 'max'),
        Avg_Discount=('Discount_Percentage', 'mean')
    ).reset_index()

    df = df.merge(promo_agg, on='Product_ID', how='left')

    # Promotion இல்லா products → 0
    df['Has_Promotion'] = df['Has_Promotion'].fillna(0).astype(int)
    df['Max_Discount']  = df['Max_Discount'].fillna(0)
    df['Avg_Discount']  = df['Avg_Discount'].fillna(0)
    print(f"   + Promotions merged: {len(df):,} rows, {len(df.columns)} cols")

    # --- Merge 5: + Inventory ---
    # Store + Product combo வச்சு stock பாக்குறோம்
    inv_agg = data['inventory'].groupby(['Product_ID', 'Store_ID']).agg(
        Total_Stock=('Stock', 'sum'),
        Avg_Stock=('Stock', 'mean')
    ).reset_index()

    df = df.merge(inv_agg, on=['Product_ID', 'Store_ID'], how='left')
    df['Total_Stock'] = df['Total_Stock'].fillna(0)
    df['Avg_Stock']   = df['Avg_Stock'].fillna(0)
    print(f"   + Inventory merged: {len(df):,} rows, {len(df.columns)} cols")

    return df


# ============================================================
# PART 4: புது FEATURES உருவாக்குறோம்
# ============================================================

def create_new_features(df):
    """
    Existing columns-ல இருந்து புது useful columns உருவாக்குறோம்.
    இந்த columns ML model-க்கு மிகவும் helpful.
    """
    print("\n✨ Creating new features...")

    # --- Feature 1: Revenue per Customer ---
    # ஒவ்வொரு customer எவ்வளவு spend பண்றாங்கன்னு தெரியும்
    df['Revenue_Per_Customer'] = (
        df['Sales'] / df['Customer_Count'].replace(0, 1)
    ).round(2)

    # --- Feature 2: Sales per Quantity ---
    # ஒவ்வொரு item average price என்ன
    df['Avg_Price_Per_Unit'] = (
        df['Sales'] / df['Quantity'].replace(0, 1)
    ).round(2)

    # --- Feature 3: High Sales Day flag ---
    # Average-ஐ விட அதிகமா sales இருந்தா = High Sales Day
    avg_sales = df['Sales'].mean()
    df['Is_High_Sales_Day'] = (df['Sales'] > avg_sales).astype(int)

    # --- Feature 4: Promotion Impact ---
    # Discount இருக்கும்போது sales எவ்வளவு
    df['Discount_Sales_Ratio'] = (
        df['Max_Discount'] / df['Sales'].replace(0, 1)
    ).round(4)

    # --- Feature 5: Stock Availability flag ---
    # Stock இருக்கான்னு simple flag
    df['Stock_Available'] = (df['Total_Stock'] > 0).astype(int)

    # --- Feature 6: Month Name ---
    month_map = {
        1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',
        5:'May', 6:'Jun', 7:'Jul', 8:'Aug',
        9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'
    }
    df['Month_Name'] = df['Month'].map(month_map)

    print(f"   ✅ New features added. Total columns: {len(df.columns)}")
    return df


# ============================================================
# PART 5: TARGET VARIABLES உருவாக்குறோம்
# ============================================================
# ML-ல target variable = நாம் predict பண்ண விரும்புவது
#
# Model 1 (Regression)     → Sales amount predict (number)
# Model 2 (Classification) → High / Low sales predict (category)
# Model 3 (Clustering)     → target இல்லை (unsupervised)
# ============================================================

def create_target_variables(df):
    """
    ML models-க்கு target columns உருவாக்குறோம்.
    """
    print("\n🎯 Creating target variables...")

    # --- Target 1: Regression → Sales (already exists) ---
    # Sales column-ஐ predict பண்றோம் (இது already இருக்கு)
    print(f"   Regression target  : 'Sales' (min={df['Sales'].min():,}, max={df['Sales'].max():,})")

    # --- Target 2: Classification → Sales_Category ---
    # Sales-ஐ 3 categories-ஆ பிரிக்குறோம்
    # pd.qcut() → equal size groups-ஆ பிரிக்கும்
    df['Sales_Category'] = pd.qcut(
        df['Sales'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # Value counts பாக்குறோம்
    counts = df['Sales_Category'].value_counts()
    print(f"   Classification target: 'Sales_Category'")
    for cat, cnt in counts.items():
        print(f"      {cat:8s} → {cnt:,} rows")

    return df


# ============================================================
# PART 6: ENCODING — Text → Numbers மாத்துறோம்
# ============================================================
# ML algorithms numbers மட்டுமே புரியும்!
# "Chennai" → 2, "Mumbai" → 5 மாதிரி மாத்தணும்
#
# LabelEncoder:
#   fit()       → unique values கத்துக்கும்
#   transform() → numbers-ஆ மாத்தும்
# ============================================================

def encode_categorical_columns(df):
    """
    Text columns-ஐ numbers-ஆ மாத்துறோம்.
    Original column-ம் keep பண்றோம், புது _Encoded column உருவாக்குறோம்.
    """
    print("\n🔢 Encoding categorical columns...")

    # இந்த columns encode பண்ணணும்
    categorical_cols = [
        'Payment_Method',   # Cash, Card, UPI → 0, 1, 2
        'Season',           # Winter, Summer → 0, 1, 2, 3
        'Category',         # Food, Sports → 0, 1, 2...
        'Brand',            # Nike, Boat → 0, 1, 2...
        'Price_Range',      # Budget, Premium → 0, 1, 2, 3
        'City',             # Chennai, Mumbai → 0, 1...
        'Region',           # North, South → 0, 1, 2, 3
        'Store_Type',       # Mall, Supermarket → 0, 1...
        'Holiday_Type',     # Religious, National → 0, 1...
        'Promotion_Type' if 'Promotion_Type' in df.columns else None,
        'Stock_Status' if 'Stock_Status' in df.columns else None,
    ]

    # None values filter பண்றோம்
    categorical_cols = [c for c in categorical_cols if c is not None and c in df.columns]

    le = LabelEncoder()

    for col in categorical_cols:
        # New column name: Category → Category_Encoded
        new_col = col + '_Encoded'

        # NaN values → 'Unknown' replace பண்றோம்
        df[col] = df[col].fillna('Unknown').astype(str)

        # Encode!
        df[new_col] = le.fit_transform(df[col])
        print(f"   ✅ {col:20s} → {new_col} ({df[new_col].nunique()} unique values)")

    # Sales_Category encode பண்றோம் (Classification target)
    cat_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Sales_Category_Encoded'] = df['Sales_Category'].map(cat_map)
    print(f"   ✅ Sales_Category → Sales_Category_Encoded (0=Low, 1=Medium, 2=High)")

    return df


# ============================================================
# PART 7: FINAL DATASET SAVE பண்றோம்
# ============================================================

def save_final_dataset(df):
    """
    Final merged + engineered dataset save பண்றோம்.
    """
    print("\n💾 Saving final dataset...")

    # Temp columns remove பண்றோம்
    cols_to_drop = ['Date_only']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final file save
    save_path = FINAL_PATH + "final_dataset.csv"
    df.to_csv(save_path, index=False)

    print(f"   ✅ Saved: {save_path}")
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {len(df.columns)}")

    return df


# ============================================================
# PART 8: FEATURE LIST PRINT பண்றோம்
# ============================================================

def print_feature_summary(df):
    """
    ML-ல use ஆகற features list காட்டுறோம்.
    """
    print("\n" + "=" * 60)
    print("📊 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    # Regression features
    reg_features = [
        'Month', 'DayOfWeek', 'Quarter', 'Is_Weekend', 'Is_Holiday',
        'Season_Encoded', 'Category_Encoded', 'Brand_Encoded',
        'Store_Type_Encoded', 'Region_Encoded', 'Price_Range_Encoded',
        'Payment_Method_Encoded', 'Has_Promotion', 'Max_Discount',
        'Store_Age_Years', 'Total_Stock', 'Customer_Count',
        'Profit_Margin', 'Revenue_Per_Customer'
    ]

    print("\n  📈 Regression Features (predict Sales amount):")
    for f in reg_features:
        if f in df.columns:
            print(f"     ✅ {f}")

    print("\n  🏷️  Classification Target:")
    print(f"     ✅ Sales_Category_Encoded (0=Low, 1=Medium, 2=High)")

    print("\n  🔵 Clustering: same features (no target needed)")

    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("   Next step → Run: python src/train_models.py")
    print("=" * 60)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # Step 1: Load clean data
    data = load_cleaned_data()

    # Step 2: Merge all tables
    df = merge_all_tables(data)

    # Step 3: Create new features
    df = create_new_features(df)

    # Step 4: Create target variables
    df = create_target_variables(df)

    # Step 5: Encode categorical columns
    df = encode_categorical_columns(df)

    # Step 6: Save final dataset
    df = save_final_dataset(df)

    # Step 7: Print summary
    print_feature_summary(df)


if __name__ == "__main__":
    main()
