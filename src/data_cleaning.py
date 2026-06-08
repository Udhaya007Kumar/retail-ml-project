# ============================================================
# STEP 1: DATA CLEANING  🧹
# ============================================================
# என்ன பண்றோம்:
#   - 6 CSV files படிக்கிறோம் (read)
#   - Date columns → datetime format மாத்துறோம்
#   - Duplicates remove பண்றோம்
#   - Data types சரி பண்றோம்
#   - Extra features உருவாக்குறோம் (month, day, season)
#   - Clean data → processed/ folder-ல save பண்றோம்
# ============================================================

import pandas as pd       # Data handle பண்ண
import numpy as np        # Math calculations-க்கு
import os                 # Folder create பண்ண
import warnings
warnings.filterwarnings('ignore')  # Warning messages hide பண்ண

# ============================================================
# PART 1: FILE PATHS DEFINE பண்றோம்
# ============================================================

# Raw data இருக்கற இடம்
RAW_PATH = "data/raw/"

# Clean data save பண்ண இடம்
PROCESSED_PATH = "data/processed/"

# Processed folder இல்லன்னா create பண்றோம்
os.makedirs(PROCESSED_PATH, exist_ok=True)

print("=" * 60)
print("🧹 DATA CLEANING STARTED")
print("=" * 60)


# ============================================================
# PART 2: DATA LOAD பண்றோம்
# ============================================================
# pandas read_csv() → CSV file-ஐ table (DataFrame) ஆக படிக்கும்

def load_all_data():
    """
    எல்லா CSV files-ஐயும் படிக்கும் function.
    Return: dictionary with all dataframes
    """
    print("\n📂 Loading all CSV files...")

    data = {}

    # ஒவ்வொரு file-ம் load பண்றோம்
    data['sales']      = pd.read_csv(RAW_PATH + "sales_fact_50000.csv")
    data['products']   = pd.read_csv(RAW_PATH + "products_500.csv")
    data['stores']     = pd.read_csv(RAW_PATH + "stores_100.csv")
    data['inventory']  = pd.read_csv(RAW_PATH + "inventory_10000.csv")
    data['promotions'] = pd.read_csv(RAW_PATH + "promotions_1000.csv")
    data['holidays']   = pd.read_csv(RAW_PATH + "holidays_120.csv")

    # எத்தனை rows load ஆச்சுன்னு print பண்றோம்
    for name, df in data.items():
        print(f"   ✅ {name:12s} → {df.shape[0]:,} rows, {df.shape[1]} columns")

    return data


# ============================================================
# PART 3: BASIC INFO CHECK பண்றோம்
# ============================================================

def check_data_quality(df, name):
    """
    Data-ல என்னென்ன problems இருக்குன்னு பாக்கும்.
    - Null values எத்தனை?
    - Duplicates எத்தனை?
    """
    print(f"\n🔍 Checking: {name}")

    # Null values count
    nulls = df.isnull().sum().sum()
    print(f"   Null values   : {nulls}")

    # Duplicate rows count
    dupes = df.duplicated().sum()
    print(f"   Duplicates    : {dupes}")

    # Total rows
    print(f"   Total rows    : {len(df):,}")


# ============================================================
# PART 4: SALES DATA CLEAN பண்றோம்
# ============================================================

def clean_sales(df):
    """
    Sales data-ஐ clean பண்றோம்.
    Sales data மிக முக்கியம் — இதுதான் நம்ம target variable இருக்கற table.
    """
    print("\n🛒 Cleaning Sales Data...")

    # --- Step 4.1: Date column fix ---
    # "2023-01-01 00:00:00" string → proper datetime object
    # pd.to_datetime() → string-ஐ date format-ல மாத்தும்
    df['Date'] = pd.to_datetime(df['Date'])

    # --- Step 4.2: Date-ல இருந்து புது columns உருவாக்குறோம் ---
    # இந்த columns feature engineering-ல மிகவும் useful

    df['Year']    = df['Date'].dt.year         # 2023, 2024
    df['Month']   = df['Date'].dt.month        # 1 to 12
    df['Day']     = df['Date'].dt.day          # 1 to 31
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Quarter'] = df['Date'].dt.quarter      # 1, 2, 3, 4

    # --- Step 4.3: Weekend flag ---
    # Saturday(5), Sunday(6) → weekend = 1, மத்தது = 0
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # --- Step 4.4: Season column ---
    # Month பார்த்து season decide பண்றோம் (India-க்கு ஏத்த மாதிரி)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'    # குளிர்காலம்
        elif month in [3, 4, 5]:
            return 'Summer'    # கோடைகாலம்
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'   # மழைக்காலம்
        else:
            return 'Autumn'    # இலையுதிர்காலம்

    df['Season'] = df['Month'].apply(get_season)

    # --- Step 4.5: Duplicates remove ---
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"   Duplicates removed: {before - after}")

    # --- Step 4.6: Negative values check ---
    # Sales, Quantity negative இருக்கக்கூடாது
    df = df[df['Sales'] >= 0]
    df = df[df['Quantity'] >= 0]

    print(f"   ✅ Sales cleaned: {len(df):,} rows")
    return df


# ============================================================
# PART 5: PRODUCTS DATA CLEAN பண்றோம்
# ============================================================

def clean_products(df):
    """
    Products data clean பண்றோம்.
    Price, Cost_Price இருக்கு — இதுல இருந்து Profit calculate பண்ணலாம்.
    """
    print("\n📦 Cleaning Products Data...")

    # --- Step 5.1: Profit Margin calculate ---
    # Profit = Price - Cost_Price
    # Margin % = (Profit / Price) * 100
    df['Profit']        = df['Price'] - df['Cost_Price']
    df['Profit_Margin'] = ((df['Profit'] / df['Price']) * 100).round(2)

    # --- Step 5.2: Price Range category ---
    # pd.cut() → number-ஐ category-ஆ மாத்தும்
    df['Price_Range'] = pd.cut(
        df['Price'],
        bins=[0, 500, 2000, 10000, float('inf')],
        labels=['Budget', 'Mid-range', 'Premium', 'Luxury']
    )

    # --- Step 5.3: Category uppercase normalize ---
    df['Category'] = df['Category'].str.strip().str.title()
    df['Brand']    = df['Brand'].str.strip().str.title()

    # --- Step 5.4: Duplicates remove ---
    df = df.drop_duplicates(subset=['Product_ID'])

    print(f"   ✅ Products cleaned: {len(df):,} rows")
    return df


# ============================================================
# PART 6: STORES DATA CLEAN பண்றோம்
# ============================================================

def clean_stores(df):
    """
    Stores data clean பண்றோம்.
    Store எத்தனை வருஷமா run ஆகுதுன்னு calculate பண்ணலாம்.
    """
    print("\n🏪 Cleaning Stores Data...")

    # --- Step 6.1: Opening Date fix ---
    df['Opening_Date'] = pd.to_datetime(df['Opening_Date'])

    # --- Step 6.2: Store Age calculate ---
    # 2024 - opening year = store எத்தனை வருஷம் பழசு
    current_year = 2024
    df['Store_Age_Years'] = current_year - df['Opening_Date'].dt.year

    # --- Step 6.3: Text normalize ---
    df['City']       = df['City'].str.strip().str.title()
    df['State']      = df['State'].str.strip().str.title()
    df['Region']     = df['Region'].str.strip().str.title()
    df['Store_Type'] = df['Store_Type'].str.strip().str.title()

    # --- Step 6.4: Duplicates remove ---
    df = df.drop_duplicates(subset=['Store_ID'])

    print(f"   ✅ Stores cleaned: {len(df):,} rows")
    return df


# ============================================================
# PART 7: INVENTORY DATA CLEAN பண்றோம்
# ============================================================

def clean_inventory(df):
    """
    Inventory data clean பண்றோம்.
    Stock level பார்த்து status decide பண்ணலாம்.
    """
    print("\n📋 Cleaning Inventory Data...")

    # --- Step 7.1: Date fix ---
    df['Last_Updated'] = pd.to_datetime(df['Last_Updated'])

    # --- Step 7.2: Stock Status ---
    # Stock < Reorder_Level → Low Stock warning
    def stock_status(row):
        if row['Stock'] == 0:
            return 'Out of Stock'      # Stock இல்லை
        elif row['Stock'] < row['Reorder_Level']:
            return 'Low Stock'         # குறைவா இருக்கு
        else:
            return 'Adequate'          # போதுமான அளவு இருக்கு

    df['Stock_Status'] = df.apply(stock_status, axis=1)

    # --- Step 7.3: Duplicates ---
    df = df.drop_duplicates()

    print(f"   ✅ Inventory cleaned: {len(df):,} rows")
    return df


# ============================================================
# PART 8: PROMOTIONS DATA CLEAN பண்றோம்
# ============================================================

def clean_promotions(df):
    """
    Promotions data clean பண்றோம்.
    Promotion எத்தனை நாள் run ஆச்சுன்னு calculate பண்ணலாம்.
    """
    print("\n🎯 Cleaning Promotions Data...")

    # --- Step 8.1: Date fix ---
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df['End_Date']   = pd.to_datetime(df['End_Date'])

    # --- Step 8.2: Promo Duration ---
    # End - Start = எத்தனை நாள் promotion இருந்துச்சு
    df['Promo_Duration_Days'] = (df['End_Date'] - df['Start_Date']).dt.days

    # --- Step 8.3: Promotion Type normalize ---
    df['Promotion_Type'] = df['Promotion_Type'].str.strip().str.title()

    # --- Step 8.4: Duplicates ---
    df = df.drop_duplicates()

    print(f"   ✅ Promotions cleaned: {len(df):,} rows")
    return df


# ============================================================
# PART 9: HOLIDAYS DATA CLEAN பண்றோம்
# ============================================================

def clean_holidays(df):
    """
    Holidays data clean பண்றோம்.
    """
    print("\n🎉 Cleaning Holidays Data...")

    # --- Step 9.1: Date fix ---
    df['Date'] = pd.to_datetime(df['Date'])

    # --- Step 9.2: Normalize text ---
    df['Holiday_Type'] = df['Holiday_Type'].str.strip().str.title()
    df['Region']       = df['Region'].str.strip().str.title()

    # --- Step 9.3: Duplicates ---
    df = df.drop_duplicates()

    print(f"   ✅ Holidays cleaned: {len(df):,} rows")
    return df


# ============================================================
# PART 10: DATA SAVE பண்றோம்
# ============================================================

def save_cleaned_data(data_dict):
    """
    Clean பண்ணிய data-ஐ processed/ folder-ல save பண்றோம்.
    index=False → row numbers save ஆகாது
    """
    print("\n💾 Saving cleaned data...")

    for name, df in data_dict.items():
        save_path = PROCESSED_PATH + f"{name}_cleaned.csv"
        df.to_csv(save_path, index=False)
        print(f"   ✅ Saved: {save_path}  ({len(df):,} rows)")


# ============================================================
# PART 11: SUMMARY REPORT
# ============================================================

def print_summary(data_dict):
    """
    Cleaning முடிஞ்சதும் summary print பண்றோம்.
    """
    print("\n" + "=" * 60)
    print("📊 CLEANING SUMMARY REPORT")
    print("=" * 60)

    for name, df in data_dict.items():
        print(f"\n  📁 {name.upper()}")
        print(f"     Rows    : {len(df):,}")
        print(f"     Columns : {len(df.columns)}")
        print(f"     Columns : {list(df.columns)}")

    print("\n" + "=" * 60)
    print("✅ DATA CLEANING COMPLETE!")
    print("   Next step → Run: python src/feature_engineering.py")
    print("=" * 60)


# ============================================================
# MAIN FUNCTION — இங்கே எல்லாம் call பண்றோம்
# ============================================================

def main():
    """
    Main function — இதை run பண்ணா எல்லாமே நடக்கும்.
    """

    # Step 1: Load data
    data = load_all_data()

    # Step 2: Quality check (before cleaning)
    print("\n" + "-" * 40)
    print("🔍 DATA QUALITY CHECK (Before Cleaning)")
    print("-" * 40)
    for name, df in data.items():
        check_data_quality(df, name)

    # Step 3: Clean each table
    print("\n" + "-" * 40)
    print("🧹 CLEANING EACH TABLE")
    print("-" * 40)

    data['sales']      = clean_sales(data['sales'])
    data['products']   = clean_products(data['products'])
    data['stores']     = clean_stores(data['stores'])
    data['inventory']  = clean_inventory(data['inventory'])
    data['promotions'] = clean_promotions(data['promotions'])
    data['holidays']   = clean_holidays(data['holidays'])

    # Step 4: Save clean data
    save_cleaned_data(data)

    # Step 5: Print summary
    print_summary(data)


# ============================================================
# PYTHON ENTRY POINT
# "python src/data_cleaning.py" னு run பண்ணும்போது
# automatically main() call ஆகும்
# ============================================================

if __name__ == "__main__":
    main()
