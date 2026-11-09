from pathlib import Path
import sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW,CLEAN=ROOT/"data/raw",ROOT/"data/clean"
DB=ROOT/"data/instamart.db"

def clean_data():
    customers=pd.read_csv(RAW/"customers.csv")
    products=pd.read_csv(RAW/"products.csv")
    orders=pd.read_csv(RAW/"orders.csv")
    tickets=pd.read_csv(RAW/"tickets.csv")
    customers["city"]=customers["city"].astype(str).str.strip().str.title()
    customers["age"]=pd.to_numeric(customers["age"],errors="coerce")
    customers.loc[~customers["age"].between(18,100),"age"]=pd.NA
    customers["gender"]=customers["gender"].astype("string").str.strip().str.lower().replace({"m":"male","f":"female","":pd.NA,"nan":pd.NA})
    orders["order_ts"]=pd.to_datetime(orders["order_ts"],errors="coerce")
    for col in ["quantity","discount_pct","order_amount","delivery_minutes"]:
        orders[col]=pd.to_numeric(orders[col],errors="coerce")
    orders=orders.sort_values(["order_id","order_ts"]).drop_duplicates("order_id",keep="last")
    orders=orders[orders["status"].isin(["completed","cancelled"])]
    orders=orders[orders["order_amount"].notna()&(orders["order_amount"]>0)]
    orders=orders[orders["quantity"].between(1,20)]
    orders.loc[orders["delivery_minutes"]<0,"delivery_minutes"]=pd.NA
    orders.loc[~orders["delivery_minutes"].between(1,180),"delivery_minutes"]=pd.NA
    orders.loc[orders["status"]=="cancelled","delivery_minutes"]=pd.NA
    orders=orders.dropna(subset=["order_ts","customer_id","product_id"])
    orders["order_amount"]=orders["order_amount"].round(2)
    tickets["text"]=tickets["text"].fillna("").astype(str)
    tickets["true_category"]=tickets["true_category"].fillna("none").astype(str).str.lower()
    tickets["customer_id"]=tickets["customer_id"].astype(str)
    CLEAN.mkdir(parents=True,exist_ok=True)
    for name,df in [("customers_clean.csv",customers),("products_clean.csv",products),("orders_clean.csv",orders),("tickets_clean.csv",tickets)]:
        df.to_csv(CLEAN/name,index=False)
    return customers,products,orders,tickets

def build_db():
    customers,products,orders,tickets=clean_data()
    DB.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(DB) as con:
        for name,df in [("customers",customers),("products",products),("orders",orders),("tickets",tickets)]:
            df.to_sql(name,con,if_exists="replace",index=False)
        con.executescript("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id); CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id); CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);")
    print(f"SQLite database created: {DB}")
    print(f"Rows: customers={len(customers)}, products={len(products)}, orders={len(orders)}, tickets={len(tickets)}")
if __name__=="__main__": build_db()
