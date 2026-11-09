from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
def run():
    OUT.mkdir(exist_ok=True)
    orders=pd.read_csv(ROOT/"data/clean/orders_clean.csv",parse_dates=["order_ts"])
    customers=pd.read_csv(ROOT/"data/clean/customers_clean.csv")
    completed=orders[orders.status.eq("completed")].copy()
    ref_date=completed.order_ts.max()+pd.Timedelta(days=1)
    rfm=completed.groupby("customer_id").agg(recency=("order_ts",lambda s:(ref_date-s.max()).days),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm=customers[["customer_id","city"]].drop_duplicates().merge(rfm,on="customer_id",how="left")
    rfm[["recency","frequency","monetary"]]=rfm[["recency","frequency","monetary"]].fillna({"recency":999,"frequency":0,"monetary":0})
    Xraw=np.log1p(rfm[["recency","frequency","monetary"]])
    scaler=StandardScaler(); X=scaler.fit_transform(Xraw)
    rfm["cluster"]=KMeans(n_clusters=4,random_state=42,n_init=20).fit_predict(X)
    profile=rfm.groupby("cluster")[["recency","frequency","monetary"]].mean()
    labels={}
    for idx,row in profile.iterrows():
        if row.frequency>=profile.frequency.quantile(.75) and row.monetary>=profile.monetary.quantile(.75): labels[idx]="High Value"
        elif row.recency>=profile.recency.quantile(.75) or row.frequency==0: labels[idx]="At Risk"
        elif row.frequency>=profile.frequency.median(): labels[idx]="Loyal"
        else: labels[idx]="Occasional"
    rfm["segment"]=rfm.cluster.map(labels)
    rfm.to_csv(OUT/"customer_segments.csv",index=False)
    coords=PCA(n_components=2,random_state=42).fit_transform(X)
    pd.DataFrame({"customer_id":rfm.customer_id,"pc1":coords[:,0],"pc2":coords[:,1],"segment":rfm.segment}).to_csv(OUT/"pca_segments.csv",index=False)
    cutoff=completed.order_ts.max()-pd.Timedelta(days=30)
    last=completed.groupby("customer_id").order_ts.max()
    rfm["lapsed"]=rfm.customer_id.map(last).isna()|(rfm.customer_id.map(last)<cutoff)
    features=rfm[["recency","frequency","monetary"]].copy()
    features["avg_order_value"]=np.where(rfm.frequency>0,rfm.monetary/rfm.frequency,0)
    y=rfm.lapsed.astype(int)
    Xtr,Xte,ytr,yte=train_test_split(features,y,test_size=.25,random_state=42,stratify=y)
    model=RandomForestClassifier(n_estimators=250,max_depth=6,min_samples_leaf=4,class_weight="balanced",random_state=42)
    model.fit(Xtr,ytr); probs=model.predict_proba(Xte)[:,1]; pred=(probs>=.5).astype(int)
    try: auc=roc_auc_score(yte,probs)
    except ValueError: auc=float("nan")
    joblib.dump({"model":model,"features":list(features.columns)},OUT/"lapse_model.joblib")
    (OUT/"lapse_model_metrics.txt").write_text(f"ROC_AUC={auc:.4f}\n\n{classification_report(yte,pred,zero_division=0)}\nConfusion matrix:\n{confusion_matrix(yte,pred)}\n",encoding="utf-8")
    rfm["lapse_probability"]=model.predict_proba(features)[:,1]
    rfm.sort_values(["lapse_probability","monetary"],ascending=[False,False]).head(100).to_csv(OUT/"retention_list.csv",index=False)
    print(f"Customer segmentation complete: {len(rfm)} customers, 4 clusters.")
    print(f"Lapse model ROC-AUC: {auc:.3f}; retention list: 100 customers.")
if __name__=="__main__": run()
