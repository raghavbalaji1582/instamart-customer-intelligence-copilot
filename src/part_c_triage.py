from pathlib import Path
import json,re,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mock_llm
CATEGORIES={"billing","delivery","technical","fraud"}
SYSTEM_ZERO="You are a support-ticket classifier. CLASSIFY. Return only JSON with category and reason."
SYSTEM_FEW="""You are a support-ticket classifier. CLASSIFY.
EXAMPLES:
- charged twice for my invoice => billing
- rider marked delivered but not received => delivery
- app crashes and OTP fails => technical
- unauthorized card transaction => fraud
Return only JSON with category and reason."""
EMAIL=re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",re.I)
PHONE=re.compile(r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)")
def redact(text):
    return PHONE.sub("[PHONE_REDACTED]",EMAIL.sub("[EMAIL_REDACTED]",str(text)))
def classify(text,few_shot=True,max_retries=2):
    safe=redact(text); system=SYSTEM_FEW if few_shot else SYSTEM_ZERO
    attempts=tokens=0
    while attempts<=max_retries:
        attempts+=1
        try:
            response=mock_llm.chat([{"role":"system","content":system},{"role":"user","content":safe}])
            tokens+=response.usage.total_tokens
            raw=response.choices[0].message.content or ""; data=json.loads(raw)
            if data.get("category") not in CATEGORIES or not isinstance(data.get("reason"),str): raise ValueError("invalid category/schema")
            return {"category":data["category"],"reason":data["reason"],"status":"classified","attempts":attempts,"tokens":tokens,"redacted_text":safe}
        except (mock_llm.TransientLLMError,json.JSONDecodeError,ValueError,TypeError):
            continue
    return {"category":"human_review","reason":"LLM response failed validation after retries","status":"fallback","attempts":attempts,"tokens":tokens,"redacted_text":safe}
def run():
    tickets=pd.read_csv(ROOT/"data/clean/tickets_clean.csv").fillna("")
    mock_llm.reset(); results=[]
    for _,row in tickets.iterrows():
        text=str(row["text"])
        if not text.strip():
            results.append({"ticket_id":row.ticket_id,"predicted_category":"human_review","true_category":row.true_category,"status":"empty_ticket","attempts":0,"tokens":0,"priority":"normal"}); continue
        r=classify(text,True)
        results.append({"ticket_id":row.ticket_id,"predicted_category":r["category"],"true_category":row.true_category,"status":r["status"],"attempts":r["attempts"],"tokens":r["tokens"],"priority":"high" if r["category"]=="fraud" else "normal","redacted_text":r["redacted_text"]})
    out=pd.DataFrame(results); out.to_csv(ROOT/"outputs/triage_results.csv",index=False)
    budget,used,demo=1800,0,[]
    for _,row in tickets.iterrows():
        estimate=mock_llm.est_tokens(SYSTEM_FEW+redact(str(row.text)))
        if used+estimate>budget: demo.append({"ticket_id":row.ticket_id,"action":"budget_deferred","estimated_tokens":estimate})
        else: used+=estimate; demo.append({"ticket_id":row.ticket_id,"action":"processed","estimated_tokens":estimate})
    pd.DataFrame(demo).to_csv(ROOT/"outputs/triage_budget_demo.csv",index=False)
    labelled=out[(out.true_category!="none")&(out.status=="classified")]
    accuracy=(labelled.predicted_category==labelled.true_category).mean() if len(labelled) else 0
    (ROOT/"outputs/triage_metrics.txt").write_text(f"Tickets processed: {len(out)}\nClassified: {(out.status=='classified').sum()}\nFallback/empty: {(out.status!='classified').sum()}\nAccuracy on classified labelled tickets: {accuracy:.3f}\nBudget: {budget} estimated tokens; used: {used}\n",encoding="utf-8")
    print(f"Ticket triage complete: {len(out)} tickets; classified-label accuracy={accuracy:.3f}; token budget used={used}/{budget}.")
if __name__=="__main__": run()
