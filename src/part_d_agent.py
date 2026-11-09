from pathlib import Path
import json,re,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mock_llm
POL=ROOT/"data/policies"
def load_policy_chunks():
    chunks=[]
    for path in sorted(POL.glob("*.txt")):
        for sentence in re.split(r"(?<=[.!?])\s+",path.read_text(encoding="utf-8").strip()):
            if sentence.strip(): chunks.append({"doc":path.name,"text":sentence.strip()})
    return chunks
def retrieve(question,chunks,top_k=4):
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus=[c["text"] for c in chunks]
    vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
    matrix=vec.fit_transform(corpus+[question]); scores=(matrix[:-1]@matrix[-1].T).toarray().ravel()
    return [chunks[i] for i in sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k] if scores[i]>0]
def policy_qa():
    chunks=load_policy_chunks(); eval_df=pd.read_csv(ROOT/"data/raw/policy_eval.csv").fillna(""); rows=[]
    for _,r in eval_df.iterrows():
        found=retrieve(r.question,chunks); context="\n".join(f"[{x['doc']}] {x['text']}" for x in found)
        messages=[{"role":"system","content":"You are a POLICY ASSISTANT. Answer only from supplied policy context; if unsupported, say you cannot find it."},{"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{r.question}"}]
        answer=mock_llm.chat(messages).choices[0].message.content
        doc_match=bool(r.expected_doc and any(x["doc"]==r.expected_doc for x in found))
        fact_match=bool(r.expected_fact and r.expected_fact.lower() in answer.lower())
        abstained="could not find" in answer.lower()
        correct=(doc_match and fact_match) if r.expected_doc else abstained
        rows.append({"question":r.question,"expected_doc":r.expected_doc,"retrieved_docs":";".join(x["doc"] for x in found),"answer":answer,"doc_match":doc_match,"fact_match":fact_match,"abstained":abstained,"correct":correct})
    pd.DataFrame(rows).to_csv(ROOT/"outputs/policy_qa_results.csv",index=False)
    acc=sum(x["correct"] for x in rows)/len(rows)
    (ROOT/"outputs/policy_qa_metrics.txt").write_text(f"Evaluation questions: {len(rows)}\nCorrect/grounded or abstained: {sum(x['correct'] for x in rows)}\nAccuracy: {acc:.3f}\n",encoding="utf-8")
    print(f"Policy RAG evaluation: {sum(x['correct'] for x in rows)}/{len(rows)} correct.")
def get_order(order_id):
    df=pd.read_csv(ROOT/"data/clean/orders_clean.csv"); hit=df[df.order_id.eq(order_id)]
    if hit.empty:return {"error":"order_not_found"}
    row=hit.iloc[0]
    return {k:(None if pd.isna(row[k]) else row[k].item() if hasattr(row[k],"item") else row[k]) for k in ["order_id","customer_id","status","order_amount","delivery_minutes"]}
def calculate_refund(order_amount,delivery_minutes):
    if delivery_minutes is None or pd.isna(delivery_minutes) or float(delivery_minutes)<=45:return {"refund_inr":0.0}
    return {"refund_inr":round(float(order_amount),2)}
def search_policy(query):
    return {"policy":"refund_policy.txt","relevant":(POL/"refund_policy.txt").read_text(encoding="utf-8")[:600]}
def issue_refund(order_id,refund_inr):
    entry={"order_id":order_id,"refund_inr":float(refund_inr),"status":"SIMULATED_ISSUED"}
    with (ROOT/"outputs/refund_ledger.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(entry)+"\n")
    return entry
def escalate_to_human(order_id,refund_inr):
    return {"order_id":order_id,"refund_inr":float(refund_inr),"status":"HUMAN_APPROVAL_REQUIRED"}
TOOLS={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
def run_agent(message,max_steps=6):
    messages=[{"role":"system","content":"You are an AGENT. Follow refund policy and use tools; never bypass eligibility or approval."},{"role":"user","content":message}]
    trace=[]; outcome="FALLBACK"
    for step in range(1,max_steps+1):
        response=mock_llm.chat(messages,tools=list(TOOLS)); msg=response.choices[0].message
        if response.choices[0].finish_reason!="tool_calls" or not msg.tool_calls:
            outcome=msg.content or "FALLBACK"; trace.append({"step":step,"assistant":outcome}); break
        call=msg.tool_calls[0]; name=call.name
        args=call.arguments if isinstance(call.arguments,dict) else json.loads(call.arguments)
        if name not in TOOLS: result={"error":"tool_not_allowed"}
        elif name=="issue_refund":
            order=get_order(args["order_id"])
            if "error" in order or order["status"]!="completed" or float(args["refund_inr"])>1000 or calculate_refund(order["order_amount"],order["delivery_minutes"])["refund_inr"]<=0:
                result={"error":"refund_not_permitted"}
            else: result=TOOLS[name](**args)
        else: result=TOOLS[name](**args)
        trace.append({"step":step,"tool":name,"arguments":args,"result":result})
        messages.append({"role":"assistant","content":None,"tool_calls":[{"id":call.id,"type":"function","function":{"name":name,"arguments":json.dumps(args)}}]})
        messages.append({"role":"tool","tool_call_id":call.id,"name":name,"content":json.dumps(result,default=str)})
    else: outcome="FALLBACK: maximum step limit reached; hand off to a human agent."
    return outcome,trace
def run():
    scenarios=pd.read_csv(ROOT/"data/raw/agent_scenarios.csv").fillna("")
    ledger=ROOT/"outputs/refund_ledger.jsonl"
    if ledger.exists():ledger.unlink()
    results=[]
    for _,r in scenarios.iterrows():
        outcome,trace=run_agent(str(r.message),max_steps=6)
        label=outcome.split(":")[0].strip().upper()
        if label not in {"REFUND","ESCALATE","DENIED","NOT_FOUND","FALLBACK"}:label="FALLBACK"
        results.append({"scenario_id":r.scenario_id,"expected_outcome":r.expected_outcome,"actual_outcome":label,"match":label==r.expected_outcome,"steps":len(trace),"trace":json.dumps(trace,default=str)})
    pd.DataFrame(results).to_csv(ROOT/"outputs/agent_results.csv",index=False)
    with (ROOT/"outputs/agent_runs.jsonl").open("w",encoding="utf-8") as f:
        for row in results:f.write(json.dumps(row)+"\n")
    matched=sum(x["match"] for x in results)
    (ROOT/"outputs/agent_metrics.txt").write_text(f"Scenarios: {len(results)}\nExpected outcomes matched: {matched}\nAccuracy: {matched/len(results):.3f}\nMax steps per scenario: 6\n",encoding="utf-8")
    print(f"Refund agent: {matched}/{len(results)} expected outcomes matched. Refunds are simulated, not real.")
if __name__=="__main__":
    policy_qa(); run()
