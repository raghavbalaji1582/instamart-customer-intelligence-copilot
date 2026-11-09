# Instamart Customer Intelligence & Support Copilot

Offline capstone project using deterministic synthetic data. It combines data cleaning and SQLite, RFM segmentation, PCA, lapse-risk modelling, support-ticket PII redaction/classification, TF-IDF policy retrieval, and a simulated refund agent. No paid API or API key is required.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Structure
- `generate_data.py`, `mock_llm.py`: deterministic synthetic data and offline mock model
- `src/part_a_data.py`: cleaning, validation, SQLite database
- `src/part_b_ml.py`: RFM, K-means, PCA, lapse model, retention list
- `src/part_c_triage.py`: PII masking, classification, retries, token budget
- `src/part_d_agent.py`: TF-IDF RAG, policy QA, tool-using refund agent
- `sql/queries.sql`: seven analysis queries
- `data/`: generated raw/clean data and policy text
- `outputs/`: evaluation metrics, model, predictions and agent traces

## Part A — Data foundation (22 marks)
Generates fictional customers, products, orders, tickets, policy questions and refund scenarios. Cleans inconsistent city/gender values, invalid ages, duplicate/pending order rows, invalid amounts and impossible delivery durations. Writes cleaned CSVs and a SQLite database with four tables and indexes. Seven SQL queries cover revenue, customers, categories, delivery SLA, ticket mix and repeat/no-order customers.

## Part B — Customer intelligence (28 marks)
Calculates recency, completed-order frequency and monetary spend; standardises the features; applies four-cluster K-means; exports PCA coordinates; trains a Random Forest lapse-risk model; and ranks the 100 highest-risk retention candidates. The lapse label is a synthetic demonstration and should not be treated as a production churn definition.

## Part C — Support copilot (22 marks)
Masks email addresses and Indian mobile numbers, classifies tickets through the supplied local mock LLM, validates JSON and categories, retries transient/malformed responses, routes unresolved or empty tickets to human review, assigns high priority to fraud and demonstrates a bounded token budget.

## Part D — RAG and refund agent (28 marks)
Retrieves policy sentences with TF-IDF and evaluates policy questions, including unsupported questions. A ReAct-style tool loop uses order lookup, policy lookup, refund calculation, simulated issuance and human escalation. Eligibility requires a completed order and delivery over 45 minutes; refunds above INR 1,000 require human approval; the agent has a six-step limit. Refunds are simulation-only.

## Outputs
- `outputs/customer_segments.csv`, `pca_segments.csv`, `retention_list.csv`
- `outputs/lapse_model.joblib`, `lapse_model_metrics.txt`
- `outputs/triage_results.csv`, `triage_budget_demo.csv`, `triage_metrics.txt`
- `outputs/policy_qa_results.csv`, `policy_qa_metrics.txt`
- `outputs/agent_results.csv`, `agent_runs.jsonl`, `agent_metrics.txt`, `refund_ledger.jsonl`

All records are synthetic. `mock_llm.py` is a deterministic mock, not a hosted LLM. TF-IDF is a transparent baseline, not semantic embedding retrieval. Do not add real customer PII, credentials or payment data to this public repository.
