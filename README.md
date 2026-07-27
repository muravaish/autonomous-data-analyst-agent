# Autonomous Data Analyst + Decision Intelligence Agent

An agentic e-commerce analytics app that converts plain-English business questions into validated SQL, runs the query on the Olist Brazilian E-Commerce database in local SQLite mode or Azure PostgreSQL mode, generates a grounded business insight, creates an interactive chart, and recommends a business action using deterministic decision rules.

This project is intentionally different from a prediction dashboard. It is not centered on training a model. Instead, it demonstrates agentic analytics, SQL reasoning, evaluation, and decision support.

## Project Positioning

**Churn project:** prediction + dashboard  
**This project:** AI data analyst + SQL reasoning + decision intelligence

The core idea is to answer three business questions in one workflow:

1. What happened in the data?
2. Can the answer be verified against the database?
3. What should the business do next?

## Features

- Plain-English question answering over a real e-commerce database
- LangGraph pipeline with explicit state between nodes
- Gemini-powered SQL and insight generation
- Schema-aware SQL validation before execution
- Database adapter with local SQLite and Azure PostgreSQL support
- Dockerized Streamlit app with Azure Container Apps deployment template
- Plotly chart generation saved as HTML
- Deterministic business rules layer for recommendations
- Session memory for follow-up questions in Streamlit
- SQL explanation for non-technical users
- Confidence and safety status panel
- Downloadable business analysis report
- Execution accuracy evaluation harness
- Faithfulness harness to catch numeric hallucinations
- Evaluation dashboard with accuracy, faithfulness, failure types, and recommendation-rule breakdown

## Architecture

```text
User Question
   ↓
Streamlit Chat UI
   ↓
Router Agent
   ↓
SQL Agent
   ↓
SQL Validation
   ↓
Execution Agent
   ↓
Insight Agent
   ↓
Business Rules / Recommendation Layer
   ↓
Chart Agent
   ↓
Final Answer + SQL + Table + Insight + Recommendation + Chart
```

## Agent Nodes

**Router Agent**  
Classifies whether the business question is a single-query or multi-step analysis request.

**SQL Agent**  
Generates one dialect-aware SQL query using the live database schema and Olist join rules.

**Execution Agent**  
Runs validated SQL against the configured analytics database and returns columns, rows, and row count.

**Insight Agent**  
Writes a short narrative using only the actual returned query results.

**Business Rules Layer**  
Maps returned KPIs to deterministic recommendations and priorities.

**Chart Agent**  
Chooses a simple chart type from the result shape and saves `charts/latest_chart.html`.

## Business KPI Layer

The system detects and reasons over e-commerce KPIs such as:

- Late delivery rate
- Average review score
- Seller revenue
- Freight cost percentage
- Order cancellation rate
- Monthly revenue trend
- Customer repeat purchase rate

## Recommendation Rules

Examples of deterministic rules in `business_rules.py`:

| KPI Pattern | Rule | Recommendation |
|---|---|---|
| `late_delivery_rate > 15%` | `logistics_risk_rule` | Logistics review |
| `avg_review_score < 3.5` | `customer_experience_rule` | Customer experience review |
| `cancellation_rate > 10%` | `seller_quality_rule` | Seller quality check |
| `freight_percentage > 30%` | `shipping_cost_rule` | Shipping cost optimization |
| `revenue_drop > 20%` | `revenue_decline_rule` | Pricing or promotion investigation |

This layer is deterministic Python logic, not another LLM judgment step. That makes the recommendation path explainable and interview-friendly.

## Evaluation Results

Current saved evaluation results:

| Evaluation Type | Result |
|---|---:|
| Execution accuracy | 8/10 |
| Insight faithfulness | 10/10 |
| Numeric grounding | 11/11 |

Recommendation-rule breakdown from the saved evaluation set:

| Rule | Count |
|---|---:|
| `monitoring_rule` | 5 |
| `concentration_risk_rule` | 2 |
| `customer_experience_rule` | 1 |
| `logistics_risk_rule` | 1 |
| `shipping_cost_rule` | 1 |

## Why The Evaluation Layer Matters

Most LLM-over-database demos only check whether the answer sounds fluent. This project checks correctness directly.

**Execution accuracy** asks:

```text
Did the generated SQL return the same answer as the gold-standard SQL?
```

**Faithfulness checking** asks:

```text
Did the written insight mention only numbers that actually appear in the returned data?
```

The faithfulness harness is deterministic. It extracts numbers from the insight text and checks them against the SQL result within a rounding tolerance. This avoids using a second LLM as a judge.

## Example Questions

Try these in the Streamlit app:

```text
What is the average review score for each payment type?
Which sellers have the highest late delivery risk?
Which sellers generated the highest total revenue?
What is the average freight cost as a percentage of item price?
How many orders were delivered later than their estimated delivery date?
```

## How To Run

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your environment file:

```powershell
Copy-Item .env.example .env
```

Add your Gemini key to `.env`:

```text
GOOGLE_API_KEY=your_key_here
```

## Azure/PostgreSQL Mode

The app is Azure-ready through `db_adapter.py`. Local SQLite remains the default for development, but the same LangGraph pipeline can run against Azure Database for PostgreSQL after the Olist tables are loaded there.

Local mode:

```text
DATABASE_BACKEND=sqlite
SQLITE_DB_PATH=olist.db
```

Azure PostgreSQL mode:

```text
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://username:password@your-server.postgres.database.azure.com:5432/your_database?sslmode=require
DATABASE_SCHEMA=public
```

The SQL Agent automatically reads the live schema, switches the prompt dialect to PostgreSQL, validates table references, blocks non-read-only SQL, and executes through the same adapter interface.

By default, make sure `olist.db` is in the project root. For Azure mode, configure the PostgreSQL settings below.

Run the Streamlit app:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Evaluation Commands

Run the execution accuracy harness:

```powershell
python eval_harness.py
```

Run the faithfulness harness:

```powershell
python faithfulness_harness.py
```

Enrich existing evaluation results with recommendation-rule fields without making new API calls:

```powershell
python decision_enrichment.py
```

## Important Files

| File | Purpose |
|---|---|
| `agent_graph.py` | LangGraph pipeline and agent nodes |
| `db_adapter.py` | Database adapter for local SQLite and Azure PostgreSQL |
| `business_rules.py` | KPI detection and deterministic recommendation rules |
| `app.py` | Streamlit chat app and evaluation dashboard |
| `eval_harness.py` | Execution accuracy evaluation |
| `faithfulness_harness.py` | Numeric faithfulness evaluation |
| `decision_enrichment.py` | Adds business-rule fields to saved eval results |
| `gold_qa.json` | Gold-standard evaluation questions |
| `Dockerfile` | Container image definition for deployment |
| `.github/workflows/deploy-azure.yml` | Manual Azure Container Apps deployment workflow |
| `docs/azure-deployment.md` | Azure deployment guide |
| `olist.db` | Local SQLite database for development mode |

## Demo Recording Script

A short 2-3 minute demo can follow this flow:

1. Open the Streamlit app.
2. Ask: `Which sellers have the highest late delivery risk?`
3. Show the generated SQL and result table.
4. Explain the business recommendation and priority.
5. Show the chart.
6. Switch to the Evaluation Dashboard.
7. Point out execution accuracy, faithfulness, and one real failure category.
8. Download the business report.

## CI/CD And Productionization

The repository includes a GitHub Actions CI workflow in `.github/workflows/ci.yml`. On every push or pull request, it installs dependencies, compiles the Python files, runs the deterministic decision-enrichment step without making any Gemini API calls, and validates that the Docker image builds.

Production-ready pieces now included:

- Dockerfile for containerizing the Streamlit app
- Docker Compose file for local container testing
- Streamlit server configuration for container hosting
- Azure PostgreSQL support through `db_adapter.py`
- Manual Azure Container Apps deployment workflow in `.github/workflows/deploy-azure.yml`
- Azure deployment guide in `docs/azure-deployment.md`

Remaining production next steps:

- Create the actual Azure resources in your Azure account
- Load Olist tables into Azure Database for PostgreSQL
- Add GitHub Actions secrets for Azure deployment
- Add n8n or scheduled jobs for automated weekly KPI reports

## Future Improvements

- Add dataset adapter files so the same architecture can support other datasets
- Add recommendation-specific gold labels and rule accuracy evaluation
- Add richer conversational memory for multi-turn analysis
- Add optional seller risk scoring as a small ML extension

## Portfolio Summary

This project demonstrates an agentic business analytics system that translates natural language into validated SQL, runs analysis on real e-commerce data, generates faithful insights, visualizes results, and converts findings into deterministic business recommendations.





