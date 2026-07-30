# Autonomous Data Analyst + Decision Intelligence Agent

An agentic analytics app that turns plain-English business questions into validated SQL, executes the query on a relational database or uploaded CSV dataset, generates a grounded insight, chooses an appropriate chart, and recommends a business action using deterministic rules.

This is not a prediction dashboard. The focus is agentic data analysis, SQL reasoning, answer verification, and decision intelligence.

## What It Does

- Converts natural-language questions into SQL
- Validates table and column names against the live schema before execution
- Runs SQL against local SQLite, uploaded CSV data, or Azure-ready PostgreSQL configuration
- Generates an insight using only the returned data
- Chooses charts automatically: pie, bar, grouped bar, line, scatter, histogram, or KPI chart
- Applies business rules to recommend actions and priority
- Shows SQL explanation, safety status, result table, chart, and downloadable report
- Evaluates correctness using gold-standard SQL and numeric faithfulness checks

## Screenshots

### Uploaded CSV Analysis

![Uploaded CSV home](assets/screenshots/01-uploaded-csv-home.png)

### Insight And Recommendation

![Weather answer and recommendation](assets/screenshots/02-weather-answer-recommendation.png)

### Smart Chart Selection

![Weather condition pie chart](assets/screenshots/03-weather-condition-pie-chart.png)

### Result Table

![Weather data table](assets/screenshots/04-weather-data-table.png)

### SQL Transparency

![Generated SQL query](assets/screenshots/05-weather-sql-query.png)

### Evaluation Dashboard

![Evaluation dashboard](assets/screenshots/06-evaluation-dashboard.png)

## Architecture

```text
User Question
   ↓
Streamlit UI
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
Recommendation Rules
   ↓
Chart Agent
   ↓
Answer + SQL + Table + Chart + Recommendation + Report
```

## Agent Nodes

| Node | Responsibility |
|---|---|
| Router | Decides whether the question is simple or multi-step |
| SQL Agent | Generates dialect-aware SQL from the live schema |
| SQL Validation | Blocks hallucinated tables/columns and non-read-only SQL |
| Execution Agent | Runs the query and returns rows, columns, and stats |
| Insight Agent | Writes a short narrative from the actual returned data |
| Recommendation Layer | Applies deterministic KPI rules and action mapping |
| Chart Agent | Chooses a suitable chart from the question intent and result shape |

## Smart Chart Agent

The chart agent does not blindly use one graph type. It combines question intent with the returned data shape.

| Question/Data Pattern | Chart Type |
|---|---|
| Percentage, share, composition, breakdown | Pie chart with top categories + Other |
| Date/month/year + metric | Line chart |
| Relationship, correlation, versus | Scatter plot |
| Category + multiple metrics | Grouped bar chart |
| Ranking, top, highest, comparison | Horizontal bar chart |
| Single numeric distribution | Histogram |
| One-row KPI result | KPI bar chart |

## Business KPI And Recommendation Layer

The system detects business metrics and maps them to deterministic recommendations.

| KPI Pattern | Example Recommendation |
|---|---|
| Late delivery rate above threshold | Logistics review |
| Average review score below threshold | Customer experience review |
| Cancellation rate above threshold | Seller quality check |
| Freight percentage above threshold | Shipping cost optimization |
| Revenue drop above threshold | Pricing or promotion investigation |

This layer is plain Python business logic, not another LLM judge, so the recommendation path is explainable.

## Evaluation Results

Current saved evaluation results:

| Evaluation Type | Result |
|---|---:|
| Execution accuracy | 17/20 (85.0%) |
| Insight faithfulness | 20/20 (100.0%) |
| Numeric grounding | 29/29 (100.0%) |

Execution accuracy by difficulty:

| Difficulty | Result |
|---|---:|
| Easy | 8/8 (100.0%) |
| Medium | 6/6 (100.0%) |
| Hard | 3/6 (50.0%) |

The evaluation layer checks two things:

- Execution accuracy: whether generated SQL returns the same result as gold-standard SQL
- Faithfulness: whether numbers mentioned in the written insight are grounded in the returned data

## Example Questions

For the Olist e-commerce database:

```text
Which sellers have the highest late delivery risk?
What is the average review score for each payment type?
Which sellers generated the highest total revenue?
What is the average freight cost as a percentage of item price?
```

For an uploaded weather CSV:

```text
Show the percentage distribution of weather conditions across all locations.
Compare PM2.5 levels across the top 15 most polluted cities.
Show the relationship between temperature and humidity.
Which locations should be flagged for weather or air quality risk?
```

## Tech Stack

- Python
- LangGraph
- Google Gemini
- SQLite
- Azure-ready PostgreSQL adapter
- Streamlit
- Plotly
- Pandas
- GitHub Actions

## How To Run Locally

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

Add your Gemini key:

```text
GOOGLE_API_KEY=your_key_here
```

Run the app:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Evaluation Commands

Run execution accuracy evaluation:

```powershell
python eval_harness.py
```

Run faithfulness evaluation:

```powershell
python faithfulness_harness.py
```

Enrich saved evaluation results with recommendation-rule fields:

```powershell
python decision_enrichment.py
```

## Azure-Ready Database Mode

Local SQLite is the default for development. The same pipeline can also run against Azure Database for PostgreSQL through `db_adapter.py`.

SQLite mode:

```text
DATABASE_BACKEND=sqlite
SQLITE_DB_PATH=olist.db
```

PostgreSQL mode:

```text
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://username:password@your-server.postgres.database.azure.com:5432/your_database?sslmode=require
DATABASE_SCHEMA=public
```

## Important Files

| File | Purpose |
|---|---|
| `agent_graph.py` | LangGraph pipeline and agent nodes |
| `app.py` | Streamlit chat UI and evaluation dashboard |
| `db_adapter.py` | SQLite/PostgreSQL database adapter |
| `business_rules.py` | KPI detection and deterministic recommendation rules |
| `eval_harness.py` | Execution accuracy evaluation |
| `faithfulness_harness.py` | Numeric faithfulness evaluation |
| `gold_qa.json` | Gold-standard question set |
| `assets/screenshots/` | README and portfolio screenshots |

## Current Limitations

- Uploaded CSV analysis depends on the model correctly interpreting generic column names
- Gold-standard evaluation is strongest for the Olist dataset; uploaded CSV evaluation is a future extension
- Complex multi-step business questions can still fail when joins or metric definitions are ambiguous
- Azure deployment templates are included, but local Streamlit remains the primary tested run mode

## Future Improvements

- Add a small gold evaluation set for uploaded CSV datasets
- Add richer memory for multi-turn follow-up analysis
- Add optional scheduled KPI reports through n8n or cloud workflows
- Add more domain-specific business rule packs for non-e-commerce datasets

## Portfolio Summary

This project demonstrates an agentic business analytics system that translates natural language into validated SQL, runs analysis on real data, generates faithful insights, visualizes results, and converts findings into deterministic business recommendations.
