from __future__ import annotations

import re
from typing import Any

KPI_KEYWORDS = {
    "late_delivery_rate": ["late", "delay", "delivered", "delivery"],
    "average_review_score": ["review", "score", "rating", "satisfaction"],
    "seller_revenue": ["seller", "revenue", "sales", "price", "freight"],
    "freight_cost_percentage": ["freight", "shipping", "cost percentage"],
    "order_cancellation_rate": ["cancel", "cancellation", "canceled"],
    "monthly_revenue_trend": ["month", "monthly", "trend", "revenue drop"],
    "customer_repeat_purchase_rate": ["repeat", "returning", "customer"],
}

RULES = {
    "logistics_risk_rule": {
        "label": "Logistics Review",
        "priority": "High",
        "reason": "Late delivery rate is above the 15% operational risk threshold.",
        "action": "Review carrier performance, shipping promises, and seller fulfillment SLAs for the flagged segment.",
    },
    "customer_experience_rule": {
        "label": "Customer Experience Review",
        "priority": "High",
        "reason": "Average review score is below the 3.5 customer satisfaction threshold.",
        "action": "Inspect review themes, delivery issues, product quality, and seller communication for this segment.",
    },
    "seller_quality_rule": {
        "label": "Seller Quality Check",
        "priority": "High",
        "reason": "Cancellation rate is above the 10% seller quality threshold.",
        "action": "Audit inventory reliability, seller responsiveness, and cancellation reasons before scaling this seller/category.",
    },
    "shipping_cost_rule": {
        "label": "Shipping Cost Optimization",
        "priority": "Medium",
        "reason": "Freight cost percentage is above the 30% margin pressure threshold.",
        "action": "Review shipping subsidies, packaging weight, carrier mix, and regional fulfillment options.",
    },
    "revenue_decline_rule": {
        "label": "Pricing/Promotion Investigation",
        "priority": "High",
        "reason": "Revenue decline is above the 20% month-over-month threshold.",
        "action": "Check competitor pricing, stock availability, marketing exposure, and promotion timing.",
    },
    "concentration_risk_rule": {
        "label": "Revenue Concentration Review",
        "priority": "Medium",
        "reason": "A single seller/category appears to dominate the returned revenue view.",
        "action": "Monitor dependency risk and consider diversifying demand across alternative sellers or categories.",
    },
}


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def row_dicts(columns: list[str], rows: list[tuple]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row)) for row in rows]


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip().rstrip("%")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
            return float(cleaned)
    return None


def detect_kpis(question: str, columns: list[str], sql: str) -> list[str]:
    text = " ".join([question or "", sql or "", " ".join(columns)]).lower()
    detected = []
    for kpi, keywords in KPI_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected.append(kpi)
    return detected or ["general_business_metric"]


def classify_difficulty(sql: str, detected_kpis: list[str]) -> str:
    lowered = (sql or "").lower()
    joins = lowered.count(" join ")
    has_grouping = " group by " in lowered
    has_trend = any(token in lowered for token in ["strftime", "month", "%y", "%m"])
    has_multiple_kpis = len([k for k in detected_kpis if k != "general_business_metric"]) >= 2

    if joins >= 2 or has_trend or has_multiple_kpis:
        return "Complex"
    if joins == 1 or has_grouping:
        return "Medium"
    return "Simple"


def explain_sql(sql: str) -> list[str]:
    lowered = (sql or "").lower()
    explanation = []
    tables = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered)
    if tables:
        explanation.append(f"Uses {', '.join(dict.fromkeys(tables))} to answer the business question.")
    if " join " in lowered:
        explanation.append("Joins related Olist tables so the metric is calculated at the right business grain.")
    if any(token in lowered for token in ["sum(", "avg(", "count("]):
        explanation.append("Calculates an aggregate KPI such as revenue, count, average score, or rate.")
    if " group by " in lowered:
        explanation.append("Groups results by the requested business segment before comparing performance.")
    if " order by " in lowered:
        explanation.append("Sorts the result so the highest-priority records appear first.")
    if " limit " in lowered:
        explanation.append("Limits the output to the most relevant records for dashboard readability.")
    return explanation or ["Runs a validated read-only query against the configured analytics database."]


def add_recommendation(recommendations: list[dict[str, Any]], rule_id: str, metric: str, value: float, entity: str | None = None) -> None:
    rule = RULES[rule_id]
    recommendations.append({
        "rule_id": rule_id,
        "recommendation": rule["label"],
        "priority": rule["priority"],
        "metric": metric,
        "value": round(value, 2),
        "entity": entity,
        "reason": rule["reason"],
        "action": rule["action"],
    })


def evaluate_business_rules(question: str, columns: list[str], rows: list[tuple], sql: str) -> dict[str, Any]:
    detected = detect_kpis(question, columns, sql)
    records = row_dicts(columns, rows)
    recommendations: list[dict[str, Any]] = []

    for record in records[:10]:
        entity = None
        for key in ["seller_id", "category", "product_category_name_english", "customer_state", "payment_type"]:
            if key in record:
                entity = str(record[key])
                break

        for column, raw_value in record.items():
            col = normalize_name(column)
            value = numeric_value(raw_value)
            if value is None:
                continue

            if any(token in col for token in ["late_delivery", "late_rate", "late_percentage", "late_delivery_percentage"]) and value > 15:
                add_recommendation(recommendations, "logistics_risk_rule", column, value, entity)
            elif any(token in col for token in ["review_score", "avg_review", "average_review"]) and value < 3.5:
                add_recommendation(recommendations, "customer_experience_rule", column, value, entity)
            elif "cancellation" in col and ("rate" in col or "percentage" in col) and value > 10:
                add_recommendation(recommendations, "seller_quality_rule", column, value, entity)
            elif any(token in col for token in ["freight_percentage", "freight_cost_percentage", "avg_freight_percentage"]) and value > 30:
                add_recommendation(recommendations, "shipping_cost_rule", column, value, entity)
            elif any(token in col for token in ["revenue_drop", "decline", "drop_percentage"]) and value > 20:
                add_recommendation(recommendations, "revenue_decline_rule", column, value, entity)

    if not recommendations and records:
        numeric_columns = []
        for column in columns:
            values = [numeric_value(record.get(column)) for record in records]
            values = [value for value in values if value is not None]
            if values:
                numeric_columns.append((column, values))
        if numeric_columns and len(records) == 1:
            col, values = numeric_columns[0]
            if any(kpi in detected for kpi in ["seller_revenue", "monthly_revenue_trend"]):
                add_recommendation(recommendations, "concentration_risk_rule", col, values[0], None)

    if not recommendations:
        recommendations.append({
            "rule_id": "monitoring_rule",
            "recommendation": "Monitor KPI",
            "priority": "Low",
            "metric": detected[0],
            "value": None,
            "entity": None,
            "reason": "No configured risk threshold was crossed in the returned result.",
            "action": "Use this result as a baseline and keep monitoring the KPI over time.",
        })

    top = recommendations[0]
    return {
        "detected_kpis": detected,
        "difficulty_label": classify_difficulty(sql, detected),
        "sql_explanation": explain_sql(sql),
        "recommendations": recommendations,
        "recommendation": top["recommendation"],
        "priority": top["priority"],
        "recommendation_rule": top["rule_id"],
        "recommendation_reason": top["reason"],
        "recommended_action": top["action"],
        "safety_status": {
            "sql_validation": "Passed",
            "tables_used": list(dict.fromkeys(re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", (sql or "").lower()))),
            "rows_returned": len(rows),
            "recommendation_rule": top["rule_id"],
            "insight_grounding": "Checked by faithfulness harness for saved eval runs",
        },
    }

