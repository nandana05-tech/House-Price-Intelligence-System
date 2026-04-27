"""Kafka topic name constants — single source of truth for all producers/consumers."""

# ── Prediction events (MCP → audit trail) ────────────────────────────────
REGRESSION_EVENTS = "property.prediction.regression"
CLASSIFICATION_EVENTS = "property.prediction.classification"
CLUSTERING_EVENTS = "property.prediction.clustering"

# ── Feedback events (API → feedback store) ───────────────────────────────
FEEDBACK_EVENTS = "property.feedback"

# ── All topics (used by admin/setup scripts) ─────────────────────────────
ALL_TOPICS = [
    REGRESSION_EVENTS,
    CLASSIFICATION_EVENTS,
    CLUSTERING_EVENTS,
    FEEDBACK_EVENTS,
]
