"""
Builds enriched prompt context combining:
- ML prediction output
- Static knowledge (area profiles, rules) via pgvector
- Dynamic price statistics (live SQL from property_embeddings)
- Comparable properties (vector similarity search)
"""
from __future__ import annotations
from rag.retriever import get_comparable_properties, get_knowledge, get_area_stats
from rag.cache import get_cached_context, set_cached_context

RAG_KEYWORDS = [
    "harga", "price", "estimasi", "lokasi", "segmen", "cluster",
    "rumah", "properti", "property", "beli", "invest", "mahal", "murah",
]


def should_use_rag(query: str) -> bool:
    """Only trigger RAG for property-related queries — saves tokens."""
    return any(kw in query.lower() for kw in RAG_KEYWORDS)


def build_prediction_context(
    prediction: dict,
    comparables: list[dict],
    knowledge: list[dict],
    area_stats: dict,
) -> str:
    lines = []

    # ── ML Prediction Result ──────────────────────────────────────────
    lines.append("## ML Prediction Result")
    lines.append(f"- Estimated Price: {prediction.get('harga_estimasi_format', 'N/A')}")
    lines.append(f"- Model Used: {prediction.get('model_digunakan', 'N/A')}")
    lines.append(f"- MAPE: {prediction.get('mape_persen', 'N/A')}%")
    lines.append(f"- Segment: {prediction.get('kelas_label', 'N/A')}")
    lines.append(f"- Cluster: {prediction.get('cluster_label', 'N/A')}")

    # ── Dynamic: Live Area Statistics ─────────────────────────────────
    lokasi = prediction.get("lokasi", "")
    if area_stats:
        lines.append(f"\n## Live Market Statistics — {lokasi}")
        lines.append(f"- Average price: Rp {area_stats['avg_harga']:,}")
        lines.append(
            f"- Price range: Rp {area_stats['min_harga']:,} – Rp {area_stats['max_harga']:,}"
        )
        lines.append(f"- Dominant segment: {area_stats['dominant_segment']}")
        lines.append(f"- Data points: {area_stats['jumlah_data']} properties")

    # ── Vector Search: Comparable Properties ──────────────────────────
    if comparables:
        lines.append("\n## Comparable Properties")
        for i, p in enumerate(comparables, 1):
            lines.append(
                f"{i}. {p.get('lokasi')} | {p.get('kamar_tidur')}BR "
                f"| LT {p.get('luas_tanah')}m² | Rp {p.get('harga'):,} "
                f"| {p.get('segment_label')} | sim: {p.get('similarity')}"
            )

    # ── Static: Knowledge Documents ───────────────────────────────────
    if knowledge:
        lines.append("\n## Area & Market Knowledge")
        for doc in knowledge:
            lines.append(f"### {doc['title']}")
            lines.append(doc["content"])

    return "\n".join(lines)


def build_rag_context(query: str, prediction: dict | None = None) -> str:
    """
    Main entry point.
    Returns cached context if available, otherwise retrieves fresh context.
    Returns empty string if query is not property-related.
    """
    if not should_use_rag(query):
        return ""

    # Check Redis cache first
    cached = get_cached_context(query, prediction)
    if cached:
        return cached

    # Static: knowledge documents via pgvector
    knowledge = get_knowledge(query, top_k=2)

    comparables = []
    area_stats = {}

    if prediction:
        lokasi = prediction.get("lokasi", "")

        # Dynamic: live price stats from property_embeddings
        area_stats = get_area_stats(lokasi)

        # Vector search: comparable properties
        comparables = get_comparable_properties(
            lokasi=lokasi,
            kamar_tidur=prediction.get("kamar_tidur", 0),
            kamar_mandi=prediction.get("kamar_mandi", 0),
            garasi=prediction.get("garasi", 0),
            luas_tanah=prediction.get("luas_tanah", 0),
            luas_bangunan=prediction.get("luas_bangunan", 0),
            harga=prediction.get("harga_estimasi", 0),
            top_k=3,
        )

    context = build_prediction_context(prediction or {}, comparables, knowledge, area_stats)

    # Cache result
    set_cached_context(query, prediction, context)
    return context