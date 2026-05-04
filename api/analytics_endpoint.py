from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(tags=["Analytics"])

class AreaAnalytics(BaseModel):
    nama: str
    avg_per_m2: float
    total_data: int
    trend: str
    segmen_dom: str
    catatan: str

SEGMENT_MAP = {
    0: "Murah",
    1: "Menengah",
    2: "Atas",
    3: "Mewah"
}

@router.get("/analytics/areas", response_model=List[AreaAnalytics])
async def get_area_analytics(request: Request):
    """
    Get top 10 areas analytics dynamically from the database.
    Trend uses average selisihPersen from Feedback.
    """
    db = request.app.state.db
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")

    # Raw SQL to get analytics grouped by location
    query = """
    WITH FeedbackStats AS (
        SELECT lokasi, AVG("selisihPersen") as avg_selisih
        FROM "Feedback"
        GROUP BY lokasi
    )
    SELECT 
        c.lokasi as nama,
        COUNT(c.id) as total_data,
        AVG(c.harga / c."luasTanah") as avg_per_m2,
        MODE() WITHIN GROUP (ORDER BY c."kelasHarga") as segmen_dom_id,
        COALESCE(f.avg_selisih, 0) as trend_selisih
    FROM "CleanPropertyClassification" c
    LEFT JOIN FeedbackStats f ON c.lokasi = f.lokasi
    GROUP BY c.lokasi, f.avg_selisih
    ORDER BY total_data DESC
    LIMIT 10;
    """

    try:
        results = await db.query_raw(query)
        
        analytics_list = []
        for row in results:
            segmen_id = int(row.get("segmen_dom_id", 1))
            segmen_dom = SEGMENT_MAP.get(segmen_id, "Menengah")
            
            trend_val = row.get("trend_selisih", 0)
            trend_str = f"±{trend_val:.1f}%" if trend_val > 0 else "N/A"
            
            analytics_list.append(
                AreaAnalytics(
                    nama=row.get("nama"),
                    avg_per_m2=float(row.get("avg_per_m2", 0)),
                    total_data=int(row.get("total_data", 0)),
                    trend=trend_str,
                    segmen_dom=segmen_dom,
                    catatan="Data dianalisis secara dinamis berdasarkan model klasifikasi."
                )
            )
            
        return analytics_list

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
