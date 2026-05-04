from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from services.predictor import predict_price, classify_segment, cluster_property
from rag.retriever import get_comparable_properties

router = APIRouter(tags=["Predict"])

class PredictPriceRequest(BaseModel):
    kamar_tidur: int = Field(..., ge=1, le=10)
    kamar_mandi: int = Field(..., ge=1, le=10)
    garasi: int = Field(..., ge=0, le=5)
    luas_tanah: float = Field(..., gt=0)
    luas_bangunan: float = Field(..., gt=0)
    lokasi: str = Field(..., min_length=2)

class SegmentRequest(PredictPriceRequest):
    harga: Optional[float] = Field(None, description="Property price (optional). If not provided, it will be estimated using the model.")

class ClusterRequest(SegmentRequest):
    pass

@router.post("/predict_price")
async def api_predict_price(body: PredictPriceRequest):
    """
    Estimate house price using the dual-model CatBoost.
    """
    try:
        return predict_price(**body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify_segment")
async def api_classify_segment(body: SegmentRequest):
    """
    Classify property price segments into 4 classes.
    """
    try:
        return classify_segment(**body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster_property")
async def api_cluster_property(body: ClusterRequest):
    """
    Determine property market clusters using KMeans + UMAP (6 clusters).
    """
    try:
        return cluster_property(**body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ComparableRequest(BaseModel):
    kamar_tidur: int = Field(..., ge=1, le=10)
    kamar_mandi: int = Field(..., ge=1, le=10)
    garasi: int = Field(..., ge=0, le=5)
    luas_tanah: float = Field(..., gt=0)
    luas_bangunan: float = Field(..., gt=0)
    lokasi: str = Field(..., min_length=2)
    harga: float = Field(..., gt=0)
    top_k: int = Field(5, ge=1, le=20)

@router.post("/comparable_properties")
async def api_comparable_properties(body: ComparableRequest):
    """Find top-K most similar properties using vector similarity search."""
    try:
        results = get_comparable_properties(**body.model_dump())
        return {"comparables": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))