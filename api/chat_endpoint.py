import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from rag.context_builder import build_rag_context

from services.predictor import classify_segment, cluster_property, predict_price

router = APIRouter(tags=["Chat Agent"])

# Async client will automatically pick up OPENAI_API_KEY from environment
client = AsyncOpenAI()

class ChatRequest(BaseModel):
    message: str = Field(..., description="Pesan atau pertanyaan dari user")

class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str]

# ── Tool Definitions for OpenAI ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_price",
            "description": "Estimasi harga rumah di Depok. Wajib digunakan jika user menanyakan harga properti atau estimasi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kamar_tidur": {"type": "integer", "description": "Jumlah kamar tidur"},
                    "kamar_mandi": {"type": "integer", "description": "Jumlah kamar mandi"},
                    "garasi": {"type": "integer", "description": "Kapasitas garasi (jumlah mobil)"},
                    "luas_tanah": {"type": "number", "description": "Luas tanah dalam meter persegi (m2)"},
                    "luas_bangunan": {"type": "number", "description": "Luas bangunan dalam meter persegi (m2)"},
                    "lokasi": {"type": "string", "description": "Nama kecamatan atau lokasi di Depok"}
                },
                "required": ["kamar_tidur", "kamar_mandi", "garasi", "luas_tanah", "luas_bangunan", "lokasi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_segment",
            "description": "Klasifikasi segmen kelas properti (Murah, Menengah, Atas, Mewah).",
            "parameters": {
                "type": "object",
                "properties": {
                    "kamar_tidur": {"type": "integer"},
                    "kamar_mandi": {"type": "integer"},
                    "garasi": {"type": "integer"},
                    "luas_tanah": {"type": "number"},
                    "luas_bangunan": {"type": "number"},
                    "lokasi": {"type": "string"},
                    "harga": {"type": "number", "description": "Harga properti dalam Rupiah (Opsional)"}
                },
                "required": ["kamar_tidur", "kamar_mandi", "garasi", "luas_tanah", "luas_bangunan", "lokasi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_property",
            "description": "Tentukan klaster pasar properti untuk analisis segmen pasar dan properti serupa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kamar_tidur": {"type": "integer"},
                    "kamar_mandi": {"type": "integer"},
                    "luas_tanah": {"type": "number"},
                    "luas_bangunan": {"type": "number"},
                    "lokasi": {"type": "string"},
                    "harga": {"type": "number", "description": "Harga properti dalam Rupiah (Opsional)"}
                },
                "required": ["kamar_tidur", "kamar_mandi", "luas_tanah", "luas_bangunan", "lokasi"]
            }
        }
    }
]

# ── Route ──────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(body: ChatRequest) -> ChatResponse:
    """
    Kirim pesan ke LLM (GPT-4o-mini) yang sudah dilengkapi dengan fungsi (tools) 
    untuk memprediksi harga, mengklasifikasikan segmen, dan melakukan clustering properti.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY tidak ditemukan di environment.")

    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah AI asisten real estate yang sangat pintar khusus untuk area Depok. "
                "Anda memiliki akses ke berbagai model machine learning melalui function calling. "
                "Bantu user untuk menaksir harga rumah, mengklasifikasikan segmen, atau menentukan klaster properti. "
                "Selalu berikan jawaban dalam bahasa Indonesia yang natural, ramah, dan informatif. "
                "Jika Anda menggunakan tool prediksi, sertakan informasi harga atau segmen di jawaban Anda dengan jelas."
            )
        },
        {"role": "user", "content": body.message}
    ]

    try:
        # Panggilan pertama ke GPT
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        tools_used = []

        # Jika GPT memutuskan untuk memanggil fungsi (tool)
        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                tools_used.append(function_name)
                
                # Eksekusi fungsi lokal
                function_response = {}
                if function_name == "predict_price":
                    function_response = predict_price(**function_args)
                elif function_name == "classify_segment":
                    function_response = classify_segment(**function_args)
                elif function_name == "cluster_property":
                    function_response = cluster_property(**function_args)
                    
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
            
            # ── RAG INJECTION ─────────────────────────────────────────────
            # Extract prediction result from tool responses (if any)
            prediction_result = None
            if tools_used and "predict_price" in tools_used:
                # Find the predict_price tool response in recent messages
                for msg in reversed(messages[-4:]):  # Check last few messages (max 3 tools)
                    if msg.get("role") == "tool" and msg.get("name") == "predict_price":
                        prediction_result = json.loads(msg.get("content", "{}"))
                        break

            # Build RAG context using user query and prediction result
            rag_context = build_rag_context(
                query=body.message,
                prediction=prediction_result,
            )

            if rag_context:
                messages.append({
                    "role": "system",
                    "content": (
                        "Use the following retrieved context to enrich your answer. "
                        "When relevant, cite comparable properties and area knowledge. "
                        "Be transparent about model confidence and limitations.\n\n"
                        + rag_context
                    ),
                })
            # ── END RAG INJECTION ─────────────────────────────────────────

            # Panggilan kedua ke GPT untuk memformulasikan jawaban akhir setelah mendapat data
            second_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            final_reply = second_response.choices[0].message.content
        else:
            final_reply = response_message.content

        return ChatResponse(
            reply=final_reply,
            tools_used=tools_used
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
