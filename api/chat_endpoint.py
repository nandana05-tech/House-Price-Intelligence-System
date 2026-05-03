import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from rag.context_builder import build_rag_context
from services.predictor import classify_segment, cluster_property, predict_price

router = APIRouter(tags=["Chat Agent"])
client = AsyncOpenAI()

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="Pesan terbaru dari user")
    history: list[ChatMessage] = Field(default=[], description="History percakapan sebelumnya")

class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_price",
            "description": "Estimasi harga rumah di Depok. Wajib digunakan jika user menanyakan harga properti atau estimasi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kamar_tidur":   {"type": "integer", "description": "Jumlah kamar tidur"},
                    "kamar_mandi":   {"type": "integer", "description": "Jumlah kamar mandi"},
                    "garasi":        {"type": "integer", "description": "Kapasitas garasi (jumlah mobil)"},
                    "luas_tanah":    {"type": "number",  "description": "Luas tanah dalam meter persegi (m2)"},
                    "luas_bangunan": {"type": "number",  "description": "Luas bangunan dalam meter persegi (m2)"},
                    "lokasi":        {"type": "string",  "description": "Nama kecamatan atau lokasi di Depok"}
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
                    "kamar_tidur":   {"type": "integer"},
                    "kamar_mandi":   {"type": "integer"},
                    "garasi":        {"type": "integer"},
                    "luas_tanah":    {"type": "number"},
                    "luas_bangunan": {"type": "number"},
                    "lokasi":        {"type": "string"},
                    "harga":         {"type": "number", "description": "Harga properti dalam Rupiah (Opsional)"}
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
                    "kamar_tidur":   {"type": "integer"},
                    "kamar_mandi":   {"type": "integer"},
                    "luas_tanah":    {"type": "number"},
                    "luas_bangunan": {"type": "number"},
                    "lokasi":        {"type": "string"},
                    "harga":         {"type": "number", "description": "Harga properti dalam Rupiah (Opsional)"}
                },
                "required": ["kamar_tidur", "kamar_mandi", "luas_tanah", "luas_bangunan", "lokasi"]
            }
        }
    }
]

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(body: ChatRequest) -> ChatResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY tidak ditemukan di environment.")

    # Bangun messages dengan history percakapan
    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah AI asisten real estate yang sangat pintar khusus untuk area Depok. "
                "Anda memiliki akses ke berbagai model machine learning melalui function calling. "
                "Bantu user untuk menaksir harga rumah, mengklasifikasikan segmen, atau menentukan klaster properti. "
                "Selalu berikan jawaban dalam bahasa Indonesia yang natural, ramah, dan informatif. "
                "Jika Anda menggunakan tool prediksi, sertakan informasi harga atau segmen di jawaban Anda dengan jelas. "
                "PENTING: Ingat selalu konteks percakapan sebelumnya. "
                "Jika user sudah memberikan detail properti sebelumnya, gunakan informasi tersebut langsung tanpa bertanya ulang."
            )
        }
    ]

    # Tambahkan history percakapan
    for msg in body.history:
        messages.append({"role": msg.role, "content": msg.content})

    # Tambahkan pesan terbaru
    messages.append({"role": "user", "content": body.message})

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        tools_used = []

        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                tools_used.append(function_name)

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

            # RAG injection
            prediction_result = None
            if "predict_price" in tools_used:
                for msg in reversed(messages[-4:]):
                    if msg.get("role") == "tool" and msg.get("name") == "predict_price":
                        prediction_result = json.loads(msg.get("content", "{}"))
                        break

            rag_context = build_rag_context(
                query=body.message,
                prediction=prediction_result,
            )

            if rag_context:
                messages.append({
                    "role": "system",
                    "content": (
                        "Gunakan konteks berikut untuk memperkaya jawaban Anda. "
                        "Sebutkan properti pembanding dan pengetahuan area jika relevan.\n\n"
                        + rag_context
                    ),
                })

            second_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            final_reply = second_response.choices[0].message.content
        else:
            final_reply = response_message.content

        return ChatResponse(reply=final_reply, tools_used=tools_used)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))