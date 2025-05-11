import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, AsyncGenerator
import json
from g4f import AsyncClient
from g4f.Provider import Copilot

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def text_generation(messages: List[Dict]) -> AsyncGenerator[str, None]:
    client = AsyncClient()
    stream = client.chat.completions.stream(
        model="gpt-4o",
        provider=Copilot,
        messages=messages,
        web_search=False,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            yield json.dumps({ "content": content, "role": "assistant" }) + "\n"

@app.post("/api/messages")
async def api_messages(request: Request):
    try:
        body = await request.json()

        if not isinstance(body, list):
            return {"error": "Request body must be a list of messages."}

        for msg in body:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return {"error": "Each message must be a dict with 'role' and 'content'."}

        return StreamingResponse(text_generation(body), media_type="application/jsonlines")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)