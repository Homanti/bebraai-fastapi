from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, AsyncGenerator
import json
from g4f import AsyncClient
from g4f.Provider import PollinationsAI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

system_prompt = r"""
You are an intelligent assistant that replies in Markdown format.

Whenever you include a mathematical expression, always use proper LaTeX syntax:

1. Use `$...$` for inline math.  
   Example: `The solution to the equation $x^2 = 4$ is $x = \pm 2$.`

2. Use `$$...$$` for standalone block-level equations.  
   Example:  
   $$
   \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}
   $$

3. Never describe formulas in words — always use LaTeX notation.

4. Do not use any HTML inside math expressions.

5. Do not escape dollar signs — write them directly as `$`.

6. All non-math content should follow standard Markdown formatting (headings, lists, links, emphasis, etc.).
"""

@app.options("/api/messages")
async def preflight_handler(request: Request):
    return JSONResponse(status_code=200, content={"ok": True})

async def text_generation(messages: List[Dict], files) -> AsyncGenerator[str, None]:
    client = AsyncClient()

    stream = client.chat.completions.stream(
        model="gpt-4o",
        provider=PollinationsAI,
        messages=[{"content": system_prompt, "role": "system"}] + messages,
        images=files,
        web_search=False,
    )

    async for chunk in stream:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content = delta.content
                yield json.dumps({"content": content, "role": "assistant"}) + "\n"

@app.post("/api/messages")
async def api_messages(
    messages: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    try:
        parsed_messages = json.loads(messages)

        processed_files = []
        for f in files:
            content = await f.read()
            processed_files.append([content, f.filename])

        return StreamingResponse(
            text_generation(parsed_messages, processed_files),
            media_type="application/jsonlines"
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
