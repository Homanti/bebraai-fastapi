from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, AsyncGenerator
import json
from g4f import AsyncClient
from g4f.Provider import PollinationsAI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.0.194:5173",
        "https://bebraai-production.up.railway.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
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

async def generation(messages: List[Dict], model, mods, files) -> AsyncGenerator[str, None]:
    parsed_mods = json.loads(mods)
    client = AsyncClient()

    if not parsed_mods.get("draw", False):
        try:
            stream = client.chat.completions.stream(
                model=model,
                provider=PollinationsAI,
                messages=[{"content": system_prompt, "role": "system"}] + messages,
                images=files,
                web_search=parsed_mods.get("web_search", False)
            )

            async for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content = delta.content
                        yield json.dumps({"content": content, "role": "assistant"}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

@app.post("/api/stream/generate")
async def generate_stream(
    messages: str = Form(...),
    model: str = Form(...),
    mods: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    try:
        parsed_messages = json.loads(messages)

        processed_files = []
        for f in files:
            content = await f.read()
            processed_files.append([content, f.filename])

        return StreamingResponse(
            generation(parsed_messages, model, mods, processed_files),
            media_type="application/jsonlines"
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)