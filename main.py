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

You must respond naturally, without ever commenting on the formatting itself. Never say that you're using Markdown or LaTeX — just format the message accordingly so it can be rendered using `react-markdown` with `remark-math` and `rehype-katex`.
"""


async def text_generation(messages: List[Dict]) -> AsyncGenerator[str, None]:
    client = AsyncClient()
    stream = client.chat.completions.stream(
        model="gpt-4o",
        provider=Copilot,
        messages=[{"content": system_prompt, "role": "system"}] + messages,
        web_search=False,
    )

    async for chunk in stream:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content = delta.content
                yield json.dumps({"content": content, "role": "assistant"}) + "\n"

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