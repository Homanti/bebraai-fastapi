import os
import uuid
import json
import base64
import urllib.parse
import requests
import boto3
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, AsyncGenerator
from botocore.client import Config
from g4f import AsyncClient
from g4f.Provider import PollinationsAI
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# -------------------- CONFIG --------------------
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
    ".jpeg",
    ".jpg",
    ".png",
    ".bmp",
    ".webp",
    ".svg"
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

session = boto3.session.Session()
r2_client = session.client(
    service_name="s3",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    endpoint_url=R2_ENDPOINT_URL,
    config=Config(signature_version="s3v4"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://192.168.0.194:5173",
        "https://bebraai-production.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- SYSTEM PROMPT --------------------
system_prompt = r"""
You are a highly intelligent assistant. Always respond strictly using valid Markdown syntax.

Mathematical Expressions
Whenever you include any math expression:

1. Use KaTeX notation only — never describe math in plain text.
2. For inline math, wrap the expression in single dollar signs: `$...$`
   - Example: `The area is given by $A = \pi r^2$.`
3. For display (block-level) equations, wrap the expression in double dollar signs: `$$...$$`
   - Example:
     $$
     E = mc^2
     $$
4. Never use HTML inside math.
5. Never escape dollar signs — write them as-is.

Code Blocks
If you write code, always use fenced Markdown code blocks with this strict format:

```<language> filename="<filename>"
<code goes here>
```

Example:
```python filename="main.py"
print("hello world")
```

General Rules
- All content must follow valid Markdown rules.
- No HTML formatting is allowed anywhere.
- Always follow the formatting rules exactly as described above.
"""
# -------------------- UTILITY FUNCTIONS --------------------
def get_image_base64(url):
    response = requests.get(url)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "image/png")
    filename = os.path.basename(url.split("?")[0]) or "image.png"

    base64_encoded = base64.b64encode(response.content).decode("utf-8")
    data_url = f"data:{content_type};base64,{base64_encoded}"

    return [data_url, filename]

async def upload_file_to_r2(file: UploadFile) -> str:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    file_size = 0
    chunks = []
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
        chunks.append(chunk)

    file_content = b"".join(chunks)
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"

    r2_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=filename,
        Body=file_content,
        ContentType=file.content_type,
    )

    return f"{R2_PUBLIC_URL}/{urllib.parse.quote(filename)}"

async def upload_image_bytes_to_r2(content: bytes, ext=".png", content_type="image/png") -> str:
    filename = f"{uuid.uuid4().hex}{ext}"

    r2_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=filename,
        Body=content,
        ContentType=content_type,
    )

    return f"{R2_PUBLIC_URL}/{urllib.parse.quote(filename)}"

# -------------------- MAIN LOGIC --------------------
async def text_generation(messages: List[Dict], model: str, files_url: List[str], web_search: bool) -> AsyncGenerator[str, None]:
    web_search = str(web_search).lower() == "true"
    client = AsyncClient()

    images = []
    for url in files_url:
        try:
            data_url, filename = get_image_base64(url)
            images.append([data_url, filename])
        except Exception as e:
            print("Error loading image:", url, str(e))

    try:
        stream = client.chat.completions.stream(
            model=model,
            provider=PollinationsAI,
            messages=[{"content": system_prompt, "role": "system"}] + messages,
            images=images,
            web_search=web_search
        )

        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield json.dumps({"content": delta.content, "role": "assistant"}) + "\n"

    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

async def image_generation(prompt: str) -> str:
    client = AsyncClient()
    try:
        response = await client.images.generate(
            provider=PollinationsAI,
            prompt=prompt,
            model="flux-pro",
            response_format="b64_json",
        )

        if response.data[0].b64_json:
            image_bytes = base64.b64decode(response.data[0].b64_json)

            image_url = await upload_image_bytes_to_r2(image_bytes, ext=".png", content_type="image/png")
            return image_url
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

# -------------------- ROUTES --------------------
@app.post("/api/stream/generate")
async def generate_stream(
    messages: str = Form(...),
    model: str = Form(...),
    web_search: str = Form(...),
    files_url: List[str] = Form(default=[])
):
    try:
        parsed_messages = json.loads(messages)
        web_search = web_search.lower() == "true"

        if len(files_url) > 10:
            raise HTTPException(status_code=400, detail="Max 10 images allowed.")

        return StreamingResponse(
            text_generation(parsed_messages, model, files_url, web_search),
            media_type="application/jsonlines",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/image/generate")
async def generate_image(prompt: str = Form(...)):
    try:
        image_url = await image_generation(prompt)
        return JSONResponse(content={"content": image_url})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/files/upload/")
async def upload_file(file: UploadFile = File(...)):
    image_url = await upload_file_to_r2(file)
    return {"image_url": image_url}

# -------------------- RUN (optional) --------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)