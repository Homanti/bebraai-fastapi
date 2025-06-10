import os
import uuid
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, AsyncGenerator
import json
from g4f import AsyncClient
from g4f.Provider import PollinationsAI
import boto3
from botocore.client import Config
import urllib.parse
import base64
import requests
app = FastAPI()

R2_ACCESS_KEY_ID = "8c79b2dd67ba7ffe39bf8c793a184936"
R2_SECRET_ACCESS_KEY = "b28fb6432d92288f528618405d6b4dd6582c0096fb286c50e237f8cb5428a3f3"
R2_BUCKET_NAME = "bebraai"
R2_ENDPOINT_URL = "https://f720b16b60c09e5011578d67e56ed282.r2.cloudflarestorage.com"

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

def get_image_base64(url):
    response = requests.get(url)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "image/png")

    content_disposition = response.headers.get("Content-Disposition")
    if content_disposition and "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[1].strip("\"")
    else:
        filename = os.path.basename(url.split("?")[0]) or "image.png"

    base64_encoded = base64.b64encode(response.content).decode("utf-8")
    data_url = f"data:{content_type};base64,{base64_encoded}"

    return [data_url, filename]

async def generation(messages: List[Dict], model, mods, files_url) -> AsyncGenerator[str, None]:
    parsed_mods = json.loads(mods)
    client = AsyncClient()

    images = []
    for url in files_url:
        try:
            data_url, filename = get_image_base64(url)
            images.append([data_url, filename])
        except Exception as e:
            print("Error image loading:", url, str(e))

    if not parsed_mods.get("draw", False):
        try:
            stream = client.chat.completions.stream(
                model=model,
                provider=PollinationsAI,
                messages=[{"content": system_prompt, "role": "system"}] + messages,
                images=images,
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
    files_url: List[str] = Form(default=[])
):
    try:
        parsed_messages = json.loads(messages)

        if len(files_url) > 10:
            raise HTTPException(status_code=400, detail="You can upload a maximum of 10 images.")

        return StreamingResponse(
            generation(parsed_messages, model, mods, files_url),
            media_type="application/jsonlines"
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
@app.post("/files/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only image files are allowed (jpeg, png, gif, webp).")

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
    random_filename = f"{uuid.uuid4().hex}{ext}"

    r2_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=random_filename,
        Body=file_content,
        ContentType=file.content_type,
    )

    filename = urllib.parse.quote(random_filename)
    public_url = f"https://pub-bbbd9fe0ee484f02954722c5d466e7c0.r2.dev/{filename}"
    return {"url": public_url}

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)