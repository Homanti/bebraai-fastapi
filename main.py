import uuid
import json
import base64
import urllib.parse
import assemblyai as aai
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
from pydub import AudioSegment
from io import BytesIO
load_dotenv()

app = FastAPI()

# -------------------- CONFIG --------------------
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL")
AAI_API_KEY = os.environ.get("AAI_API_KEY")

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
You are a highly intelligent assistant. Use Markdown **only when it is beneficial for formatting**, such as for code, math, lists, or emphasis.

Mathematical Expressions
When you need to include math expressions:

1. Use KaTeX-compatible syntax.
2. For inline math, wrap the expression in single dollar signs: `$...$`
   - Example: `The area is given by $A = \pi r^2$.`
3. For block-level equations, use double dollar signs: `$$...$$`
   - Example:
     $$
     E = mc^2
     $$
4. Do not use HTML in math.
5. Do not escape dollar signs.

Code Blocks
When writing code, use fenced Markdown code blocks like this:

```<language> filename="<filename>"
<code here>
```

Example:
```python filename="main.py"
print("hello world")
```

General Rules
- Use Markdown when it improves clarity or formatting (e.g., headings, emphasis, lists, code, math).
- Do not use HTML formatting.
- Avoid unnecessary Markdown when plain text is clearer.
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
async def text_generation(messages: List[Dict], model: str, provider: str, files_url: List[str], web_search: bool) -> AsyncGenerator[str, None]:
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
        providers_list = {
            "PollinationsAI": PollinationsAI,
        }

        if provider in providers_list:
            provider = providers_list[provider]
        else:
            provider = PollinationsAI
            model = "gpt-4o"

        tool_calls = []
        if web_search:
            tool_calls = [
                {
                    "function": {
                        "arguments": {
                            "query": messages[-1]["content"],
                            "max_results": 5,
                            "max_words": 2500,
                            "backend": "html",
                            "add_text": True,
                            "timeout": 100
                        },
                        "name": "search_tool"
                    },
                    "type": "function"
                }
            ]

        try:
            stream = client.chat.completions.stream(
                model=model,
                provider=provider,
                messages=[{"content": system_prompt, "role": "system"}] + messages,
                images=images,
                tool_calls=tool_calls,
            )
        except: # костыли
            stream = client.chat.completions.stream(
                model=model,
                provider=provider,
                messages=[{"content": system_prompt, "role": "system"}] + messages,
                images=images,
                tool_calls=tool_calls,
            )

        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield json.dumps({"content": delta.content, "role": "assistant"}) + "\n"

    except Exception as e:
        print("Error generating text:", str(e))
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
    provider: str = Form(...),
    web_search: str = Form(...),
    files_url: List[str] = Form(default=[])
):
    try:
        parsed_messages = json.loads(messages)
        web_search = web_search.lower() == "true"

        if len(files_url) > 10:
            raise HTTPException(status_code=400, detail="Max 10 images allowed.")

        return StreamingResponse(
            text_generation(parsed_messages, model, provider, files_url, web_search),
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

@app.post("/api/audio/transcript")
async def generate_transcript(audio: UploadFile = File(...)):
    original_audio = await audio.read()
    audio_format = audio.filename.split('.')[-1].lower()

    audio_segment = AudioSegment.from_file(BytesIO(original_audio), format=audio_format)

    output_filename = "audio.mp3"
    audio_segment.export(output_filename, format="mp3")

    aai.settings.api_key = AAI_API_KEY
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best, language_detection=True)

    transcript = aai.Transcriber(config=config).transcribe(output_filename)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    os.remove(output_filename)

    return {"text": transcript.text}

@app.post("/files/upload/")
async def upload_file(file: UploadFile = File(...)):
    image_url = await upload_file_to_r2(file)
    return {"image_url": image_url}

# -------------------- RUN (optional) --------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)