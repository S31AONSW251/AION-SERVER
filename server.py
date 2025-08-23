# server.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

app = FastAPI()

# Frontend URL (Vercel) – set via environment variable in Render
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://your-vercel-app.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use hosted LLM (OpenAI) or local fallback
USE_HOSTED = os.environ.get("USE_HOSTED_LLM", "true").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

async def query_openai(prompt: str):
    """Query OpenAI (or hosted model)"""
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set on server"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
            },
        )
        r.raise_for_status()
        data = r.json()
        return {"reply": data["choices"][0]["message"]["content"]}

@app.post("/ai")
async def run_ai(request: Request):
    """Main AI endpoint"""
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}

    if USE_HOSTED:
        return await query_openai(prompt)
    else:
        return {"reply": f"(Local AI disabled) Echo: {prompt}"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets $PORT
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
