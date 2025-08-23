# AION Server (FastAPI)

This is the backend server for AION, designed to be deployed on **Render** and connect with the **Vercel frontend**.

---

## 🚀 Setup

### Local Development
```bash
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Test health check:
```bash
curl http://127.0.0.1:8000/health
```

Test AI:
```bash
curl -X POST http://127.0.0.1:8000/ai -H "Content-Type: application/json" -d '{"prompt":"Hello AION"}'
```

---

## 🌐 Deploy on Render

1. Push this repo to GitHub.
2. On [Render](https://render.com):
   - Create **New → Web Service**
   - Connect repo
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
3. Add Environment Variables in Render:
   - `FRONTEND_URL` = `https://your-vercel-app.vercel.app`
   - `USE_HOSTED_LLM` = `true`
   - `OPENAI_API_KEY` = `<your-openai-key>`

After deploy, Render gives you a URL like:
```
https://aion-server.onrender.com
```

---

## 🔗 Connect Vercel Frontend

In your frontend code, fetch from:
```
const AI_SERVER_URL = import.meta.env.VITE_AI_SERVER_URL || "http://localhost:8000";
```

Set in **Vercel → Project Settings → Environment Variables**:
```
VITE_AI_SERVER_URL = https://aion-server.onrender.com
```

Then redeploy frontend.

---

## ✅ Done
- Your Vercel frontend talks to Render backend.
- `/health` returns status.
- `/ai` connects to model (OpenAI or local echo).
