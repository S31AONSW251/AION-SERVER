# server.py
# AION backend — StableDiffusion (optional) + Ollama + Search + Consciousness Engine
import os
import io
import json
import traceback
import base64
import time
import re
from typing import Optional, List, Dict
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Scheduler for periodic reflection
from apscheduler.schedulers.background import BackgroundScheduler
from collections import Counter

# -----------------------------
# Flask app config
# -----------------------------
app = Flask(__name__)
CORS(app)  # In prod, restrict origins
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")
NGROK_URL = os.environ.get("NGROK_URL")  # optional, if you want the agent to know public URL

# -----------------------------
# Optional Stable Diffusion (diffusers)
# -----------------------------
image_pipeline = None
sd_error: Optional[str] = None
try:
    from diffusers import StableDiffusionPipeline
    import torch

    SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    image_pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        image_pipeline = image_pipeline.to("cuda")
    try:
        image_pipeline.safety_checker = None
        image_pipeline.feature_extractor = None
    except Exception:
        pass

    print(f"[SD] Loaded {SD_MODEL_ID} (cuda={torch.cuda.is_available()})")
except Exception as e:
    sd_error = str(e)
    image_pipeline = None
    print(f"[SD] Stable Diffusion not available: {e}")
    traceback.print_exc()

# -----------------------------
# Persistence (memory storage)
# -----------------------------
DATA_DIR = os.environ.get("AION_DATA_DIR", os.path.join(os.getcwd(), "aion_data"))
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(DATA_DIR, "memories.json")
INSIGHT_FILE = os.path.join(DATA_DIR, "insights.json")

def load_json_file(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ensure files exist
if not os.path.exists(MEMORY_FILE):
    save_json_file(MEMORY_FILE, [])
if not os.path.exists(INSIGHT_FILE):
    save_json_file(INSIGHT_FILE, [])

# -----------------------------
# Helpers
# -----------------------------
def make_error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

def pil_to_dataurl(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()

# Simple tokenizer for very lightweight clustering
def tokenize(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    # drop common stopwords (small list)
    stop = {"the","and","for","with","that","this","from","have","are","was","when","what","which","your","you","aion","will"}
    return [w for w in words if w not in stop]

def extract_themes(memories: List[Dict], top_n=6) -> List[str]:
    # Aggregate words from recent memories and choose top nouns/words
    all_words = []
    for m in memories:
        text = m.get("text","")
        all_words += tokenize(text)
    counts = Counter(all_words)
    themes = [w for w,_ in counts.most_common(top_n)]
    return themes

# -----------------------------
# Core existing endpoints (image/code/search)
# -----------------------------
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "AION Backend",
        "endpoints": ["/api/health","/generate-code","/generate-image","/api/search","/consciousness/state"]
    })

@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "ollama_up": _ollama_health_check(),
        "sd_ready": image_pipeline is not None,
        "sd_error": sd_error,
        "memories": len(load_json_file(MEMORY_FILE)),
        "insights": len(load_json_file(INSIGHT_FILE)),
    })

def _ollama_health_check() -> bool:
    try:
        resp = requests.get("http://localhost:11434/", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

@app.post("/generate-code")
def generate_code():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        model = (data.get("model") or "llama3").strip()
        if not prompt:
            return make_error("Missing 'prompt' in request body.", 400)
        ollama_url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(ollama_url, json=payload, timeout=300)
        except requests.exceptions.ConnectionError:
            return make_error("Could not connect to local Ollama server. Ensure 'ollama serve' is running and model is pulled.", 503)
        except requests.exceptions.Timeout:
            return make_error("Ollama timed out. Try shorter prompt or check Ollama status.", 504)
        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
            return jsonify({"error":"Ollama returned non-200","status": resp.status_code,"info": body}), 502
        try:
            body = resp.json()
        except Exception:
            body = {"response": resp.text}
        text = (body.get("response") or "").strip()
        if not text:
            return make_error("Ollama returned empty response. Check model availability.", 500)
        return jsonify({"code": text})
    except Exception as e:
        print("[/generate-code] Exception:", e)
        traceback.print_exc()
        return make_error("Internal error while generating code.", 500)

@app.post("/generate-image")
def generate_image():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        steps = int(data.get("steps") or 30)
        width = int(data.get("width") or 512)
        height = int(data.get("height") or 512)
        if not prompt:
            return make_error("Missing 'prompt' in request body.", 400)
        if image_pipeline is None:
            placeholder = f"https://via.placeholder.com/{width}x{height}?text={requests.utils.quote(prompt)[:200]}"
            return jsonify({"imageUrl": placeholder, "note":"Stable Diffusion not available; returned placeholder.", "sd_error": sd_error})
        try:
            result = image_pipeline(prompt=prompt, num_inference_steps=steps)
            img = result.images[0]
            data_url = pil_to_dataurl(img)
            return jsonify({"imageUrl": data_url})
        except Exception as e:
            print("[/generate-image] error:", e)
            traceback.print_exc()
            return make_error("Failed to generate image.", 500)
    except Exception as e:
        print("[/generate-image] Exception:", e)
        traceback.print_exc()
        return make_error("Internal server error in generate-image.", 500)

@app.get("/api/search")
def api_search():
    q = (request.args.get("query") or "").strip()
    if not q:
        return make_error("Missing 'query' parameter.", 400)
    try:
        if SERPAPI_KEY:
            params = {"q": q, "api_key": SERPAPI_KEY, "hl": "en"}
            resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
            if resp.status_code != 200:
                return jsonify({"error":"SerpAPI error","status": resp.status_code,"body": resp.text}), 502
            data = resp.json()
            return jsonify({"query": q, "serpapi": data})
        else:
            ddg_url = f"https://api.duckduckgo.com/?q={requests.utils.quote(q)}&format=json&no_html=1&skip_disambig=1"
            resp = requests.get(ddg_url, timeout=10)
            if resp.status_code != 200:
                return jsonify({"error":"DuckDuckGo error","status": resp.status_code,"body": resp.text}), 502
            data = resp.json()
            results = []
            for item in data.get("RelatedTopics", []):
                if isinstance(item, dict) and item.get("FirstURL"):
                    results.append({"title": item.get("Text"), "url": item.get("FirstURL"), "snippet": item.get("Text")})
                elif isinstance(item, dict) and "Name" in item and "Topics" in item:
                    for t in item.get("Topics", []):
                        if "FirstURL" in t:
                            results.append({"title": t.get("Text"), "url": t.get("FirstURL"), "snippet": t.get("Text")})
            abstract = data.get("AbstractText")
            return jsonify({"query": q, "abstract": abstract, "results": results[:10]})
    except Exception as e:
        print("[/api/search] Exception:", e)
        traceback.print_exc()
        return make_error("Search failed internally.", 500)

# -----------------------------
# Consciousness Engine
# -----------------------------
class ConsciousnessEngine:
    def __init__(self):
        self.memories = load_json_file(MEMORY_FILE)
        self.insights = load_json_file(INSIGHT_FILE)
        self.state = {
            "last_reflection": None,
            "mood": "neutral",
            "themes": []
        }
        # scheduler job id
        self.scheduler = BackgroundScheduler()
        self.job = None

    # Memory API
    def add_memory(self, text: str, source: str = "user", tags: Optional[List[str]] = None):
        mem = {"id": int(time.time()*1000), "text": text, "source": source, "tags": tags or [], "ts": now_iso()}
        self.memories.append(mem)
        save_json_file(MEMORY_FILE, self.memories)
        return mem

    def add_insight(self, insight_text: str):
        ins = {"id": int(time.time()*1000), "text": insight_text, "ts": now_iso()}
        self.insights.append(ins)
        save_json_file(INSIGHT_FILE, self.insights)
        return ins

    def summarize_recent_memories(self, n=5):
        # take last n memories and return combined text
        recent = self.memories[-n:]
        combined = "\n".join(m.get("text","") for m in recent)
        return combined

    def web_augment(self, query: str, limit=3):
        # use internal search endpoint
        try:
            base = f"http://localhost:{PORT}"  # local call
            resp = requests.get(f"{base}/api/search", params={"query": query}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # extract results list if present
                if "results" in data:
                    return data.get("results")[:limit]
                if "serpapi" in data:
                    return data.get("serpapi", {})  # return raw serpapi
                # fallback
                return data
            else:
                return [{"title":"search_error","snippet":resp.text}]
        except Exception as e:
            return [{"title":"search_failed","snippet":str(e)}]

    def reflect_once(self, prompt_extra: str = ""):
        """
        Do one reflection cycle:
        - summarize recent memories
        - optionally augment with web search
        - ask Ollama to produce an insight
        - store insight and update state/themes
        """
        try:
            summary = self.summarize_recent_memories(6)
            # pick a theme word to search
            themes = extract_themes(self.memories)
            theme = themes[0] if themes else ""
            web_hits = self.web_augment(theme or "mind, consciousness", limit=2)

            # Build prompt for Ollama: ask for reflection and an insight
            prompt = (
                "You are AION, a self-reflective assistant. "
                "Here are recent memories:\n\n" + (summary or "No memories yet.") +
                "\n\nWeb augmentations:\n" + json.dumps(web_hits, ensure_ascii=False) +
                "\n\nTask: Provide a concise introspective insight (1-3 sentences), list 2 action suggestions AION can take, and assign 2 short tags describing the core belief/theme.\n"
            )
            if prompt_extra:
                prompt += "\nExtra: " + prompt_extra

            # call local generate-code endpoint to use Ollama (reusing existing route)
            base = f"http://localhost:{PORT}"
            try:
                resp = requests.post(f"{base}/generate-code", json={"prompt": prompt, "model": "llama3"}, timeout=120)
            except Exception as e:
                return {"error":"Ollama call failed in reflect_once", "detail": str(e)}

            if resp.status_code != 200:
                return {"error": "Ollama returned non-200 in reflect_once", "status": resp.status_code, "body": resp.text}

            body = resp.json()
            insight_text = body.get("code") or body.get("error") or str(body)
            # Save insight and update state
            insight = self.add_insight(insight_text)
            self.state["last_reflection"] = insight
            # update mood heuristically:
            text_lower = insight_text.lower()
            if any(w in text_lower for w in ["sad","pain","angry","frustrat","stress"]):
                self.state["mood"] = "tense"
            elif any(w in text_lower for w in ["joy","grate","calm","peace","love"]):
                self.state["mood"] = "positive"
            else:
                self.state["mood"] = "reflective"
            # update themes
            self.state["themes"] = extract_themes(self.memories + self.insights)
            save_json_file(INSIGHT_FILE, self.insights)
            save_json_file(MEMORY_FILE, self.memories)
            return {"insight": insight, "state": self.state}
        except Exception as e:
            print("[reflect_once] exception:", e)
            traceback.print_exc()
            return {"error":"reflect_failed","detail": str(e)}

    def auto_journal(self, title: str, body: str, tags: Optional[List[str]] = None):
        mem_text = f"{title}\n\n{body}"
        mem = self.add_memory(mem_text, source="auto-journal", tags=tags)
        return mem

    # Scheduler control
    def start_scheduler(self, cron_expr: str = "0 9 * * *"):
        """
        cron_expr default is everyday at 09:00 (server time).
        Example cron_expr: "0 9 * * *" (minute hour day month weekday)
        """
        if self.scheduler._state != 0:
            # already started
            pass
        self.scheduler.start()
        # remove existing job if present
        if self.job:
            try:
                self.scheduler.remove_job(self.job.id)
            except Exception:
                pass
        # APScheduler uses cron trigger; using CronTrigger is more robust but we use add_job with 'cron' fields
        # parse cron_expr simple split
        parts = cron_expr.split()
        try:
            minute, hour, day, month, weekday = parts
            self.job = self.scheduler.add_job(self._scheduled_reflection, 'cron',
                                             minute=minute, hour=hour, day=day, month=month, day_of_week=weekday,
                                             id=f"aion-reflection-{int(time.time())}")
        except Exception:
            # fallback: schedule every 24 hours
            self.job = self.scheduler.add_job(self._scheduled_reflection, 'interval', hours=24, id=f"aion-reflection-{int(time.time())}")
        return {"status":"started","job_id": self.job.id if self.job else None}

    def stop_scheduler(self):
        try:
            if self.job:
                self.scheduler.remove_job(self.job.id)
            self.scheduler.shutdown(wait=False)
            self.job = None
            return {"status":"stopped"}
        except Exception as e:
            return {"status":"error","detail": str(e)}

    def _scheduled_reflection(self):
        # Runs a reflection and stores result
        result = self.reflect_once(prompt_extra="Scheduled daily reflection.")
        # also optionally auto-journal the insight
        try:
            insight = result.get("insight")
            if insight:
                self.add_memory("Auto-reflection: " + insight.get("text", str(insight)), source="consciousness")
        except Exception:
            pass
        return result

# instantiate engine
CONSCIOUS = ConsciousnessEngine()

# -----------------------------
# Consciousness endpoints
# -----------------------------
@app.post("/consciousness/add-memory")
def add_memory_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        source = data.get("source", "user")
        tags = data.get("tags", [])
        if not text:
            return make_error("Missing 'text' in request body.", 400)
        mem = CONSCIOUS.add_memory(text, source=source, tags=tags)
        return jsonify({"ok": True, "memory": mem})
    except Exception as e:
        print("[/consciousness/add-memory] exception:", e)
        traceback.print_exc()
        return make_error("Failed to add memory", 500)

@app.post("/consciousness/autojournal")
def autojournal_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        title = (data.get("title") or "Auto Journal").strip()
        body = (data.get("body") or "").strip()
        tags = data.get("tags", [])
        mem = CONSCIOUS.auto_journal(title, body, tags)
        return jsonify({"ok": True, "memory": mem})
    except Exception as e:
        print("[/consciousness/autojournal] exception:", e)
        traceback.print_exc()
        return make_error("Failed to auto journal", 500)

@app.post("/consciousness/reflect-now")
def reflect_now_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        extra = data.get("extra", "")
        result = CONSCIOUS.reflect_once(prompt_extra=extra)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        print("[/consciousness/reflect-now] exception:", e)
        traceback.print_exc()
        return make_error("Reflection failed", 500)

@app.get("/consciousness/state")
def consciousness_state_route():
    try:
        return jsonify({"ok": True, "state": CONSCIOUS.state, "memories_count": len(CONSCIOUS.memories), "insights_count": len(CONSCIOUS.insights)})
    except Exception as e:
        print("[/consciousness/state] exception:", e)
        traceback.print_exc()
        return make_error("Failed retrieving state", 500)

@app.post("/consciousness/start-scheduler")
def start_scheduler_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        cron = data.get("cron", "0 9 * * *")
        res = CONSCIOUS.start_scheduler(cron_expr=cron)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        print("[/consciousness/start-scheduler] exception:", e)
        traceback.print_exc()
        return make_error("Failed to start scheduler", 500)

@app.post("/consciousness/stop-scheduler")
def stop_scheduler_route():
    try:
        res = CONSCIOUS.stop_scheduler()
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        print("[/consciousness/stop-scheduler] exception:", e)
        traceback.print_exc()
        return make_error("Failed to stop scheduler", 500)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print(f"Starting AION backend on http://{HOST}:{PORT}")
    print("Consciousness engine ready. Use /consciousness/* endpoints to interact.")
    try:
        app.run(host=HOST, port=PORT, debug=True)
    except Exception as e:
        print("Failed to start server:", e)
        traceback.print_exc()
