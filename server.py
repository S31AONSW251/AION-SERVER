import os
import traceback
from flask import Flask, request, jsonify
from consciousness_engine import ConsciousnessEngine

app = Flask(__name__)

# Initialize your AI brain
CONSCIOUS = ConsciousnessEngine()

# -----------------------------
# Existing routes (kept intact)
# -----------------------------
@app.post("/generate-code")
def generate_code():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400
        # your code generation logic here
        result = {"code": f"# generated code for: {prompt}"}
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/generate-image")
def generate_image():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400
        # your image generation logic here
        result = {"url": f"https://dummyimage.com/600x400/000/fff&text={prompt}"}
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/api/search")
def api_search():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing query"}), 400
        # your search logic here
        result = {"results": [f"Result 1 for {query}", f"Result 2 for {query}"]}
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Consciousness-related routes
# -----------------------------
@app.get("/consciousness/memories")
def get_memories():
    try:
        return jsonify(CONSCIOUS.get_recent_memories())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/consciousness/reflect")
def reflect():
    try:
        data = request.get_json(force=True, silent=True) or {}
        extra = data.get("prompt_extra") or ""
        result = CONSCIOUS.reflect_once(prompt_extra=extra)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# New endpoint for App.js chat
# -----------------------------
@app.post("/query")
def query_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_input = (data.get("text") or "").strip()
        if not user_input:
            return jsonify({"response": "⚠️ Missing 'text' in request body."}), 400

        # Store memory
        CONSCIOUS.add_memory(user_input, source="user")

        # Reflect with context
        result = CONSCIOUS.reflect_once(prompt_extra=f"User said: {user_input}")

        # Extract reply
        reply = None
        if isinstance(result, dict):
            if "insight" in result and isinstance(result["insight"], dict):
                reply = result["insight"].get("text")
            elif "error" in result:
                reply = f"⚠️ Error: {result['error']}"
        if not reply:
            reply = "🤖 AION could not generate a proper response."

        # Save reply
        CONSCIOUS.add_memory(reply, source="assistant")

        return jsonify({"response": reply})
    except Exception as e:
        print("[/query] exception:", e)
        traceback.print_exc()
        return jsonify({"response": "⚠️ Internal error in /query"}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
