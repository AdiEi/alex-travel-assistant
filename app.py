from flask import Flask, render_template, request, jsonify
from assistant import TravelAssistant

app = Flask(__name__)
_assistant = TravelAssistant()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    return jsonify({"reply": _assistant.chat(message)})


@app.route("/reset", methods=["POST"])
def reset():
    _assistant.reset()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
