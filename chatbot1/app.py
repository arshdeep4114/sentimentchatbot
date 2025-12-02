import os
import traceback
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from textblob import TextBlob

# New imports for optimized sentiment analyzer
import functools
import math
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as Vader
except Exception:
    Vader = None

try:
    from transformers import pipeline, Pipeline
    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False

load_dotenv()

try:
    from groq import Groq
except Exception as e:
    raise ImportError("Install groq SDK: pip install groq") from e

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set. Put it in .env or export in env.")

client = Groq(api_key=API_KEY)
CURRENT_MODEL = {"name": os.getenv("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")}

# In-memory conversation storage
conversations = {}


# --- NEW SentimentAnalyzer (hybrid + cache + neutral handling) ---
class SentimentAnalyzer:
    """
    Hybrid Sentiment Analyzer with improved NEUTRAL detection.
      - Fast path: VADER.
      - Fallback: transformer classifier.
      - Final fallback: TextBlob.
      - LRU cache still used. Neutral detection uses combined signals and thresholds.
    """

    _vader = None
    _hf_pipeline: "Pipeline | None" = None
    # keep existing model; transformer may not emit 'neutral' for SST-2,
    # so we use confidence thresholds to decide 'Neutral'
    _model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    _HF_CONF_THRESHOLD = 0.65  # require this score to trust HF decision
    _VADER_NEUTRAL_BAND = 0.2  # compound between -0.2 and 0.2 considered neutral-leaning
    _TEXTBLOB_NEUTRAL_POLARITY = 0.05
    _TEXTBLOB_NEUTRAL_SUBJECTIVITY = 0.35

    @classmethod
    def _init_vader(cls):
        if cls._vader is None and Vader is not None:
            try:
                cls._vader = Vader()
            except Exception:
                cls._vader = None

    @classmethod
    def _init_hf_pipeline(cls):
        if cls._hf_pipeline is None and _HAVE_TRANSFORMERS:
            try:
                cls._hf_pipeline = pipeline("sentiment-analysis", model=cls._model_name, truncation=True)
            except Exception:
                cls._hf_pipeline = None

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _cached_analyze(text_normalized: str) -> Dict:
        return SentimentAnalyzer._analyze_uncached(text_normalized)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 2 / (1 + math.exp(-5 * x)) - 1

    @classmethod
    def analyze_statement(cls, text: str) -> Dict:
        if not text or not text.strip():
            return {
                "text": text,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "Neutral",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        key = text.strip()
        return cls._cached_analyze(key)

    @classmethod
    def _analyze_uncached(cls, text: str) -> Dict:
        cls._init_vader()
        cls._init_hf_pipeline()

        # TextBlob subjectivity/polarity (used as fallback or tie-breaker)
        try:
            tb = TextBlob(text)
            tb_polarity = round(tb.sentiment.polarity, 3)
            tb_subjectivity = round(tb.sentiment.subjectivity, 3)
        except Exception:
            tb_polarity = 0.0
            tb_subjectivity = 0.0

        # VADER quick scores (if available)
        vader_scores = None
        if cls._vader:
            try:
                vader_scores = cls._vader.polarity_scores(text)
            except Exception:
                vader_scores = None

        # 1) If VADER strong positive/negative, accept quickly
        if vader_scores:
            compound = vader_scores.get("compound", 0.0)
            if compound >= 0.6:
                return {
                    "text": text,
                    "polarity": round(compound, 3),
                    "subjectivity": tb_subjectivity,
                    "label": "Positive",
                    "confidence": round(vader_scores.get("pos", 0.6), 3),
                    "timestamp": datetime.now().isoformat()
                }
            if compound <= -0.6:
                return {
                    "text": text,
                    "polarity": round(compound, 3),
                    "subjectivity": tb_subjectivity,
                    "label": "Negative",
                    "confidence": round(vader_scores.get("neg", 0.6), 3),
                    "timestamp": datetime.now().isoformat()
                }

        # 2) If VADER is neutral-leaning, mark as candidate neutral unless HF strongly disagrees
        vader_neutral_candidate = False
        if vader_scores and abs(vader_scores.get("compound", 0.0)) <= cls._VADER_NEUTRAL_BAND:
            vader_neutral_candidate = True

        # 3) Try HF transformer if available (may be more accurate on ambiguous text)
        hf_result = None
        if cls._hf_pipeline:
            try:
                hf_result = cls._hf_pipeline(text[:512])
                if isinstance(hf_result, list) and len(hf_result) > 0:
                    hf_result = hf_result[0]
            except Exception:
                hf_result = None

        hf_low_confidence = False
        if hf_result and "label" in hf_result and "score" in hf_result:
            lbl = hf_result["label"].lower()
            score = float(hf_result["score"])
            confidence = round(score, 3)

            # Only accept HF label if confidence high enough; else mark low-confidence and continue
            if score >= cls._HF_CONF_THRESHOLD:
                if lbl.startswith("pos"):
                    polarity = round(cls._sigmoid(score - 0.5), 3)
                    return {
                        "text": text,
                        "polarity": polarity,
                        "subjectivity": tb_subjectivity,
                        "label": "Positive",
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                elif lbl.startswith("neg"):
                    polarity = round(-cls._sigmoid(score - 0.5), 3)
                    return {
                        "text": text,
                        "polarity": polarity,
                        "subjectivity": tb_subjectivity,
                        "label": "Negative",
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # transformer explicitly says neutral (rare for SST-2 models)
                    return {
                        "text": text,
                        "polarity": 0.0,
                        "subjectivity": tb_subjectivity,
                        "label": "Neutral",
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                hf_low_confidence = True

        # 4) Heuristic combination for Neutral:
        # - VADER neutral band OR HF low confidence OR TextBlob near-zero polarity + low subjectivity
        tb_neutral = (abs(tb_polarity) <= cls._TEXTBLOB_NEUTRAL_POLARITY and
                      tb_subjectivity <= cls._TEXTBLOB_NEUTRAL_SUBJECTIVITY)

        if vader_neutral_candidate or hf_low_confidence or tb_neutral:
            # Determine a sensible polarity estimate (prefer VADER compound if exists)
            est_polarity = 0.0
            if vader_scores:
                est_polarity = round(vader_scores.get("compound", 0.0), 3)
            else:
                est_polarity = round(tb_polarity, 3)

            # Confidence for neutral: inverse of absolute polarity (higher when near zero) clipped
            neutral_confidence = round(max(0.25, 1 - min(1.0, abs(est_polarity))), 3)
            return {
                "text": text,
                "polarity": est_polarity,
                "subjectivity": tb_subjectivity,
                "label": "Neutral",
                "confidence": neutral_confidence,
                "timestamp": datetime.now().isoformat()
            }

        # 5) If we reached here, neither VADER nor HF decided neutral; use HF if present with score
        if hf_result and "label" in hf_result and "score" in hf_result:
            lbl = hf_result["label"].lower()
            score = float(hf_result["score"])
            confidence = round(score, 3)
            if lbl.startswith("pos"):
                polarity = round(cls._sigmoid(score - 0.5), 3)
                return {
                    "text": text,
                    "polarity": polarity,
                    "subjectivity": tb_subjectivity,
                    "label": "Positive",
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
            elif lbl.startswith("neg"):
                polarity = round(-cls._sigmoid(score - 0.5), 3)
                return {
                    "text": text,
                    "polarity": polarity,
                    "subjectivity": tb_subjectivity,
                    "label": "Negative",
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }

        # 6) Final fallback: TextBlob polarity-based label, but include Neutral as possible
        try:
            tb2 = TextBlob(text)
            pattern_polarity = round(tb2.sentiment.polarity, 3)
            pattern_subjectivity = round(tb2.sentiment.subjectivity, 3)
            if abs(pattern_polarity) <= cls._TEXTBLOB_NEUTRAL_POLARITY:
                label = "Neutral"
                confidence = round(max(0.25, 1 - abs(pattern_polarity)), 3)
            elif pattern_polarity > 0:
                label = "Positive"
                confidence = 0.5
            else:
                label = "Negative"
                confidence = 0.5

            return {
                "text": text,
                "polarity": pattern_polarity,
                "subjectivity": pattern_subjectivity,
                "label": label,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return {
                "text": text,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "Neutral",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    @staticmethod
    def analyze_conversation(sentiments: List[Dict]) -> Dict:
        if not sentiments:
            return {"error": "No sentiments to analyze"}

        polarities = [s.get("polarity", 0.0) for s in sentiments]
        avg_polarity = round(sum(polarities) / len(polarities), 3)
        min_p = round(min(polarities), 3)
        max_p = round(max(polarities), 3)

        if len(polarities) < 2:
            trend = "Stable"
        else:
            half = len(polarities) // 2
            first_half = sum(polarities[:half]) / max(1, half)
            second_half = sum(polarities[half:]) / max(1, len(polarities) - half)
            diff = second_half - first_half
            if diff > 0.15:
                trend = "Improving"
            elif diff < -0.15:
                trend = "Declining"
            else:
                trend = "Stable"

        max_change = 0.0
        if len(polarities) >= 2:
            max_change = max(abs(polarities[i] - polarities[i - 1]) for i in range(1, len(polarities)))

        if max_change > 0.5:
            sentiment_shift = f"Significant shift (change: {round(max_change, 2)})"
        elif max_change > 0.3:
            sentiment_shift = f"Moderate shift (change: {round(max_change, 2)})"
        else:
            sentiment_shift = "Consistent mood"

        if avg_polarity > 0.05:
            overall_label = "Positive"
        elif avg_polarity < -0.05:
            overall_label = "Negative"
        else:
            overall_label = "Neutral"

        distribution = {
            "positive": sum(1 for s in sentiments if s.get("label") == "Positive"),
            "neutral": sum(1 for s in sentiments if s.get("label") == "Neutral"),
            "negative": sum(1 for s in sentiments if s.get("label") == "Negative"),
        }

        return {
            "overall_sentiment": overall_label,
            "average_polarity": avg_polarity,
            "polarity_range": {"min": min_p, "max": max_p},
            "trend": trend,
            "sentiment_shift": sentiment_shift,
            "total_messages": len(sentiments),
            "distribution": distribution
        }
# --- END SentimentAnalyzer ---


class ConversationManager:
    """Manages conversation state and history."""

    @staticmethod
    def create_session(session_id: str):
        """Initialize a new conversation session."""
        conversations[session_id] = {
            "messages": [],
            "sentiments": [],
            "created_at": datetime.now().isoformat()
        }

    @staticmethod
    def add_message(session_id: str, role: str, content: str, sentiment: Dict = None):
        """Add a message to conversation history."""
        if session_id not in conversations:
            ConversationManager.create_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        conversations[session_id]["messages"].append(message)

        if sentiment and role == "user":
            conversations[session_id]["sentiments"].append(sentiment)

    @staticmethod
    def get_conversation(session_id: str) -> Dict:
        """Retrieve full conversation data."""
        return conversations.get(session_id, {})


# ==================== API ENDPOINTS ====================

@app.route("/")
def index():
    """Serve the web interface."""
    return send_file('static/index.html')


@app.route("/chat", methods=["POST"])
def chat():
    """Send a message and receive AI response with sentiment analysis."""
    try:
        body = request.get_json(force=True)
        message = body.get("message")
        session_id = body.get("session_id", "default")

        if not message:
            return jsonify({"ok": False, "error": "message required"}), 400

        # Analyze user message sentiment with enhanced method
        sentiment = SentimentAnalyzer.analyze_statement(message)

        # Store user message with sentiment
        ConversationManager.add_message(session_id, "user", message, sentiment)

        # Get conversation history for context
        conversation = ConversationManager.get_conversation(session_id)
        messages = [{"role": "system", "content": "You are a helpful, friendly assistant."}]

        # Add conversation history (limit to last 10 messages for context)
        for msg in conversation["messages"][-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Get AI response
        resp = client.chat.completions.create(
            model=CURRENT_MODEL["name"],
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        out_text = _extract_text_from_resp(resp)

        # Store assistant response
        ConversationManager.add_message(session_id, "assistant", out_text)

        return jsonify({
            "ok": True,
            "reply": out_text,
            "model": CURRENT_MODEL["name"],
            "sentiment": sentiment,
            "session_id": session_id
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/sentiment/conversation", methods=["GET"])
def get_conversation_sentiment():
    """Get full conversation sentiment analysis."""
    try:
        session_id = request.args.get("session_id", "default")
        conversation = ConversationManager.get_conversation(session_id)

        if not conversation:
            return jsonify({"ok": False, "error": "Session not found"}), 404

        sentiments = conversation.get("sentiments", [])

        if not sentiments:
            return jsonify({
                "ok": True,
                "message": "No user messages to analyze yet"
            })

        # Generate comprehensive analysis
        analysis = SentimentAnalyzer.analyze_conversation(sentiments)

        return jsonify({
            "ok": True,
            "session_id": session_id,
            "conversation_analysis": analysis,
            "message_sentiments": sentiments
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/conversation/history", methods=["GET"])
def get_conversation_history():
    """Get full conversation history with all sentiments."""
    try:
        session_id = request.args.get("session_id", "default")
        conversation = ConversationManager.get_conversation(session_id)

        if not conversation:
            return jsonify({"ok": False, "error": "Session not found"}), 404

        return jsonify({
            "ok": True,
            "session_id": session_id,
            "messages": conversation.get("messages", []),
            "sentiments": conversation.get("sentiments", []),
            "created_at": conversation.get("created_at")
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/session/new", methods=["POST"])
def new_session():
    """Create a new conversation session."""
    try:
        data = request.get_json(force=True)
        session_id = data.get("session_id", f"session_{datetime.now().timestamp()}")

        ConversationManager.create_session(session_id)

        return jsonify({
            "ok": True,
            "session_id": session_id,
            "message": "New session created"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


def _extract_text_from_resp(resp):
    """Extract text from Groq API response."""
    try:
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                msg = choice.message
                if hasattr(msg, "content"):
                    return msg.content
        return str(resp)
    except Exception:
        return str(resp)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("=" * 60)
    print("🤖 Sentiment Analysis Chatbot")
    print("=" * 60)
    print(f"Server: http://localhost:{port}")
    print(f"Model: {CURRENT_MODEL['name']}")
    print(f"Enhanced Sentiment Analysis: Active")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=True)
