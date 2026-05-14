import json
import logging
import os
import re
from dataclasses import dataclass, replace
from openai import OpenAI
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT, DESTINATION_DISCOVERY_PROMPT
from apis.exchange import ExchangeRateClient
from apis.unsplash import UnsplashClient

load_dotenv()
logger = logging.getLogger(__name__)

# Ordered from most specific to least — first match wins.
_DESTINATION_PATTERNS = [
    r"(?:photos?|pictures?|images?)\s+of\s+([A-Za-z][A-Za-z\s]+?)(?:\s*[?!.,]|$)",
    r"what\s+does\s+([A-Za-z][A-Za-z\s]+?)\s+look",
    r"show\s+me\s+([A-Za-z][A-Za-z\s]+?)(?:\s+photos?|\s+pictures?|[?!.,]|$)",
    r"(?:going\s+to|want\s+to\s+go\s+to|trip\s+to|travel(?:l?ing)?\s+to|visit(?:ing)?|fly(?:ing)?\s+to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
]


UNCLEAR_INPUT_MESSAGE = (
    "Hmm, I didn't quite catch that! Could you rephrase? "
    "I'm here to help plan your perfect trip ✈️"
)

GREETING_RESPONSE = (
    "Hey, great to meet you! I'm Alex — your personal travel assistant. "
    "I can help you plan trips, discover destinations, and pack smart. "
    "Where are you dreaming of going? ✈️"
)

OFF_TOPIC_RESPONSE = (
    "That's a bit outside my travel expertise! "
    "I'm best at helping you plan trips, discover destinations, and pack smart. "
    "What travel adventure can I help with?"
)

FAREWELL_RESPONSE = (
    "Happy travels! 🌍 I hope your trip is everything you dreamed of. "
    "Come back anytime you need travel help! ✈️"
)

# Detects explicit currency amounts in a message (e.g. $1000, 500 euros, 2000 JPY).
# Used to guarantee needs_exchange_rate=True even when the LLM misses it.
_CURRENCY_AMOUNT_RE = re.compile(
    r'(?:\$|€|£|¥|₹|₩|฿)\s*[\d,]+'
    r'|[\d,]+\s*(?:dollars?|euros?|pounds?|yen|rupees?|baht|won|peso|'
    r'ringgit|franc|lira|shekel|'
    r'USD|EUR|GBP|JPY|INR|THB|KRW|MXN|SGD|AUD|CAD|NZD|ILS|TRY|ZAR)',
    re.IGNORECASE,
)


# Budget signal words used by the router's safe-fallback path.
_BUDGET_SIGNAL_WORDS = frozenset({
    "budget", "cost", "price", "afford", "how much",
    "expensive", "cheap", "spend", "money", "currency",
})

# ── Router ─────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """
You are a router for a travel assistant. Analyze the user message and conversation history, then return a JSON object with these fields:

{
  "intent": "greeting" | "farewell" | "gibberish" | "off_topic" | "trip_planning" | "destination_discovery" | "packing_advice" | "photo_request" | "budget_question",
  "needs_exchange_rate": true | false,
  "needs_photo": true | false,
  "destination": "city/country name or null",
  "response_tone": "warm" | "practical" | "inspirational",
  "reasoning": "one sentence explaining the decision"
}

Rules:
- needs_exchange_rate: true when user mentions budget, cost, or money — especially when \
  they name a specific amount ($1000, 500 euros, 2000 USD) or pair a country with a budget
- needs_photo: true when user mentions a destination or asks to see photos
- destination: extract the place name if mentioned
- response_tone: warm for emotional messages, practical for planning, inspirational for discovery
- When in doubt about intent → use 'destination_discovery'
- Never use 'off_topic' unless completely unrelated to travel (sports, politics, cooking, math)

Any message expressing a negative emotion, tiredness, stress, or desire to escape/relax/travel — classify as destination_discovery with warm tone. Trust the meaning, not exact words.

greeting is ONLY for pure hellos with zero travel content:
- "hi", "hello", "hey", "hey there", "hi there" → greeting
- "hi, I want to go somewhere warm" → destination_discovery
- "hey, can you help me plan a trip?" → trip_planning

farewell is for conversation endings — never classify these as off_topic or gibberish:
- Goodbyes: "bye", "goodbye", "see you", "take care", "ciao"
- Thanks: "thank you", "thanks", "thanks!", "thank you so much"
- Satisfaction + thanks (commas and exclamation marks are common): "great, thanks",
  "great, thanks, bye", "Great, Thanks!", "Thanks! Bye!", "perfect, thanks",
  "perfect thanks", "great thanks", "great!"
- Completion signals: "that's all", "all good", "I'm done", "I'm set"
- Polite decline after recommendations: "no thanks", "I'm fine", "no, I'm good"
- Standalone "no" after Alex has given recommendations — they're declining further help

Return JSON only, no explanation.
"""

VALID_INTENTS = frozenset({
    "greeting", "farewell", "gibberish", "off_topic", "trip_planning",
    "destination_discovery", "packing_advice", "photo_request", "budget_question",
})
VALID_TONES = frozenset({"warm", "practical", "inspirational"})

# Tone hints injected into message context for non-default tones.
_TONE_CONTEXT = {
    "warm":          "[Tone: respond with warmth and empathy]",
    "inspirational": "[Tone: respond with inspiration and vivid imagery]",
}


@dataclass
class RouterDecision:
    intent: str
    needs_exchange_rate: bool
    needs_photo: bool
    destination: str | None
    response_tone: str
    reasoning: str

    @property
    def is_terminal(self) -> bool:
        """True when the intent maps to an immediate fixed response (no LLM call)."""
        return self.intent in {"greeting", "farewell", "gibberish", "off_topic"}


class Router:
    """
    Single structured LLM call that classifies a message and decides which
    external APIs to call.

    Cheap code-level pre-checks run first; the LLM is only called when
    they don't match.
    """

    def __init__(self, client, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def route(self, message: str, history: list[dict]) -> RouterDecision:
        try:
            stripped = message.strip()

            # Gibberish pre-check only — too short (≤ 2 chars) or all-consonant single word.
            # Threshold is 3 so that short but meaningful words like "bye" reach the LLM.
            if len(stripped) < 3:
                return self._make("gibberish", reason="Too short to be meaningful")
            words = stripped.split()
            if len(words) == 1:
                w = words[0].lower().strip(".,!?")
                if not any(c in "aeiou" for c in w):
                    return self._make("gibberish", reason="Single word with no vowels")

            # Packing trigger — any message containing "pack" is always packing_advice.
            # Catches "what should I pack?", "what to pack", "packing list", etc.
            if "pack" in stripped.lower():
                return self._make("packing_advice", tone="practical",
                                  reason="Packing keyword detected")

            decision = self._llm_route(message, history)

            # Guarantee exchange rates for budget questions or explicit currency amounts,
            # regardless of what the LLM returned for needs_exchange_rate.
            if decision.intent == "budget_question" or bool(_CURRENCY_AMOUNT_RE.search(message)):
                decision = replace(decision, needs_exchange_rate=True)

            return decision

        except Exception as exc:
            logger.debug("Router.route failed for %r: %s", message, exc)
            lowered = message.lower()
            needs_er = bool(_CURRENCY_AMOUNT_RE.search(message)) or any(
                kw in lowered for kw in _BUDGET_SIGNAL_WORDS
            )
            return self._make(
                "destination_discovery",
                needs_exchange_rate=needs_er,
                reason="Router error — safe fallback",
            )

    # ── Private helpers ────────────────────────────────────────────────────

    def _make(self, intent: str, *, tone: str = "practical", reason: str = "",
              needs_exchange_rate: bool = False, needs_photo: bool = False,
              destination: str | None = None) -> RouterDecision:
        return RouterDecision(
            intent=intent,
            needs_exchange_rate=needs_exchange_rate,
            needs_photo=needs_photo,
            destination=destination,
            response_tone=tone,
            reasoning=reason,
        )

    def _llm_route(self, message: str, history: list[dict]) -> RouterDecision:
        ctx_lines: list[str] = []
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Alex"
            content = re.split(r"\n\n\[", msg["content"])[0][:80]
            ctx_lines.append(f"  {role}: {content}")
        ctx = "\n".join(ctx_lines) or "None"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": f"Conversation history:\n{ctx}\n\nNew message: {message}"},
                ],
                max_tokens=150,
                temperature=0,
            )
            return self._parse(response.choices[0].message.content)
        except Exception as exc:
            logger.debug("Router LLM call failed: %s", exc)
            return self._make("destination_discovery",
                              reason="Router call failed — safe fallback")

    def _parse(self, raw: str) -> RouterDecision:
        try:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            data = json.loads(text)

            intent = data.get("intent", "destination_discovery")
            if intent not in VALID_INTENTS:
                intent = "destination_discovery"

            tone = data.get("response_tone", "warm")
            if tone not in VALID_TONES:
                tone = "warm"

            destination = data.get("destination") or None
            if destination and not isinstance(destination, str):
                destination = None

            return RouterDecision(
                intent=intent,
                needs_exchange_rate=bool(data.get("needs_exchange_rate", False)),
                needs_photo=bool(data.get("needs_photo", False)),
                destination=destination,
                response_tone=tone,
                reasoning=str(data.get("reasoning", ""))[:200],
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Router JSON parse failed: %s — raw: %.100s", exc, raw)
            return self._make("destination_discovery",
                              reason="Failed to parse router JSON")


# ── UserPreferences ────────────────────────────────────────────────────────

# Regex patterns for extracting budget and duration from free-text messages.
_BUDGET_RE = re.compile(
    r'\$\s*[\d,]+(?:\s*k)?\b'
    r'|(?:about|around|roughly|approximately)?\s*\d[\d,]*(?:\s*k)?\s*'
    r'(?:dollars?|USD|bucks?)',
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r'\d+(?:\s*[-–]\s*\d+)?\s*(?:days?|nights?|weeks?|months?)',
    re.IGNORECASE,
)
_VIBE_KEYWORDS: dict[str, list[str]] = {
    "adventure":  ["adventure", "adventurous", "active", "hiking", "outdoor", "thrill"],
    "relaxation": ["relax", "relaxation", "beach", "peaceful", "calm", "rest", "unwind", "recharge"],
    "culture":    ["culture", "cultural", "history", "historical", "museum", "art", "architecture"],
    "romance":    ["romance", "romantic", "honeymoon", "couple", "couples", "anniversary"],
    "food":       ["food", "foodie", "culinary", "cuisine", "eating", "gastronomy"],
}


@dataclass
class UserPreferences:
    """Permanently retained travel preferences extracted from the conversation.

    These survive the sliding-window history cut — they are always injected
    into the system message so the LLM never forgets stated preferences.
    """
    budget: str | None = None
    vibe: str | None = None
    duration: str | None = None
    destination: str | None = None

    def to_context(self) -> str:
        """Return a one-line context block, or '' if no preferences are set yet."""
        fields = [
            (k, v) for k, v in [
                ("destination", self.destination),
                ("budget", self.budget),
                ("duration", self.duration),
                ("vibe", self.vibe),
            ] if v is not None
        ]
        if not fields:
            return ""
        parts = ", ".join(f"{k}={v}" for k, v in fields)
        return f"[User preferences: {parts}]"


# ── TravelAssistant ────────────────────────────────────────────────────────

class TravelAssistant:
    def __init__(self, model: str | None = None):
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "deepseek":
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
            self.model = model or "deepseek-chat"
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"

        self.exchange = ExchangeRateClient()
        self.unsplash = UnsplashClient()
        self.history: list[dict] = []
        self.preferences = UserPreferences()
        self.router = Router(self.client, model=self.model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        decision = self.router.route(user_message, self.history)
        logger.debug(
            "Router: intent=%s tone=%s er=%s photo=%s dest=%s — %s",
            decision.intent, decision.response_tone,
            decision.needs_exchange_rate, decision.needs_photo,
            decision.destination, decision.reasoning,
        )

        if decision.intent == "gibberish":
            return UNCLEAR_INPUT_MESSAGE
        if decision.intent == "greeting":
            return GREETING_RESPONSE
        if decision.intent == "farewell":
            return FAREWELL_RESPONSE
        if decision.intent == "off_topic":
            return OFF_TOPIC_RESPONSE

        # Auto-inject exchange rates when the user has already shared a budget
        # AND is now mentioning a destination — no need to ask explicitly.
        if (not decision.needs_exchange_rate
                and self.preferences.budget is not None
                and decision.destination is not None):
            decision = replace(decision, needs_exchange_rate=True)

        # Travel intent — build context and call main LLM.
        context, supported_currencies = self._build_context(user_message, decision)
        try:
            self._update_preferences(user_message, decision)
            self.history.append({"role": "user", "content": user_message + context})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(),
                max_tokens=300,
                temperature=0.7,
            )
            # Guard against None content — can happen on unexpected API responses.
            raw = response.choices[0].message.content or ""
            reply = self._sanitize_response(raw, supported_currencies)
        except Exception as exc:
            logger.error("chat() failed: %s: %s", type(exc).__name__, exc, exc_info=True)
            reply = self._handle_error(exc)

        # Store only the clean reply (no injected context)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.history = []
        self.preferences = UserPreferences()

    def get_photo(self, destination: str) -> str | None:
        try:
            return self.unsplash.get_photo_url(destination)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _messages(self) -> list[dict]:
        system_parts = [SYSTEM_PROMPT, DESTINATION_DISCOVERY_PROMPT]

        # Permanently retained preferences — always present regardless of history length.
        prefs = self.preferences.to_context()
        if prefs:
            system_parts.append(prefs)

        # Rolling summary of recent conversation.
        summary = self._build_conversation_summary()
        if summary:
            system_parts.append(summary)

        # Destination continuity hint.
        last_dest = self._last_destination_from_history()
        if last_dest:
            system_parts.append(
                f"[Current conversation context: user was previously discussing {last_dest}.]"
            )

        # Sliding window — last 10 messages cap to limit token usage.
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + self.history[-10:]

    def _build_context(self, message: str, decision: RouterDecision) -> tuple[str, set[str]]:
        """Return (context_string, supported_currency_codes).

        The second element is passed to _sanitize_response() so post-processing
        knows which currencies were actually verified by the API.
        """
        parts = []
        supported_currencies: set[str] = set()

        if decision.needs_exchange_rate:
            try:
                dest = decision.destination or self._extract_destination(message)
                dest_currency = self.exchange.currency_for(dest)
                supported = self.exchange.get_supported_rates_only(
                    extra_currency=dest_currency
                )
                if supported:
                    supported_currencies = set(supported.keys())
                    lines = [
                        "[VERIFIED LIVE RATES FROM API - ONLY USE THESE RATES, NO OTHERS:"
                    ]
                    for currency, rate in sorted(supported.items()):
                        lines.append(f"1 USD = {rate:.2f} {currency}")
                    lines.append("ANY CURRENCY NOT LISTED ABOVE IS NOT SUPPORTED.")
                    lines.append(
                        "FOR UNSUPPORTED CURRENCIES: say 'I only have verified rates "
                        "for major currencies. Please check XE.com for this currency'"
                        " - NEVER INVENT A RATE]"
                    )
                    parts.append("\n".join(lines))
            except Exception:
                pass

        if decision.needs_photo and self.unsplash.is_configured:
            destination = decision.destination or self._extract_destination(message) or message
            try:
                url = self.unsplash.get_photo_url(destination)
                if url:
                    parts.append(f"[Photo of {destination}: {url}]")
            except Exception:
                pass

        # Inject tone hint for non-default tones (warm/inspirational).
        tone_note = _TONE_CONTEXT.get(decision.response_tone, "")
        if tone_note:
            parts.append(tone_note)

        context_str = ("\n\n" + "\n\n".join(parts)) if parts else ""
        return context_str, supported_currencies

    def _sanitize_response(self, reply: str, supported_currencies: set[str]) -> str:
        """Remove hallucinated currency conversions from the LLM response.

        Scans for currency codes the LLM used that are not in supported_currencies.
        Any line containing such a code alongside digits (i.e. a conversion) is
        stripped. If anything is removed, an XE.com note is appended.

        Only runs when supported_currencies is non-empty — if no rates were injected
        there is nothing to validate against, so the response is returned unchanged.
        """
        if not supported_currencies:
            return reply

        # USD is the base currency; the LLM may always express amounts in USD.
        always_ok = supported_currencies | {"USD"}

        # Identify currency codes in the response that we did NOT verify.
        known = self.exchange.all_known_currencies
        unsupported_used: set[str] = set()
        for m in re.finditer(r'\b([A-Z]{3})\b', reply):
            code = m.group(1)
            if code in known and code not in always_ok:
                unsupported_used.add(code)

        if not unsupported_used:
            return reply

        # Remove every line that contains any unsupported currency code.
        clean_lines: list[str] = []
        removed = False
        for line in reply.split("\n"):
            if any(code in line for code in unsupported_used):
                removed = True
            else:
                clean_lines.append(line)

        if not removed:
            return reply

        cleaned = "\n".join(clean_lines).strip()
        cleaned += (
            "\n\nNote: Live exchange rate for this destination is not available. "
            "Budget shown in USD. Check XE.com for local currency rates."
        )
        return cleaned

    def _update_preferences(self, message: str, decision: RouterDecision) -> None:
        """Extract and persist travel preferences from the current user message."""
        # Destination — Router's extraction is the most reliable source.
        if decision.destination:
            self.preferences.destination = decision.destination

        # Budget — regex scan for dollar amounts and word forms.
        m = _BUDGET_RE.search(message)
        if m:
            self.preferences.budget = m.group(0).strip()

        # Duration — regex scan for day/night/week counts.
        m = _DURATION_RE.search(message)
        if m:
            self.preferences.duration = m.group(0).strip()

        # Vibe — keyword scan; first match wins, existing value preserved if no match.
        lowered = message.lower()
        for vibe, keywords in _VIBE_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                self.preferences.vibe = vibe
                break

    def _build_conversation_summary(self) -> str:
        if not self.history:
            return ""
        recent = self.history[-6:]  # last 3 exchanges
        lines = ["[Prior conversation — reference when relevant:"]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Alex"
            # Strip injected context blocks (exchange rates, photo URLs) before summarising.
            content = re.split(r"\n\n\[", msg["content"])[0].strip()
            if len(content) > 100:
                content = content[:100] + "…"
            lines.append(f"  {role}: {content}")
        lines.append("]")
        return "\n".join(lines)

    def _extract_destination(self, message: str) -> str | None:
        for pattern in _DESTINATION_PATTERNS:
            m = re.search(pattern, message, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".,!? ")
        return None

    def _last_destination_from_history(self) -> str | None:
        if not self.history:
            return None
        # First pass: use existing extraction patterns on user messages (most precise).
        for msg in reversed(self.history[-10:]):
            if msg["role"] == "user":
                content = re.split(r"\n\n\[", msg["content"])[0]
                dest = self._extract_destination(content)
                if dest:
                    return dest
        # Second pass: find capitalised proper-noun sequences across all recent messages.
        skip = {
            "The", "For", "And", "But", "Its", "Our", "You", "Yes", "No", "Alex",
            "What", "How", "Where", "When", "Who", "Why", "Here", "There", "That",
            "This", "These", "Those", "Your", "My", "We", "Let", "Sure", "Also",
            "Just", "Too", "Now", "Then", "Day", "Days", "Week", "Month", "Year",
            "Night", "Tell", "Trip",
        }
        for msg in reversed(self.history[-10:]):
            content = re.split(r"\n\n\[", msg["content"])[0]
            matches = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", content)
            for match in reversed(matches):
                if all(w not in skip for w in match.split()) and len(match) > 3:
                    return match
        return None

    def _handle_error(self, error: Exception) -> str:
        msg = str(error).lower()
        if "api_key" in msg or "authentication" in msg or "unauthorized" in msg:
            return "I'm having trouble connecting — looks like an API key issue. Check your .env file."
        if "rate_limit" in msg or "quota" in msg:
            return "I'm getting a lot of requests right now. Give me a moment and try again!"
        if "timeout" in msg or "timed out" in msg:
            return "That took too long to load — try again in a second."
        return "Something went wrong on my end. Try asking again!"
