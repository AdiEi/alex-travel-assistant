# Alex ✈️ — Your Personal Travel Companion

A conversational travel assistant powered by GPT-4o-mini. Alex helps with trip planning, destination discovery, and packing advice. It has a web UI (Flask) and a CLI (Rich terminal). Live exchange rates and destination photos are injected into the conversation automatically when relevant.

---

## What Makes Alex Different

Most AI travel assistants treat trip planning as a search problem. Before building Alex we researched the leading tools — ChatGPT's travel mode, MindTrip, Layla, and iMean — and found the same gaps across all of them.

**Gaps we found:**

| Problem | How it showed up |
|---------|-----------------|
| Generic recommendations | "You might enjoy Paris" — before knowing anything about the person |
| No emotional intelligence | Budget questions fired the moment someone said "I need to escape" |
| Outdated exchange rates | Currency figures that could be weeks or months stale |
| Overwhelming UX | Multiple questions per turn, walls of text, no visual inspiration |
| Weak context memory | Re-asking the same question, forgetting destinations mid-conversation |

**What Alex does differently:**

- **Mood detection** — Alex reads emotional tone before asking practical questions. "I'm exhausted and need to escape" triggers peaceful destination suggestions, not "What's your budget?" The `DESTINATION_DISCOVERY_PROMPT` has an explicit mood layer that runs first.
- **Real-time exchange rates** — live data from open.er-api.com is injected into the conversation only when cost questions arise, so figures are always current, never cached.
- **One question at a time** — never an interrogation. Each turn moves the conversation forward with a single focused question, which produces more natural dialogue and better answers.
- **Visual inspiration** — real destination photos from Unsplash appear inline when a place is mentioned, making recommendations tangible rather than abstract.
- **Context memory** — Alex tracks the last destination discussed and references it when the user seems uncertain, so the conversation stays continuous rather than resetting on every message.

---

## Running for Free

Alex supports DeepSeek as a completely free alternative to OpenAI.

1. Get a free API key at [platform.deepseek.com](https://platform.deepseek.com)
2. In your `.env` set:
   ```
   LLM_PROVIDER=deepseek
   DEEPSEEK_API_KEY=your_key_here
   ```
3. Everything works identically — same conversation quality, same Router logic,
   same exchange rates, same photos, same memory

DeepSeek uses the OpenAI-compatible API, so no code changes are needed when switching
providers.

---

## Setup

1. Clone the repo
   ```bash
   git clone <repo-url>
   cd travel_assistant
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your OpenAI key
   ```bash
   cp .env.example .env
   # then open .env and fill in OPENAI_API_KEY
   ```

4. Start the web server
   ```bash
   python app.py
   ```

5. Open [http://localhost:5000](http://localhost:5000)

To use the CLI instead: `python main.py`

---

## API Keys

| Key | Required | Purpose |
|-----|----------|---------|
| `OPENAI_API_KEY` | Yes *(or DeepSeek)* | Powers all chat via GPT-4o-mini |
| `DEEPSEEK_API_KEY` | Alternative to OpenAI | Free alternative LLM — set `LLM_PROVIDER=deepseek` |
| `UNSPLASH_ACCESS_KEY` | No | Destination photos — app works fully without it, photos simply won't appear |

All keys go in `.env` (see `.env.example` for the format).

---

## Prompt Engineering Decisions

### Chain of thought for destination discovery

The `DESTINATION_DISCOVERY_PROMPT` tells the model to reason through five dimensions internally before responding — mood/vibe, travel style, practical constraints, season, and past trips the user has loved or hated. The key word is *internally*: the output is one focused question, not the reasoning itself.

Without this scaffold, the model tends to jump straight to generic recommendations ("Have you considered Paris?") before it knows anything meaningful about the person. The chain-of-thought forces it to figure out what it still doesn't know before speaking.

### One question per turn

Stacking questions ("What's your budget? When can you travel? Do you prefer cities or nature?") feels like a form and kills the conversational feel. Limiting to one question per turn forces the model to prioritise — which question matters most right now — and that prioritisation almost always surfaces more relevant follow-ups than the user would have volunteered.

The constraint lives in the system prompt, not in code, so it can be adjusted without a deploy.

### 150-word response cap

Travel responses left unconstrained become encyclopedic. The cap makes Alex feel like a conversation partner rather than a guidebook, and forces selectivity: the model has to choose the most useful thing to say rather than listing everything it knows. Users who want more can always ask a follow-up.

### Detecting when to call external APIs

Every incoming message passes through the `Router` class in [`assistant.py`](assistant.py), which makes a single structured LLM call and returns a `RouterDecision` with explicit `needs_exchange_rate`, `needs_photo`, `intent`, `destination`, and `response_tone` fields. The `Router` replaced the earlier `should_use_external_data` / `DataSourcePlan` approach during refactoring.

**Exchange Rates API** — fires when the message contains budget or money signals: `cost`, `afford`, `how much`, `convert`, currency names (`dollar`, `euro`, `yen`, etc.). Live rates from open.er-api.com are injected into the message context so the LLM quotes current figures. Without live rates, cost estimates can be 10–40 % off — which destroys trust.

**Unsplash Photos API** — fires on explicit visual requests (`photo`, `show me`, `looks like`) or destination intro phrases (`trip to`, `visiting`, `going to` + a capitalised place). Only runs when `UNSPLASH_ACCESS_KEY` is configured; the app is fully functional without it.

**LLM only** — packing advice, general destination questions, itinerary planning, culture, climate. GPT-4o-mini's training data covers these well; injecting live data adds no value and wastes tokens.

Both APIs inject their result as context appended to the user message before it reaches the LLM, rather than using tool calls. This keeps the implementation simple and the prompts clean.

### Input classification before the main assistant

Every message passes through the `Router` — a structured LLM call that returns one of eight intents (`trip_planning`, `destination_discovery`, `packing_advice`, `photo_request`, `budget_question`, `greeting`, `farewell`, `gibberish`, `off_topic`) plus the `needs_exchange_rate`, `needs_photo`, and `response_tone` flags. Cheap code-level pre-checks (length, vowel test, packing keyword) short-circuit the LLM call entirely for obvious cases.

This keeps conversation history clean — terminal intents (greeting, farewell, gibberish, off_topic) never enter history or consume context — and means a single Router call replaces what used to be three separate classification and API-decision steps.

---

## Sample Conversations

The [`samples/`](samples/) folder contains six annotated transcripts:

- [`conversation1.txt`](samples/conversation1.txt) — greeting detection, gibberish handling, off-topic, farewell
- [`conversation2.txt`](samples/conversation2.txt) — live exchange rates, budget breakdown in local currency (ILS)
- [`conversation3.txt`](samples/conversation3.txt) — staged questioning, tailored packing list
- [`conversation4.txt`](samples/conversation4.txt) — complex honeymoon planning, destination comparison, 15+ messages
- [`conversation5.txt`](samples/conversation5.txt) — unsupported currency, hallucination prevention, graceful degradation
- [`conversation6.txt`](samples/conversation6.txt) — full journey with error recovery and context persistence through connection failures

## Tests

Run the test suite:

```bash
pytest tests/
```

Tests cover: conversation memory, intent classification, exchange rate injection, error handling, and hallucination prevention.

---

## Demo

See [`samples/demo.mp4`](samples/demo.mp4) for a full video walkthrough of Alex in action.
