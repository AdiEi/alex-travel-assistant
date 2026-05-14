# Alex — Prompt Engineering Reference

This document explains the design decisions behind the prompts in [`prompts.py`](prompts.py).
It is the authoritative reference for anyone modifying Alex's behaviour.

---

## Why the 150-word response cap?

Travel planning can generate very long responses. Without a cap, models tend toward
exhaustive lists — every neighbourhood in Tokyo, every packing consideration for
South-East Asia — which reads like a Wikipedia article, not a conversation.

The 150-word limit does three things:

1. **Forces selectivity.** The model must choose the most relevant thing to say rather
   than listing everything it knows. This produces better answers, not shorter ones.
2. **Keeps pacing conversational.** A response the user can read in 30 seconds invites
   a reply. A wall of text stops the conversation.
3. **Leaves room for follow-up.** Users can always ask for more detail. The cap creates
   a natural dialogue loop rather than front-loading everything at once.

If a topic genuinely needs more depth (e.g. a full itinerary), the user asks and Alex
can build it across several turns.

---

## Why one question at a time?

Stacking questions — "What's your budget? When can you travel? Do you prefer cities or
nature? Are you going solo?" — feels like an intake form, not a conversation. It also
puts all the cognitive load on the user at once.

Limiting to **one clarifying question per turn** has two effects:

1. **Forces prioritisation.** The model must decide which piece of missing information
   matters most right now. That prioritisation is itself useful — it reflects what a
   good travel advisor would actually ask first.
2. **Produces better questions.** When the model can only ask one thing, it tends to ask
   a more open, generative question ("What kind of trip are you after?") rather than a
   narrow binary ("City or beach?"). Open questions surface preferences the user hadn't
   thought to mention.

This constraint lives in `SYSTEM_PROMPT` (not in code) so it can be tuned without a
deploy.

---

## How the chain-of-thought prompt works

`DESTINATION_DISCOVERY_PROMPT` is labelled **CHAIN OF THOUGHT PROMPT** in the code.
It is a reasoning scaffold, not a response template.

The prompt instructs the model to reason through five dimensions internally before
responding:

1. Mood / vibe (adventure, relaxation, culture, food, nature)
2. Travel style (budget backpacker, mid-range, luxury)
3. Practical constraints (time, budget, flight range)
4. Season and weather
5. Past trips loved or hated

The key phrase is **"reason through this internally"**. The model works through the
framework but does not output the reasoning — it outputs one focused question, or (after
3–4 turns) a synthesised set of 2–3 concrete recommendations.

**Why this matters:** Without the scaffold, models skip straight to generic suggestions
("Have you considered Paris?") before knowing anything meaningful about the person. The
chain-of-thought forces the model to identify what it still doesn't know before speaking.
This is the same technique used in step-by-step reasoning prompts, applied to a
conversational domain where the output must remain short and natural.

The mood-detection section runs first, before the discovery questions. If the user
sounds stressed or emotional, Alex acknowledges the feeling before asking any practical
question. This is also chain-of-thought: read → classify → respond to emotion → then
ask logistics.

---

## How external API data is blended into context

Alex has access to two live data sources: exchange rates (open.er-api.com) and
destination photos (Unsplash). Neither is injected on every message — that would add
noise and waste tokens. Instead, the `Router` decides per-message whether to fetch data
at all.

The Router returns a `RouterDecision` with `needs_exchange_rate` and `needs_photo`
boolean flags. When a flag is true, `_build_context()` in `assistant.py`:

1. Calls the relevant API
2. Formats the result as a bracketed annotation:
   - `[Live exchange rates: 1 USD = 149.50 JPY, 0.92 EUR, ...]`
   - `[Photo of Kyoto: https://images.unsplash.com/...]`
3. Appends the annotation to the user's message **before** it reaches the main LLM

The main LLM therefore sees the live data as part of the conversation context — the same
channel it uses for history, system instructions, and everything else. `SYSTEM_PROMPT`
tells Alex how to use each annotation when it appears.

**Why context injection rather than tool calls?**

OpenAI function calling / tool use would require the model to emit a structured tool
request, wait for a result, and resume. Context injection is simpler: the decision
about *whether* to fetch data is made by the `Router` (a separate fast LLM call), the
data is fetched in Python, and the result is ready before the main prompt is sent.
There is no mid-generation pause and the prompts stay clean — the main LLM never needs
to know how the data arrived.

The `Router`'s decision logic is documented in `assistant.py` and summarised in `README.md`.

---

## The Router Decision Tree

Every message passes through this decision flow before reaching the main LLM:

```
Message arrives
↓
Step 1: FREE PRE-CHECKS (no API call)
├── < 3 characters? → gibberish
├── single word, no vowels? → gibberish
└── contains 'pack'? → packing_advice
↓
Step 2: LLM ROUTER CALL (one structured GPT call)
→ returns JSON: intent + needs_exchange_rate + needs_photo + tone + destination
↓
Step 3: OVERRIDE CHECKS
├── intent is 'budget_question'? → force needs_exchange_rate=True
└── message contains '$2000' or '€500'? → force needs_exchange_rate=True
↓
Step 4: BUILD CONTEXT
├── needs_exchange_rate=True → call Exchange Rates API → inject rates
└── needs_photo=True → call Unsplash API → inject photo URL
↓
Step 5: MAIN LLM CALL
→ system prompt + preferences + last 10 history + live context
↓
Step 6: POST-PROCESSING
└── _sanitize_response() removes hallucinated currency conversions
↓
Step 7: ERROR FALLBACK (if anything fails)
└── defaults to destination_discovery, never crashes
```

This replaces three separate functions (`classify_input`, `should_use_external_data`,
`_needs_exchange_rates`) with one structured call — reducing latency and keeping
decision logic in one place.

---

## Memory Management Flow

```
User sends message
↓
Router classifies intent
↓
Terminal intent? (greeting/gibberish/farewell/off_topic)
├── YES → respond immediately, nothing stored in history
└── NO → continue
↓
Store in history:
├── UserPreferences updated (budget/vibe/duration/destination)
└── Full message + context appended to history list
↓
Build LLM messages:
├── SYSTEM_PROMPT + DESTINATION_DISCOVERY_PROMPT
├── UserPreferences (permanent, never deleted)
├── _build_conversation_summary() → last 6 messages
├── _last_destination_from_history() → last 10 messages scanned
└── last 10 history messages (sliding window)
↓
User clicks "New Journey":
└── history = [] (full wipe)
    UserPreferences reset to None
```

---

## Tone Injection

The Router returns a `response_tone` field (`warm`, `practical`, or `inspirational`) as
part of its structured JSON output. For non-default tones, `_build_context()` appends an
annotation to the message context before the main LLM call:

- `warm` → `[Tone: respond with warmth and empathy]`
- `inspirational` → `[Tone: respond with inspiration and vivid imagery]`
- `practical` → no annotation (default behaviour)

The main LLM adapts its language style based on this instruction without needing separate
system prompts for each tone. Emotional messages (stress, exhaustion, desire to escape)
automatically receive the warm tone; destination discovery gets inspirational; planning
and budgeting get practical.

---

## Structured Router Output

`ROUTER_PROMPT` forces the LLM to return valid JSON with eight specific fields: `intent`,
`needs_exchange_rate`, `needs_photo`, `destination`, `response_tone`, `reasoning`, and
the full intent enum. This turns what would be three separate classification and
API-gating calls into a single structured decision:

- Intent classification (replacing the old `classify_input` function)
- External API flags (replacing `should_use_external_data`)
- Destination extraction and tone selection

If JSON parsing fails, the system falls back gracefully to `destination_discovery` so
the conversation continues without errors. Code-level pre-checks (gibberish detection,
packing keyword) short-circuit the LLM call entirely for the simplest cases.

---

## UserPreferences Injection

`budget`, `vibe`, `duration`, and `destination` are extracted from every user message
via regex and keyword matching, then stored in a `UserPreferences` dataclass on the
`TravelAssistant` instance.

On every LLM call, `to_context()` serialises the non-null fields into a single line
injected directly into the system message:

```
[User preferences: destination=Japan, budget=$2500, duration=10 days, vibe=relaxation]
```

This block is always present regardless of how many messages have been trimmed by the
10-message sliding window. The result: Alex never forgets a stated budget, destination,
or travel vibe even in long conversations. `reset()` clears preferences alongside
history, so a "New Journey" is a complete fresh start.

---

## Hallucination Prevention

When exchange rate data is unavailable from our API, Alex is explicitly instructed NOT
to guess rates from training data — LLM training data for exotic currencies can be years
out of date or simply wrong. Alex instead directs users to a reliable source (XE.com)
and continues helping in USD terms.

**How it works technically:** `format_for_context()` in `apis/exchange.py` checks
whether the destination currency was actually returned by the live API. If it was not,
it appends a `[NO VERIFIED LIVE RATE for XXX]` annotation to the context block instead
of silently omitting the currency. `SYSTEM_PROMPT` contains an explicit rule that fires
on this annotation — giving Alex the exact fallback sentence to say and ending with a
hard prohibition on estimating rates from training knowledge.

The annotation travels through the same context-injection channel as verified rates, so
the model sees the gap as an explicit instruction rather than an absence it might fill
with its own (potentially stale) knowledge.

---

## Error Handling & Recovery

Alex handles errors at multiple layers — no single failure can crash the system.

### 1. Hallucination Prevention (Post-processing)

`_sanitize_response()` scans every LLM response for currency codes. If a currency is
not in the verified API rates set (`exchange.all_known_currencies`), the conversion line
is removed and replaced with: *"Check XE.com for this currency."* This catches
hallucinations **after** generation — defence in depth.

### 2. Unsupported Currency (Pre-processing)

Before the main LLM call, if the destination currency is absent from the API response,
`[Exchange rate not available for this destination. Provide budget analysis in USD only.]`
is injected into context. `SYSTEM_PROMPT` explicitly instructs Alex never to estimate —
preventing hallucination before it happens.

### 3. Router Failure Fallback

If the Router crashes for any reason, it defaults to `destination_discovery` intent and
checks the message for budget signals to set `needs_exchange_rate` appropriately. The
user never sees a technical error — the conversation continues naturally.

### 4. API Failures

| Failure | Behaviour |
|---------|-----------|
| Exchange rate API down | Alex responds without rates; no crash |
| Unsplash API down | Photos silently skipped; full functionality maintained |
| OpenAI timeout or rate limit | Warm error message; user asked to retry |
| LLM returns `None` content | Guarded with `content or ""` — never propagates as `None` |

### 5. Input Recovery

- **Gibberish** — warm clarification: *"Hmm, I didn't quite catch that!"* — caught before any API call
- **Off-topic** — gentle redirect back to travel planning
- **Short unclear input** — pre-filter (< 3 chars, no vowels) catches before Router LLM call

### Design Principle

Every error path returns something useful to the user. No failure is surfaced as a
technical message. The system fails gracefully at every layer.

### Context Persistence Through Errors

When a connection error or API failure occurs mid-conversation, `UserPreferences`
(destination, budget, vibe, duration) are stored in Python memory — not in the API
call chain. This means:

- Connection errors never wipe conversation context
- Alex remembers destination, budget, and preferences after recovery
- Only clicking "New Journey" resets preferences

This was validated in testing: after multiple `APIConnectionError` failures, Alex
correctly remembered the user's destination (Israel) and continued the conversation
naturally.
