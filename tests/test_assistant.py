import pytest
from unittest.mock import MagicMock, patch

from assistant import (
    TravelAssistant,
    Router,
    RouterDecision,
    UserPreferences,
    UNCLEAR_INPUT_MESSAGE,
    GREETING_RESPONSE,
    FAREWELL_RESPONSE,
    OFF_TOPIC_RESPONSE,
)
from apis.exchange import ExchangeRateClient


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    """Real Router with a mocked OpenAI client — for Router unit tests."""
    return Router(MagicMock())


@pytest.fixture
def assistant():
    """TravelAssistant with mocked router, exchange, and unsplash."""
    with patch("assistant.OpenAI"):
        a = TravelAssistant()
        a.exchange = MagicMock()
        # Give the mock a realistic currency set so _sanitize_response works correctly.
        a.exchange.all_known_currencies = frozenset({
            "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "ILS", "THB",
            "KPW", "IDR", "INR", "MXN", "BRL", "SGD", "NZD", "AED", "TRY",
        })
        a.unsplash = MagicMock()
        # Default: any message routed as a practical trip-planning query.
        a.router = MagicMock()
        a.router.route.return_value = RouterDecision(
            intent="trip_planning",
            needs_exchange_rate=False,
            needs_photo=False,
            destination=None,
            response_tone="practical",
            reasoning="default test decision",
        )
        return a


def _mock_reply(assistant: TravelAssistant, text: str) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = text
    assistant.client.chat.completions.create.return_value = mock_response


def _mock_router_llm(router_instance: Router, json_content: str) -> None:
    """Configure Router's client to return a specific JSON string."""
    resp = MagicMock()
    resp.choices[0].message.content = json_content
    router_instance.client.chat.completions.create.return_value = resp


def _route(assistant, intent: str = "trip_planning", **kwargs) -> RouterDecision:
    """Override the router mock to return a specific decision."""
    decision = RouterDecision(
        intent=intent,
        needs_exchange_rate=kwargs.get("needs_exchange_rate", False),
        needs_photo=kwargs.get("needs_photo", False),
        destination=kwargs.get("destination", None),
        response_tone=kwargs.get("response_tone", "practical"),
        reasoning="test",
    )
    assistant.router.route.return_value = decision
    return decision


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_history_starts_empty(self, assistant):
        assert assistant.history == []

    def test_user_and_assistant_messages_stored(self, assistant):
        _mock_reply(assistant, "Paris is lovely!")
        assistant.chat("Where should I go?")
        assert len(assistant.history) == 2
        assert assistant.history[0]["role"] == "user"
        assert assistant.history[1]["role"] == "assistant"
        assert assistant.history[1]["content"] == "Paris is lovely!"

    def test_history_accumulates_across_turns(self, assistant):
        _mock_reply(assistant, "Great choice!")
        assistant.chat("Recommend a destination")
        assistant.chat("Tell me more about Tokyo")
        assert len(assistant.history) == 4

    def test_reset_clears_history(self, assistant):
        _mock_reply(assistant, "Sure!")
        assistant.chat("Hello")
        assistant.reset()
        assert assistant.history == []

    def test_system_prompt_not_stored_in_history(self, assistant):
        _mock_reply(assistant, "Done!")
        assistant.chat("Test")
        roles = [m["role"] for m in assistant.history]
        assert "system" not in roles


# ---------------------------------------------------------------------------
# API failure fallback
# ---------------------------------------------------------------------------

class TestAPIFailureFallback:
    def test_exchange_api_failure_does_not_inject_junk(self, assistant):
        assistant.exchange.get_supported_rates_only.side_effect = Exception("API down")
        decision = RouterDecision(
            intent="budget_question", needs_exchange_rate=True, needs_photo=False,
            destination=None, response_tone="practical", reasoning="test",
        )
        context, supported = assistant._build_context("budget in euros", decision)
        assert "VERIFIED LIVE RATES" not in context
        assert context == ""
        assert supported == set()

    def test_chat_continues_when_exchange_fails(self, assistant):
        _route(assistant, "budget_question", needs_exchange_rate=True)
        assistant.exchange.get_supported_rates_only.side_effect = Exception("Network error")
        _mock_reply(assistant, "Great question about your budget!")
        reply = assistant.chat("What's my budget in euros?")
        assert reply == "Great question about your budget!"

    def test_unsplash_failure_returns_none(self, assistant):
        assistant.unsplash.get_photo_url.side_effect = Exception("API down")
        result = assistant.get_photo("Paris")
        assert result is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.parametrize("error_text,expected_fragment", [
        ("Invalid API key — authentication failed", "key"),
        ("rate_limit exceeded", "requests"),
        ("Request timed out", "long"),
        ("Unknown internal error", "went wrong"),
    ])
    def test_error_messages_are_user_friendly(self, assistant, error_text, expected_fragment):
        result = assistant._handle_error(Exception(error_text))
        assert expected_fragment in result.lower()

    def test_openai_error_returns_message_not_exception(self, assistant):
        assistant.client.chat.completions.create.side_effect = Exception("rate_limit")
        reply = assistant.chat("Hello")
        assert isinstance(reply, str)
        assert len(reply) > 0


# ---------------------------------------------------------------------------
# Router unit tests
# ---------------------------------------------------------------------------

class TestRouter:
    # ── Pre-checks (no LLM call) ──────────────────────────────────────────

    def test_short_inputs_are_gibberish_without_llm(self, router):
        # Threshold is < 3 chars; "bye" (3) is intentionally let through to the LLM.
        for msg in ("q", "x", "zz"):
            assert router.route(msg, []).intent == "gibberish"
        router.client.chat.completions.create.assert_not_called()

    def test_no_vowel_single_word_is_gibberish_without_llm(self, router):
        for msg in ("bvlvlf", "pfft", "hmm", "zxcv"):
            assert router.route(msg, []).intent == "gibberish"
        router.client.chat.completions.create.assert_not_called()

    @pytest.mark.parametrize("phrase", [
        "what should I pack?",
        "what to pack for Japan",
        "packing list for 2 weeks",
        "help me pack for Bali",
    ])
    def test_pack_keyword_routes_to_packing_advice_without_llm(self, router, phrase):
        d = router.route(phrase, [])
        assert d.intent == "packing_advice"
        router.client.chat.completions.create.assert_not_called()

    def test_non_gibberish_reaches_llm(self, router):
        """Any message that passes the gibberish check goes to the LLM."""
        _mock_router_llm(router, '{"intent":"destination_discovery","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":null,"response_tone":'
                                 '"warm","reasoning":"test"}')
        router.route("I need a break", [])
        router.client.chat.completions.create.assert_called_once()

    def test_router_prompt_includes_emotional_guidance(self, router):
        from assistant import ROUTER_PROMPT
        assert "negative emotion" in ROUTER_PROMPT
        assert "Trust the meaning" in ROUTER_PROMPT
        assert "destination_discovery" in ROUTER_PROMPT
        assert "greeting is ONLY" in ROUTER_PROMPT

    # ── JSON parsing ───────────────────────────────────────────────────────

    def test_valid_json_parsed_correctly(self, router):
        _mock_router_llm(router, '{"intent":"budget_question","needs_exchange_rate":true,'
                                 '"needs_photo":false,"destination":"Japan",'
                                 '"response_tone":"practical","reasoning":"Budget question"}')
        d = router.route("How much does Japan cost?", [])
        assert d.intent == "budget_question"
        assert d.needs_exchange_rate is True
        assert d.needs_photo is False
        assert d.destination == "Japan"
        assert d.response_tone == "practical"
        assert d.reasoning == "Budget question"

    def test_markdown_fenced_json_parsed(self, router):
        _mock_router_llm(router, '```json\n{"intent":"photo_request","needs_exchange_rate"'
                                 ':false,"needs_photo":true,"destination":"Bali",'
                                 '"response_tone":"inspirational","reasoning":"photo"}\n```')
        d = router.route("Show me Bali", [])
        assert d.intent == "photo_request"
        assert d.needs_photo is True
        assert d.destination == "Bali"

    def test_unknown_intent_falls_back_to_discovery(self, router):
        _mock_router_llm(router, '{"intent":"cooking","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":null,'
                                 '"response_tone":"practical","reasoning":"test"}')
        assert router.route("How do I cook pasta?", []).intent == "destination_discovery"

    def test_unknown_tone_falls_back_to_warm(self, router):
        _mock_router_llm(router, '{"intent":"trip_planning","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":null,'
                                 '"response_tone":"aggressive","reasoning":"test"}')
        assert router.route("Plan my trip", []).response_tone == "warm"

    def test_invalid_json_falls_back_to_discovery(self, router):
        _mock_router_llm(router, "not json at all")
        assert router.route("Plan my trip", []).intent == "destination_discovery"

    def test_llm_error_falls_back_to_discovery(self, router):
        router.client.chat.completions.create.side_effect = Exception("API error")
        assert router.route("Plan my trip", []).intent == "destination_discovery"

    # ── RouterDecision properties ──────────────────────────────────────────

    @pytest.mark.parametrize("intent", ["greeting", "farewell", "gibberish", "off_topic"])
    def test_terminal_intents_are_is_terminal(self, router, intent):
        d = RouterDecision(intent=intent, needs_exchange_rate=False, needs_photo=False,
                           destination=None, response_tone="warm", reasoning="")
        assert d.is_terminal is True

    @pytest.mark.parametrize("intent", ["trip_planning", "destination_discovery",
                                         "budget_question", "photo_request", "packing_advice"])
    def test_travel_intents_are_not_terminal(self, router, intent):
        d = RouterDecision(intent=intent, needs_exchange_rate=False, needs_photo=False,
                           destination=None, response_tone="practical", reasoning="")
        assert d.is_terminal is False

    def test_currency_amount_forces_exchange_rate_regardless_of_llm(self, router):
        # LLM returns needs_exchange_rate: false, but message has a dollar amount.
        _mock_router_llm(router, '{"intent":"trip_planning","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":"Japan",'
                                 '"response_tone":"practical","reasoning":"test"}')
        d = router.route("I have $2000 for my Japan trip", [])
        assert d.needs_exchange_rate is True

    def test_budget_question_intent_forces_exchange_rate(self, router):
        # LLM returns budget_question but forgets to set needs_exchange_rate.
        _mock_router_llm(router, '{"intent":"budget_question","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":null,'
                                 '"response_tone":"practical","reasoning":"test"}')
        d = router.route("Is Japan expensive to visit?", [])
        assert d.needs_exchange_rate is True

    def test_non_budget_message_does_not_force_exchange_rate(self, router):
        _mock_router_llm(router, '{"intent":"trip_planning","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":"Tokyo",'
                                 '"response_tone":"inspirational","reasoning":"test"}')
        d = router.route("What are the best things to do in Tokyo?", [])
        assert d.needs_exchange_rate is False

    def test_farewell_json_parsed_as_farewell(self, router):
        _mock_router_llm(router, '{"intent":"farewell","needs_exchange_rate":false,'
                                 '"needs_photo":false,"destination":null,'
                                 '"response_tone":"warm","reasoning":"User said goodbye"}')
        d = router.route("thanks, that's all!", [])
        assert d.intent == "farewell"

    def test_router_prompt_includes_farewell_guidance(self, router):
        from assistant import ROUTER_PROMPT
        assert "farewell" in ROUTER_PROMPT
        assert "goodbye" in ROUTER_PROMPT


# ---------------------------------------------------------------------------
# Chat routing
# ---------------------------------------------------------------------------

class TestChatRouting:
    def test_gibberish_returns_clarification(self, assistant):
        _route(assistant, "gibberish")
        assert assistant.chat("bvlvlf") == UNCLEAR_INPUT_MESSAGE

    def test_greeting_returns_welcome_response(self, assistant):
        _route(assistant, "greeting", response_tone="warm")
        assert assistant.chat("Hey!") == GREETING_RESPONSE

    def test_off_topic_returns_off_topic_message(self, assistant):
        _route(assistant, "off_topic")
        assert assistant.chat("Who won the game?") == OFF_TOPIC_RESPONSE

    def test_farewell_returns_farewell_response(self, assistant):
        _route(assistant, "farewell", response_tone="warm")
        assert assistant.chat("thanks, bye!") == FAREWELL_RESPONSE

    def test_terminal_intents_not_stored_in_history(self, assistant):
        for intent in ("gibberish", "greeting", "farewell", "off_topic"):
            _route(assistant, intent)
            assistant.chat("some message")
        assert len(assistant.history) == 0

    def test_terminal_intents_do_not_call_main_api(self, assistant):
        for intent in ("gibberish", "greeting", "farewell", "off_topic"):
            _route(assistant, intent)
            assistant.chat("some message")
        assistant.client.chat.completions.create.assert_not_called()

    def test_travel_intent_stored_in_history(self, assistant):
        _route(assistant, "trip_planning")
        _mock_reply(assistant, "Tokyo is amazing!")
        assistant.chat("I want to go to Tokyo")
        assert len(assistant.history) == 2

    def test_very_short_input_caught_as_gibberish_end_to_end(self):
        """End-to-end: real Router (not mock) classifies 'q' → gibberish."""
        with patch("assistant.OpenAI"):
            a = TravelAssistant()
            a.exchange = MagicMock()
            a.unsplash = MagicMock()
            # Real router uses the same mocked client
            assert a.chat("q") == UNCLEAR_INPUT_MESSAGE
            a.client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Destination context injection
# ---------------------------------------------------------------------------
# User preferences
# ---------------------------------------------------------------------------

class TestUserPreferences:
    def test_to_context_empty_returns_empty_string(self):
        assert UserPreferences().to_context() == ""

    def test_to_context_with_all_fields(self):
        p = UserPreferences(destination="Japan", budget="$2000",
                            duration="10 days", vibe="cultural")
        ctx = p.to_context()
        assert ctx == "[User preferences: destination=Japan, budget=$2000, duration=10 days, vibe=cultural]"

    def test_to_context_partial_fields(self):
        p = UserPreferences(destination="Italy", budget="$3000")
        ctx = p.to_context()
        assert "destination=Italy" in ctx
        assert "budget=$3000" in ctx
        assert "vibe" not in ctx

    def test_update_preferences_captures_destination_from_decision(self, assistant):
        decision = RouterDecision(intent="trip_planning", needs_exchange_rate=False,
                                  needs_photo=False, destination="Japan",
                                  response_tone="practical", reasoning="test")
        assistant._update_preferences("I want to visit Japan", decision)
        assert assistant.preferences.destination == "Japan"

    def test_update_preferences_extracts_budget(self, assistant):
        decision = RouterDecision(intent="budget_question", needs_exchange_rate=True,
                                  needs_photo=False, destination=None,
                                  response_tone="practical", reasoning="test")
        assistant._update_preferences("My budget is $2000 for the trip", decision)
        assert assistant.preferences.budget is not None
        assert "2000" in assistant.preferences.budget

    def test_update_preferences_extracts_duration(self, assistant):
        decision = RouterDecision(intent="trip_planning", needs_exchange_rate=False,
                                  needs_photo=False, destination=None,
                                  response_tone="practical", reasoning="test")
        assistant._update_preferences("I'm planning a 10 days trip", decision)
        assert assistant.preferences.duration is not None
        assert "10" in assistant.preferences.duration

    def test_update_preferences_extracts_vibe(self, assistant):
        decision = RouterDecision(intent="destination_discovery", needs_exchange_rate=False,
                                  needs_photo=False, destination=None,
                                  response_tone="warm", reasoning="test")
        assistant._update_preferences("I love hiking and outdoor adventure", decision)
        assert assistant.preferences.vibe == "adventure"

    def test_preferences_injected_into_system_prompt(self, assistant):
        assistant.preferences = UserPreferences(destination="Japan", budget="$2000",
                                                duration="10 days", vibe="cultural")
        system = assistant._messages()[0]["content"]
        assert "User preferences" in system
        assert "Japan" in system
        assert "$2000" in system

    def test_messages_uses_sliding_window_of_10(self, assistant):
        assistant.history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        messages = assistant._messages()
        # System message + last 10 history messages
        assert len(messages) == 11
        assert messages[-1]["content"] == "msg 19"
        assert messages[1]["content"] == "msg 10"

    def test_reset_clears_preferences(self, assistant):
        assistant.preferences = UserPreferences(destination="Japan", budget="$2000")
        _mock_reply(assistant, "ok")
        assistant.reset()
        assert assistant.preferences.destination is None
        assert assistant.preferences.budget is None


# ---------------------------------------------------------------------------

class TestDestinationContextInjection:
    def test_returns_none_for_empty_history(self, assistant):
        assert assistant._last_destination_from_history() is None

    def test_extracts_destination_from_user_message(self, assistant):
        assistant.history = [
            {"role": "user", "content": "I'm planning a trip to Japan"},
            {"role": "assistant", "content": "Japan is wonderful!"},
        ]
        assert assistant._last_destination_from_history() == "Japan"

    def test_extracts_multiword_destination(self, assistant):
        assistant.history = [
            {"role": "user", "content": "trip to New Zealand"},
            {"role": "assistant", "content": "New Zealand is stunning!"},
        ]
        assert assistant._last_destination_from_history() == "New Zealand"

    def test_destination_context_injected_into_system_prompt(self, assistant):
        assistant.history = [
            {"role": "user", "content": "planning a trip to Japan"},
            {"role": "assistant", "content": "Japan is wonderful!"},
        ]
        system = assistant._messages()[0]["content"]
        assert "previously discussing" in system
        assert "Japan" in system

    def test_no_destination_context_when_history_empty(self, assistant):
        system = assistant._messages()[0]["content"]
        assert "previously discussing" not in system


# ---------------------------------------------------------------------------
# Conversation context memory
# ---------------------------------------------------------------------------

class TestConversationContextMemory:
    def test_empty_history_produces_no_summary(self, assistant):
        assert assistant._build_conversation_summary() == ""

    def test_summary_includes_prior_message_content(self, assistant):
        assistant.history = [
            {"role": "user", "content": "I want to visit Japan for 10 days"},
            {"role": "assistant", "content": "Japan is a great choice!"},
        ]
        assert "Japan" in assistant._build_conversation_summary()

    def test_summary_injected_into_system_prompt(self, assistant):
        assistant.history = [
            {"role": "user", "content": "Planning a trip to Japan"},
            {"role": "assistant", "content": "Japan is wonderful for 10 days"},
        ]
        assert "Japan" in assistant._messages()[0]["content"]

    def test_injected_context_stripped_from_summary(self, assistant):
        assistant.history = [
            {"role": "user",
             "content": "Budget for Japan?\n\n[Live exchange rates: 1 USD = 149 JPY]"},
            {"role": "assistant", "content": "Japan on $3000 is doable!"},
        ]
        summary = assistant._build_conversation_summary()
        assert "exchange rates" not in summary.lower()
        assert "Japan" in summary

    def test_summary_capped_at_last_six_messages(self, assistant):
        assistant.history = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        summary = assistant._build_conversation_summary()
        assert "Message 9" in summary
        assert "Message 3" not in summary

    def test_long_messages_truncated_in_summary(self, assistant):
        long_message = "I want to go to " + "Tokyo " * 50
        assistant.history = [{"role": "user", "content": long_message}]
        assert "…" in assistant._build_conversation_summary()


# ---------------------------------------------------------------------------
# Exchange rate injection
# ---------------------------------------------------------------------------

class TestExchangeRateInjection:
    def test_rates_injected_when_router_requests_it(self, assistant):
        _route(assistant, "budget_question", needs_exchange_rate=True)
        assistant.exchange.get_supported_rates_only.return_value = {"EUR": 0.92, "JPY": 149.5}
        _mock_reply(assistant, "Here's the budget info.")
        assistant.chat("What's my budget in euros?")
        content = assistant.history[0]["content"]
        assert "VERIFIED LIVE RATES FROM API" in content
        assert "EUR" in content

    def test_rates_not_injected_when_not_needed(self, assistant):
        # Default fixture: needs_exchange_rate=False
        _mock_reply(assistant, "Sure!")
        assistant.chat("I want to go to Paris")
        assert "VERIFIED LIVE RATES" not in assistant.history[0]["content"]


# ---------------------------------------------------------------------------
# Photo injection
# ---------------------------------------------------------------------------

class TestPhotoInjection:
    @pytest.mark.parametrize("message,expected", [
        ("show me photos of Bali", "Bali"),
        ("what does Tokyo look like?", "Tokyo"),
        ("I'm going to New Zealand", "New Zealand"),
        ("trip to Costa Rica", "Costa Rica"),
    ])
    def test_extract_destination(self, assistant, message, expected):
        assert assistant._extract_destination(message) == expected

    def test_photo_injected_when_router_requests_it(self, assistant):
        _route(assistant, "photo_request", needs_photo=True, destination="Bali")
        assistant.unsplash.get_photo_url.return_value = "https://images.unsplash.com/photo-123"
        _mock_reply(assistant, "Here's Bali!")
        assistant.chat("Show me photos of Bali")
        assert "https://images.unsplash.com/photo-123" in assistant.history[0]["content"]

    def test_photo_not_injected_when_not_needed(self, assistant):
        # Default fixture: needs_photo=False
        _mock_reply(assistant, "Sure!")
        assistant.chat("Help me pack for a beach trip")
        assert "Photo of" not in assistant.history[0]["content"]

    def test_photo_fetch_failure_does_not_break_chat(self, assistant):
        _route(assistant, "photo_request", needs_photo=True, destination="Bali")
        assistant.unsplash.get_photo_url.side_effect = Exception("API error")
        _mock_reply(assistant, "Bali is amazing!")
        assert assistant.chat("Show me photos of Bali") == "Bali is amazing!"

    def test_photo_and_rates_both_injected(self, assistant):
        _route(assistant, "budget_question", needs_exchange_rate=True,
               needs_photo=True, destination="Bali")
        assistant.exchange.get_supported_rates_only.return_value = {"IDR": 16350.0}
        assistant.unsplash.get_photo_url.return_value = "https://images.unsplash.com/photo-bali"
        _mock_reply(assistant, "Bali on a budget!")
        assistant.chat("How much does Bali cost? Show me photos")
        content = assistant.history[0]["content"]
        assert "VERIFIED LIVE RATES FROM API" in content
        assert "https://images.unsplash.com/photo-bali" in content

    def test_photo_not_injected_when_unsplash_not_configured(self, assistant):
        _route(assistant, "photo_request", needs_photo=True, destination="Bali")
        assistant.unsplash.is_configured = False
        _mock_reply(assistant, "Bali is beautiful!")
        assistant.chat("Show me photos of Bali")
        assert "Photo of" not in assistant.history[0]["content"]


# ---------------------------------------------------------------------------
# Post-processing: hallucination sanitisation
# ---------------------------------------------------------------------------

NOTE_TEXT = (
    "Note: Live exchange rate for this destination is not available. "
    "Budget shown in USD. Check XE.com for local currency rates."
)

class TestSanitizeResponse:
    def test_hallucinated_conversion_line_removed(self, assistant):
        reply = "Your $2,000 would be approximately 7,400 ILS in Israel."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert "ILS" not in result
        assert NOTE_TEXT in result

    def test_kpw_detected_and_removed(self, assistant):
        reply = "In North Korea that would be 1,800,000 KPW — not that you can go there."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert "KPW" not in result
        assert NOTE_TEXT in result

    def test_unsupported_currency_mention_without_digits_also_removed(self, assistant):
        # Aggressive mode: the whole line is removed regardless of digits
        reply = "The local currency ILS is commonly used here."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert "ILS" not in result
        assert NOTE_TEXT in result

    def test_supported_currency_kept_intact(self, assistant):
        reply = "Your $2,000 ≈ 298,000 JPY — that goes a long way in Tokyo."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert "298,000 JPY" in result
        assert NOTE_TEXT not in result

    def test_usd_always_allowed(self, assistant):
        reply = "Your budget of $2,000 USD is very reasonable for this trip."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert "USD" in result
        assert NOTE_TEXT not in result

    def test_no_unsupported_code_no_change(self, assistant):
        reply = "Japan is a wonderful destination with great food and culture."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert result == reply

    def test_empty_supported_set_skips_sanitisation(self, assistant):
        reply = "That costs about 7,400 ILS or 500 EUR."
        result = assistant._sanitize_response(reply, set())
        assert result == reply

    def test_non_currency_acronyms_not_affected(self, assistant):
        reply = "The API and URL are working. SQL queries are fast."
        result = assistant._sanitize_response(reply, {"EUR", "JPY"})
        assert result == reply


# ---------------------------------------------------------------------------
# ExchangeRateClient unit tests
# ---------------------------------------------------------------------------

class TestExchangeRateClient:
    def test_convert_usd_to_eur(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "GBP": 0.79}):
            assert abs(client.convert(100, "USD", "EUR") - 92.0) < 0.01

    def test_convert_eur_to_usd(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "GBP": 0.79}):
            assert abs(client.convert(92, "EUR", "USD") - 100.0) < 0.01

    def test_convert_cross_rate(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "GBP": 0.79}):
            result = client.convert(100, "EUR", "GBP")
            assert abs(result - (100 / 0.92) * 0.79) < 0.01

    def test_format_for_context_returns_string(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "JPY": 149.5, "USD": 1.0}):
            result = client.format_for_context()
            assert "EUR" in result
            assert "1.00 USD" not in result  # USD should not appear as a destination currency

    def test_is_supported_true_for_existing_currency(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "ILS": 3.72}):
            assert client.is_supported("ILS") is True

    def test_is_supported_false_for_missing_currency(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "JPY": 149.5}):
            assert client.is_supported("XYZ") is False

    def test_is_supported_false_on_api_error(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", side_effect=Exception("Network error")):
            assert client.is_supported("EUR") is False

    def test_unsupported_currency_not_in_format_output(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "JPY": 149.5}):
            result = client.format_for_context(extra_currency="XYZ")
            assert "XYZ" not in result  # silently skipped — caller owns the notice

    def test_known_extra_currency_has_no_warning(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "ILS": 3.72}):
            result = client.format_for_context(extra_currency="ILS")
            assert "NO VERIFIED" not in result
            assert "ILS" in result

    def test_no_extra_currency_has_no_warning(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", return_value={"EUR": 0.92, "JPY": 149.5}):
            result = client.format_for_context(extra_currency=None)
            assert "NO VERIFIED" not in result

    def test_format_for_context_handles_failure(self):
        client = ExchangeRateClient()
        with patch.object(client, "get_rates", side_effect=Exception("Network error")):
            assert client.format_for_context() == ""

    def test_all_known_currencies_populated_after_get_rates(self):
        client = ExchangeRateClient()
        assert client.all_known_currencies == set()  # empty before first call
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rates": {"EUR": 0.92, "JPY": 149.5, "ILS": 3.72}}
        with patch("apis.exchange.httpx.Client") as mock_http:
            mock_http.return_value.__enter__.return_value.get.return_value = mock_resp
            client.get_rates()
        assert "EUR" in client.all_known_currencies
        assert "ILS" in client.all_known_currencies

    def test_all_known_currencies_empty_before_first_call(self):
        assert ExchangeRateClient().all_known_currencies == set()
