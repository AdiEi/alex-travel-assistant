import httpx

# Always-shown currencies — the universally relevant ones for travellers.
_TOP_CURRENCIES = {"EUR", "GBP", "JPY", "AUD", "CAD"}

# Maps destination names (lowercase) to their ISO currency code.
_DESTINATION_TO_CURRENCY: dict[str, str] = {
    # Asia-Pacific
    "japan": "JPY", "tokyo": "JPY", "kyoto": "JPY", "osaka": "JPY",
    "thailand": "THB", "bangkok": "THB", "phuket": "THB", "chiang mai": "THB",
    "indonesia": "IDR", "bali": "IDR", "jakarta": "IDR",
    "vietnam": "VND", "hanoi": "VND", "ho chi minh": "VND", "hoi an": "VND",
    "india": "INR", "delhi": "INR", "mumbai": "INR", "goa": "INR", "jaipur": "INR",
    "singapore": "SGD",
    "south korea": "KRW", "korea": "KRW", "seoul": "KRW", "busan": "KRW",
    "china": "CNY", "beijing": "CNY", "shanghai": "CNY",
    "hong kong": "HKD",
    "taiwan": "TWD", "taipei": "TWD",
    "australia": "AUD", "sydney": "AUD", "melbourne": "AUD",
    "new zealand": "NZD", "auckland": "NZD", "queenstown": "NZD",
    # Europe
    "france": "EUR", "paris": "EUR",
    "germany": "EUR", "berlin": "EUR", "munich": "EUR",
    "italy": "EUR", "rome": "EUR", "milan": "EUR", "venice": "EUR", "florence": "EUR",
    "spain": "EUR", "barcelona": "EUR", "madrid": "EUR", "seville": "EUR",
    "portugal": "EUR", "lisbon": "EUR", "porto": "EUR",
    "greece": "EUR", "athens": "EUR", "santorini": "EUR", "mykonos": "EUR",
    "netherlands": "EUR", "amsterdam": "EUR",
    "belgium": "EUR", "brussels": "EUR",
    "austria": "EUR", "vienna": "EUR",
    "ireland": "EUR", "dublin": "EUR",
    "croatia": "EUR", "dubrovnik": "EUR", "split": "EUR",
    "uk": "GBP", "united kingdom": "GBP", "england": "GBP",
    "london": "GBP", "edinburgh": "GBP", "scotland": "GBP",
    "switzerland": "CHF", "zurich": "CHF", "geneva": "CHF",
    "norway": "NOK", "oslo": "NOK",
    "sweden": "SEK", "stockholm": "SEK",
    "denmark": "DKK", "copenhagen": "DKK",
    "czech republic": "CZK", "czechia": "CZK", "prague": "CZK",
    "hungary": "HUF", "budapest": "HUF",
    "poland": "PLN", "warsaw": "PLN", "krakow": "PLN",
    "turkey": "TRY", "istanbul": "TRY",
    # Americas
    "mexico": "MXN", "cancun": "MXN", "mexico city": "MXN", "oaxaca": "MXN",
    "canada": "CAD", "toronto": "CAD", "vancouver": "CAD", "montreal": "CAD",
    "brazil": "BRL", "rio de janeiro": "BRL", "sao paulo": "BRL",
    "argentina": "ARS", "buenos aires": "ARS",
    "colombia": "COP", "medellin": "COP", "cartagena": "COP",
    "peru": "PEN", "lima": "PEN", "cusco": "PEN",
    "costa rica": "CRC",
    "chile": "CLP", "santiago": "CLP",
    # Middle East / Africa
    "israel": "ILS", "tel aviv": "ILS", "jerusalem": "ILS",
    "uae": "AED", "dubai": "AED", "abu dhabi": "AED",
    "morocco": "MAD", "marrakech": "MAD", "casablanca": "MAD",
    "south africa": "ZAR", "cape town": "ZAR", "johannesburg": "ZAR",
    "egypt": "EGP", "cairo": "EGP",
    "kenya": "KES", "nairobi": "KES",
    "tanzania": "TZS", "zanzibar": "TZS",
}

BASE_URL = "https://open.er-api.com/v6/latest/USD"


class ExchangeRateClient:
    def __init__(self) -> None:
        self._cached_all_currencies: set[str] = set()

    @property
    def all_known_currencies(self) -> set[str]:
        """All currency codes returned by the most recent successful API call."""
        return self._cached_all_currencies

    def get_rates(self) -> dict[str, float]:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(BASE_URL)
            response.raise_for_status()
            data = response.json()["rates"]
            self._cached_all_currencies = set(data.keys())
            return data

    def get_travel_rates(self) -> dict[str, float]:
        all_rates = self.get_rates()
        return {k: v for k, v in all_rates.items() if k in _TOP_CURRENCIES}

    def currency_for(self, destination: str | None) -> str | None:
        """Return the ISO currency code for a destination name, or None if unknown."""
        if not destination:
            return None
        return _DESTINATION_TO_CURRENCY.get(destination.lower().strip())

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        rates = self.get_rates()
        if from_currency == "USD":
            return amount * rates[to_currency]
        elif to_currency == "USD":
            return amount / rates[from_currency]
        # Cross-rate via USD
        return (amount / rates[from_currency]) * rates[to_currency]

    def is_supported(self, currency_code: str) -> bool:
        """Return True if the currency code is present in the live rate feed."""
        try:
            return currency_code in self.get_rates()
        except Exception:
            return False

    def get_supported_rates_only(self, extra_currency: str | None = None) -> dict[str, float]:
        """Return ONLY rates confirmed in the live API response.

        Includes the top currencies (EUR, GBP, JPY, AUD, CAD) plus extra_currency
        if it is present in the API response. Currencies absent from the API are
        silently excluded — callers must not invent rates for missing currencies.
        """
        all_rates = self.get_rates()
        currencies = set(_TOP_CURRENCIES)
        if extra_currency:
            currencies.add(extra_currency)
        return {k: v for k, v in all_rates.items() if k in currencies}

    def format_for_context(self, extra_currency: str | None = None) -> str:
        """Return a compact rate string for the top currencies plus the destination's.

        Shows EUR, GBP, JPY, AUD, CAD by default. If extra_currency is provided and
        present in the live feed, it is appended. Unsupported currencies are silently
        skipped here — callers should check is_supported() first and inject a notice
        if needed.
        """
        try:
            all_rates = self.get_rates()
            currencies = set(_TOP_CURRENCIES)
            if extra_currency and extra_currency not in currencies and extra_currency in all_rates:
                currencies.add(extra_currency)
            rates = {k: v for k, v in all_rates.items() if k in currencies}
            parts = [f"1 USD = {v:.2f} {k}" for k, v in sorted(rates.items())]
            return "[Live exchange rates: " + ", ".join(parts) + "]"
        except Exception:
            return ""
