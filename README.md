# SolaScript Crypto Scalper

An experimental family of asynchronous Kraken cryptocurrency scalpers developed through multiple strategy iterations.

The primary implementation is [SolaScript.py](SolaScript.py), based on the newest available v5.2 engine. Earlier versions are retained under [archive/](archive/) to document the project's evolution.

> [!CAUTION]
> These programs contain order-execution code and may place real trades when supplied with exchange credentials. They are experimental research, not financial advice, and have not been independently audited. Review [RISK_DISCLOSURE.md](RISK_DISCLOSURE.md) before running anything.

## Strategy themes

Across its iterations, SolaScript explores:

- Asynchronous market scanning
- Relative-volume and Bollinger Band signals
- Momentum and mean-reversion modes
- Time-of-day strategy switching
- Multi-position management
- Stop-loss, take-profit, stagnation, and trailing exits
- SQLite trade and position state
- Kraken execution through CCXT

## Versions

- **Current:** `SolaScript.py` — extracted from v5.2
- `archive/v4-unicorn.py` — dual-personality Unicorn Hunter / Deep Researcher
- `archive/v3-t2.py` and `archive/v3.py` — SQLite-backed iterations
- `archive/v2.py`
- `archive/hybrid.py`
- `archive/v1.py`

Archived versions are provided for study and may contain obsolete assumptions or incomplete safeguards.

## Installation

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Credentials

Use a dedicated Kraken API key. Enable only the permissions genuinely required by the selected script and keep withdrawals disabled.

Never commit `.env`, databases, logs, or state files.

## Running

```sh
python SolaScript.py
```

Read the source first. Verify whether the chosen version submits live orders, review its position sizing, and adapt it to paper trading before considering live use.

## License

[MIT](LICENSE)
