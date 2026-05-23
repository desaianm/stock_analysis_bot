"""Probe Polygon and Financial Datasets endpoints we'd use for the quant funnel."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def header(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


# Verify keys
header("Keys present")
for k in ["POLYGON_API_KEY", "FINANCIAL_DATASETS_API_KEY", "FIRECRAWL_API_KEY", "SERPER_API_KEY"]:
    v = os.getenv(k)
    print(f"  {k}: {'set (' + str(len(v)) + ' chars)' if v else 'MISSING'}")


polygon_key = os.getenv("POLYGON_API_KEY")
fd_key = os.getenv("FINANCIAL_DATASETS_API_KEY")


def probe(label, url, headers=None, timeout=15):
    print(f"\n--- {label} ---")
    print(f"  GET {url[:120]}{'...' if len(url) > 120 else ''}")
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        print(f"  status={r.status_code}")
        body = r.text
        if len(body) > 600:
            body = body[:600] + " ..."
        print(f"  body={body}")
    except Exception as exc:
        print(f"  EXC: {exc}")


# Polygon
if polygon_key:
    header("Polygon endpoints")
    probe(
        "Insider trading - Polygon experimental",
        f"https://api.polygon.io/v2/reference/insider-trading/AAPL?apiKey={polygon_key}&limit=5",
    )
    probe(
        "Tickers list (snapshot)",
        f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=10&apiKey={polygon_key}",
    )
    probe(
        "Market snapshot (all US stocks)",
        f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey={polygon_key}&limit=5",
    )

# Financial Datasets
if fd_key:
    header("Financial Datasets endpoints")
    fd_headers = {"X-API-KEY": fd_key}
    probe(
        "Insider trades (AAPL)",
        "https://api.financialdatasets.ai/insider-trades/?ticker=AAPL&limit=5",
        headers=fd_headers,
    )
    probe(
        "Financials (AAPL annual)",
        "https://api.financialdatasets.ai/financials/income-statements/?ticker=AAPL&period=annual&limit=2",
        headers=fd_headers,
    )
    probe(
        "Available tickers",
        "https://api.financialdatasets.ai/financials/income-statements/?ticker=AAPL&period=annual&limit=1",
        headers=fd_headers,
    )
    probe(
        "Company facts",
        "https://api.financialdatasets.ai/company-facts/?ticker=AAPL",
        headers=fd_headers,
    )
    probe(
        "Earnings estimates",
        "https://api.financialdatasets.ai/earnings/estimates/?ticker=AAPL&limit=3",
        headers=fd_headers,
    )
    probe(
        "Price snapshot",
        "https://api.financialdatasets.ai/prices/snapshot/?ticker=AAPL",
        headers=fd_headers,
    )
