import math
import os
import random
import re
from datetime import datetime
from html import unescape
from typing import Any, Dict, List, Optional, Type
from urllib.parse import parse_qs, unquote, urlparse

import matplotlib
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot
import matplotlib.pyplot as plt

load_dotenv()


class BaseTool:
    """Minimal Agno-compatible tool base. Subclasses define ._run(); we expose .run()."""
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)

def _clean_symbol(symbol: str) -> str:
    """Normalize symbols for yfinance while preserving exchange suffixes."""
    return (symbol or "").strip().upper().replace(":US", "")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _latest_number(values: list[Any]) -> Optional[float]:
    for value in reversed(values or []):
        number = _safe_float(value)
        if number is not None:
            return number
    return None


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _statement_series(statement: Any, row_names: list[str]) -> list[float]:
    """Return one yfinance statement row as oldest-to-newest numeric values."""
    if statement is None or getattr(statement, "empty", True):
        return []

    lower_index = {str(index).lower(): index for index in statement.index}
    row_key = None
    for row_name in row_names:
        row_key = lower_index.get(row_name.lower())
        if row_key is not None:
            break
    if row_key is None:
        return []

    values = []
    for value in reversed(statement.loc[row_key].tolist()):
        number = _safe_float(value)
        if number is not None:
            values.append(number)
    return values


def _build_yfinance_financial_report(symbol: str) -> Dict[str, Any]:
    """Build a QuickFS-like financial metric payload from yfinance."""
    import yfinance as yf

    normalized_symbol = _clean_symbol(symbol)
    stock = yf.Ticker(normalized_symbol)
    info = stock.info or {}

    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cash_flow

    revenue = _statement_series(
        income_stmt,
        ["Total Revenue", "Operating Revenue", "Revenue"],
    )
    net_income = _statement_series(
        income_stmt,
        ["Net Income", "Net Income Common Stockholders"],
    )
    operating_cash_flow = _statement_series(
        cash_flow,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
    )
    free_cash_flow = _statement_series(cash_flow, ["Free Cash Flow"])
    if not free_cash_flow and operating_cash_flow:
        capex = _statement_series(
            cash_flow,
            ["Capital Expenditure", "Capital Expenditures"],
        )
        free_cash_flow = [
            ocf + capex_value
            for ocf, capex_value in zip(operating_cash_flow[-len(capex) :], capex)
        ]

    total_debt = _statement_series(balance_sheet, ["Total Debt"])
    shareholders_equity = _statement_series(
        balance_sheet,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity",
        ],
    )
    current_assets = _statement_series(balance_sheet, ["Current Assets"])
    current_liabilities = _statement_series(balance_sheet, ["Current Liabilities"])

    latest_current_assets = _latest_number(current_assets)
    latest_current_liabilities = _latest_number(current_liabilities)
    latest_total_debt = _latest_number(total_debt)
    latest_equity = _latest_number(shareholders_equity)

    current_ratio = (
        _safe_float(info.get("currentRatio"))
        or _ratio(latest_current_assets, latest_current_liabilities)
    )
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    if debt_to_equity is not None and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100
    if debt_to_equity is None:
        debt_to_equity = _ratio(latest_total_debt, latest_equity)

    report = {
        "symbol": normalized_symbol,
        "source": "yfinance",
        "revenue": revenue,
        "net_income": net_income,
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
        "total_debt": total_debt,
        "shareholders_equity": shareholders_equity,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "price_to_earnings": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "eps": _safe_float(info.get("trailingEps")),
        "market_cap": _safe_float(info.get("marketCap")),
        "current_price": _safe_float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        ),
        "volume": _safe_float(info.get("volume") or info.get("regularMarketVolume")),
    }

    if not any(
        report.get(metric)
        for metric in ("revenue", "net_income", "total_debt", "current_assets")
    ) and report.get("market_cap") is None:
        raise RuntimeError(f"No yfinance financial data returned for {normalized_symbol}")

    return report


def _duckduckgo_html_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Small no-key web-search fallback used when Tavily/DDGS is unavailable."""
    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "stock-analysis-bot/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    html = response.text
    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for url, title, snippet in pattern.findall(html):
        parsed = urlparse(unescape(url))
        if parsed.path == "/l/":
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            url = unquote(target) if target else url
        clean_title = re.sub("<[^>]+>", "", unescape(title)).strip()
        clean_snippet = re.sub("<[^>]+>", "", unescape(snippet)).strip()
        results.append({"title": clean_title, "url": url, "snippet": clean_snippet})
        if len(results) >= max_results:
            break
    return results


def _fallback_web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS

        return [
            {
                "title": item.get("title"),
                "url": item.get("href") or item.get("url"),
                "snippet": item.get("body") or item.get("snippet"),
            }
            for item in DDGS().text(query, max_results=max_results)
        ]
    except Exception:
        try:
            from duckduckgo_search import DDGS

            return [
                {
                    "title": item.get("title"),
                    "url": item.get("href") or item.get("url"),
                    "snippet": item.get("body") or item.get("snippet"),
                }
                for item in DDGS().text(query, max_results=max_results)
            ]
        except Exception:
            return _duckduckgo_html_search(query, max_results=max_results)


class TavilySearchInput(BaseModel):
    """Input schema for TavilySearch."""
    query: str = Field(..., description="The query to search the web with")

class TavilySearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = """Search the web. Uses Tavily when TAVILY_API_KEY is set; otherwise falls back to DuckDuckGo."""
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            try:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": 10,
                        "include_raw_content": False,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json() or {}
                return payload.get("results", [])
            except Exception:
                pass
        return _fallback_web_search(query, max_results=8)


class CreateChartInput(BaseModel):
    metric: str
    data: List[float]

class CreateChartOutput(BaseModel):
    file_path: str

class ChartingToolInput(BaseModel):
    """Input schema for ChartingTool."""
    metric_name: str = Field(..., description="Name of the metric to be visualized")
    data: List[float] = Field(..., description="List of numerical data points")

class ChartingTool(BaseTool):
    name: str = "Create a chart of the data"
    description: str = """Creates a bar chart graphic based on the provided metric and data.

        Parameters:
        - metric_name (str): The name of the metric to be visualized on the chart.
        - data (List[float]): A list of numerical data points representing the metric over time.

        Returns:
        - file_path (str): The file path to the saved chart image.
        
        Example:
        - Input: metric_name='revenue', data=[100000000, 150000000, 120000000, 200000000, 180000000]
        - Output: CreateChartOutput(file_path='./plots/revenue_chart.png')
        """
    args_schema: Type[BaseModel] = ChartingToolInput

    def _run(self, metric_name: str, data: List) -> CreateChartOutput:
        years = list(range(len(data)))
        bar_color = f'#{random.randint(0, 0xFFFFFF):06x}'

        os.makedirs("plots", exist_ok=True)
        
        plt.figure(figsize=(10, 6))  # Create a new figure
        plt.bar(years, data, color=bar_color)
        plt.xlabel('Years')
        plt.title(metric_name)
        
        file_path = f"plots/{metric_name.replace(' ', '_')}_chart.png"
        plt.savefig(file_path, format='png')
        plt.close()  # Close the figure to free up memory
        
        return CreateChartOutput(file_path=file_path)


class FinancialReportToolInput(BaseModel):
    """Input schema for FinancialReportTool."""
    symbol: str = Field(..., description="The stock symbol to create a financial report for")

class FinancialReportTool(BaseTool):
    name: str = "Create a financial report"
    description: str = """
    Useful to create a financial report from symbol.

    :param symbol: str, the symbol to create a financial report for
    :return data: dict, a dictionary containing the financial data

    Example:
    - Input: symbol="AAPL"
    - Output: {
        "revenue": [265595000000, 274515000000, 365817000000, 394328000000, 383285000000],
        "net_income": [55256000000, 57411000000, 94680000000, 99803000000, 96995000000],
        "eps": [3.28, 3.31, 5.61, 6.11, 6.15]
    }
    """
    args_schema: Type[BaseModel] = FinancialReportToolInput

    def _run(self, symbol: str) -> Dict[str, Any]:
        return _build_yfinance_financial_report(symbol)


import yfinance as yf


class StockPriceDataToolInput(BaseModel):
    """Input schema for StockPriceDataTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve data for")
    period: str = Field(..., description="Time period for data retrieval (e.g., '5y', '6mo')")

class StockPriceDataTool(BaseTool):
    name: str = "Retrieve historical stock price data"
    description: str = """Useful to retrieve historical stock price data from Yahoo Finance based on the given symbol and time period.

        :param symbol: str, only one symbol
        :param period: str, time period for data retrieval (e.g., '5y' for 5 years, '6mo' for 6 months)
        :return value: list, A list containing the closing prices
        Return value example: [...closing_prices]

        Example:
        - Input: symbol="AAPL", period="5y"
        - Output: [145.85, 147.92, 149.26, 150.65, 148.97, ...]
        """
    args_schema: Type[BaseModel] = StockPriceDataToolInput

    def _run(self, symbol: str, period: str) -> List:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty or "Close" not in data:
            return []
        return [
            round(float(value), 4)
            for value in data["Close"].dropna().tolist()
            if _safe_float(value) is not None
        ]

class RealTimeQuoteToolInput(BaseModel):
    """Input schema for RealTimeQuoteTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve real-time quote for")

class RealTimeQuoteTool(BaseTool):
    name: str = "Retrieve real-time stock quote"
    description: str = """Useful to retrieve real-time stock quote from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing current price, volume, and market cap
        Return value example: [current_price, volume, market_cap]

        Example:
        - Input: symbol="AAPL"
        - Output: [150.65, 75000000, 2500000000000]
        """
    args_schema: Type[BaseModel] = RealTimeQuoteToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        volume = info.get("volume") or info.get("regularMarketVolume")
        market_cap = info.get("marketCap")
        try:
            fast_info = stock.fast_info
            price = price or fast_info.get("last_price")
            market_cap = market_cap or fast_info.get("market_cap")
        except Exception:
            pass
        return [_safe_float(price), _safe_float(volume), _safe_float(market_cap)]

class OptionsChainToolInput(BaseModel):
    """Input schema for OptionsChainTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve options data for")
    expiration_date: str = Field(..., description="Expiration date for options (format: 'YYYY-MM-DD')")

class OptionsChainTool(BaseTool):
    name: str = "Retrieve options chain data"
    description: str = """Useful to retrieve options chain data from Yahoo Finance based on the given symbol and expiration date.

        :param symbol: str, only one symbol
        :param expiration_date: str, expiration date for options (format: 'YYYY-MM-DD')
        :return value: list, A list containing calls and puts data, and the actual expiration date used
        Return value example: [[...calls_data], [...puts_data], 'YYYY-MM-DD']

        Example:
        - Input: symbol="AAPL", expiration_date="2023-07-21"
        - Output: [[{'strike': 140, 'lastPrice': 10.5, ...}], [{'strike': 140, 'lastPrice': 0.5, ...}], '2023-07-21']
        """
    args_schema: Type[BaseModel] = OptionsChainToolInput

    def _run(self, symbol: str, expiration_date: str) -> List:
        stock = yf.Ticker(symbol)
        
        # Get available expiration dates
        expirations = stock.options

        if not expirations:
            return [[], [], None]  # No options data available

        # Convert input expiration_date to datetime
        target_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()

        # Find the closest available expiration date
        closest_date = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - target_date))

        # Fetch options data for the closest available date
        options = stock.option_chain(date=closest_date)
        
        return [options.calls.to_dict('records'), options.puts.to_dict('records'), closest_date]

class AnalystRecommendationsToolInput(BaseModel):
    """Input schema for AnalystRecommendationsTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve analyst recommendations for")

class AnalystRecommendationsTool(BaseTool):
    name: str = "Retrieve analyst recommendations"
    description: str = """Useful to retrieve analyst recommendations from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing analyst recommendations
        Return value example: [...recommendations]

        Example:
        - Input: symbol="AAPL"
        - Output: [{'firm': 'Barclays', 'toGrade': 'Overweight', 'fromGrade': 'Equal Weight', 'action': 'up'}, ...]
        """
    args_schema: Type[BaseModel] = AnalystRecommendationsToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        if recommendations is None or recommendations.empty:
            return []
        dict_recoms = recommendations.to_dict('records')
        return dict_recoms

class StockNewsToolInput(BaseModel):
    """Input schema for StockNewsTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve news for")

class StockNewsTool(BaseTool):
    name: str = "Retrieve recent stock news"
    description: str = """Useful to retrieve recent news headlines from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing recent news headlines
        Return value example: [...headlines]

        Example:
        - Input: symbol="AAPL"
        - Output: ["Apple Launches New iPhone", "Apple's Q2 Earnings Beat Expectations", ...]
        """
    args_schema: Type[BaseModel] = StockNewsToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        news_items = stock.news or []
        headlines: List[Dict[str, Any]] = []

        for item in news_items[:5]:
            title = url = published = None
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, dict):
                    title = content.get("title") or content.get("headline")
                    published = content.get("pubDate") or content.get("displayTime")
                    canonical = content.get("canonicalUrl") or {}
                    clickthrough = content.get("clickThroughUrl") or {}
                    if isinstance(canonical, dict):
                        url = canonical.get("url")
                    if not url and isinstance(clickthrough, dict):
                        url = clickthrough.get("url")
                title = title or item.get("title") or item.get("headline")
                url = url or item.get("link") or item.get("url")
                published = published or item.get("publisher") or item.get("providerPublishTime")
            if title:
                headlines.append(
                    {
                        "title": str(title)[:180],
                        "url": url,
                        "published": published,
                    }
                )

        return headlines

class CompanyInfoToolInput(BaseModel):
    """Input schema for CompanyInfoTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve company information for")

class CompanyInfoTool(BaseTool):
    name: str = "Retrieve company information"
    description: str = """Useful to retrieve detailed company information from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing key company information
        Return value example: [long_name, sector, industry, market_cap, forward_pe, dividend_yield]

        Example:
        - Input: symbol="AAPL"
        - Output: ["Apple Inc.", "Technology", "Consumer Electronics", 2500000000000, 25.5, 0.65]
        """
    args_schema: Type[BaseModel] = CompanyInfoToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        info = stock.info
        return [
            info.get('longName'),
            info.get('sector'),
            info.get('industry'),
            info.get('marketCap'),
            info.get('forwardPE'),
            info.get('dividendYield')
        ]

# class CompanySearchToolInput(BaseModel):
#     """Input schema for CompanySearchTool."""
#     company_name: str = Field(..., description="Company name to search for")

# class CompanySearchTool(BaseTool):
#     name: str = "Search for a company by name"
#     description: str = """Useful to search for a company by name and retrieve its symbol."""
#     args_schema: Type[BaseModel] = CompanySearchToolInput

#     def _run(self, company_name: str) -> str:
#         return yf.Ticker.search(company_name)


class WebSearchTool():

    def run(self, query: str) -> List[Dict[str, Any]]:
        if os.getenv("EXA_API_KEY"):
            try:
                from exa_py import Exa

                exa = Exa(api_key=os.getenv("EXA_API_KEY"))
                result = exa.search_and_contents(query, text=True, type="auto")
                if hasattr(result, "results"):
                    return [
                        {
                            "title": getattr(item, "title", None),
                            "url": getattr(item, "url", None),
                            "snippet": getattr(item, "text", None),
                        }
                        for item in result.results[:8]
                    ]
                return result
            except Exception:
                pass
        return _fallback_web_search(query, max_results=8)
    
