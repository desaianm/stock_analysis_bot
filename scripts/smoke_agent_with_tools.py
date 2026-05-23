"""Live smoke: a minimal Agno agent that must call get_real_time_quote(AAPL)."""

import asyncio
import json

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from stockbot.tools.data import RealTimeQuoteTool

load_dotenv()

_quote_tool = RealTimeQuoteTool()


def get_real_time_quote(symbol: str) -> str:
    """Return price, volume, market cap for a stock symbol."""
    try:
        price, volume, market_cap = _quote_tool.run(symbol)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {"symbol": symbol.upper(), "price": price, "volume": volume, "market_cap": market_cap}
    )


async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-5.4-mini", temperature=1, max_completion_tokens=300),
        instructions=[
            "You are a stock quote assistant. When asked about a stock, call the tool, then summarize the price in one short sentence."
        ],
        tools=[get_real_time_quote],
        markdown=False,
    )
    response = await agent.arun("What's the current price of AAPL?")
    content = getattr(response, "content", None) or str(response)
    print("Agent response:", str(content).strip()[:300])


if __name__ == "__main__":
    asyncio.run(main())
