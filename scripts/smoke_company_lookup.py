"""Live smoke: company_lookup agent resolves a company name to a ticker."""

import asyncio

from dotenv import load_dotenv

from stockbot.flows.company_lookup import lookup_company

load_dotenv()


async def main():
    for name in ["Microsoft", "Shopify", "Tesla"]:
        result = await lookup_company(name)
        print(f"{name:>12} -> {result}")


if __name__ == "__main__":
    asyncio.run(main())
