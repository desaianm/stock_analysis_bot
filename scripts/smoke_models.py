"""Live agent smoke: send tiny prompts to verify configured model ids resolve."""

import asyncio

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat

load_dotenv()


async def smoke():
    for model_id in ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-nano"]:
        try:
            agent = Agent(
                model=OpenAIChat(id=model_id, temperature=1, max_completion_tokens=20),
                instructions=["Reply with exactly one word."],
                markdown=False,
            )
            response = await agent.arun("ping?")
            content = getattr(response, "content", None) or str(response)
            print(f"OK   {model_id}: {str(content).strip()[:60]}")
        except Exception as exc:
            msg = str(exc)[:200]
            print(f"FAIL {model_id}: {type(exc).__name__}: {msg}")


if __name__ == "__main__":
    asyncio.run(smoke())
