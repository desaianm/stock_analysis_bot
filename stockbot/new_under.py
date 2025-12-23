from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
from tools import StockPriceDataTool

load_dotenv()

# Create the Agent
agno_agent = Agent(
    name="Agno Agent",
    model=Claude(id="claude-sonnet-4-5"),
    # Add a database to the Agent
    db=SqliteDb(db_file="agno.db"),
    instructions="answer the question given in one word",
    # Add the Agno MCP server to the Agent
    tools=[StockPriceDataTool()],
    # Add the previous session history to the context
    add_history_to_context=True,
    markdown=True,
)


response = agno_agent.run("What is AI ?")
print(response.content)