# Agent Prompts Directory

This directory contains all agent prompts and task descriptions used throughout the stock analysis bot. Prompts are organized by module/flow for easy maintenance and version control.

## Directory Structure

```
prompts/
├── README.md                    # This file
├── undervalued/                 # Undervalued stock flow (Agno framework)
│   ├── screening_agent_instructions.txt
│   ├── screening_prompt.txt
│   ├── turnaround_agent_instructions.txt
│   ├── turnaround_prompt.txt
│   ├── reddit_sentiment_agent_instructions.txt
│   └── reddit_sentiment_prompt.txt
├── agents/                      # Reusable CrewAI agents
│   ├── markdown_report_creator.txt
│   ├── stock_analysis_agent.txt
│   ├── chart_creator.txt
│   ├── markdown_writer.txt
│   ├── company_research_agent.txt
│   └── company_lookup_agent.txt
└── single_stock/                # Single stock analysis flow
    ├── financial_data_researcher.txt
    ├── technical_analyst.txt
    ├── fundamental_analyst.txt
    ├── market_sentiment_analyst.txt
    ├── investment_report_writer.txt
    ├── gather_basic_data_task.txt
    ├── technical_analysis_task.txt
    └── fundamental_analysis_task.txt
```

## Prompt Format

### Agent Definition Files

Agent definition files contain three sections (for CrewAI agents):
- **Role**: The agent's job title/function
- **Goal**: What the agent aims to achieve
- **Backstory**: The agent's expertise and context

Example:
```
Role: Stock Analysis Agent

Goal: Fetch all the information using the given tools and create a detailed report.

Backstory: Expert in analyzing stock data with 10 years of experience...
```

### Agno Agent Instructions

For Agno agents (used in undervalued flow), instructions are line-separated directives:
```
You identify fundamentally strong but underpriced US equities.
Always call the available Python functions to gather real-time prices.
Return polished markdown with clear sections.
```

### Task Description Files

Task description files contain the task prompt with placeholders for dynamic values:
```
Research and gather the following information for {ticker}:
1. Current stock price and basic company information
2. Historical stock prices for the past 5 years
...
```

## Loading Prompts

### Undervalued Flow (Agno)

The undervalued flow uses the `load_prompt()` helper function:

```python
from pathlib import Path

def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "undervalued" / f"{prompt_name}.txt"
    return prompt_path.read_text(encoding="utf-8")

# Usage
screening_instructions = load_prompt("screening_agent_instructions").strip().split('\n')
template = load_prompt("screening_prompt")
prompt = template.format(ticker="AAPL", max_price="100.00", ...)
```

### CrewAI Agents

For CrewAI agents, prompts are loaded using the `textwrap.dedent()` pattern:

```python
from textwrap import dedent

# Read from file instead of inline
agent = Agent(
    role="Stock Analysis Agent",
    goal=dedent("""Fetch all the information..."""),
    backstory=dedent("""Expert in analyzing..."""),
    tools=[...],
)
```

## Editing Prompts

When modifying agent behavior:

1. **Edit the prompt file** in this directory (not inline code)
2. **Test the changes** by running the relevant flow
3. **Document significant changes** in commit messages
4. **Version control** allows easy rollback if needed

## Benefits

- **Centralized Management**: All prompts in one location
- **Easy Comparison**: See prompt evolution in git history
- **No Code Changes**: Iterate on prompts without touching Python code
- **Better Testing**: A/B test different prompt versions
- **Documentation**: Clear separation between logic and prompts
- **Collaboration**: Non-coders can improve prompts

## Dynamic Values

Prompts use Python format string placeholders for dynamic values:

- `{ticker}` - Stock ticker symbol
- `{timestamp}` - Current date/time
- `{max_price}` - Maximum price constraint
- `{reddit_section}` - Reddit summary (conditional)
- `{historical_data}` - Historical market data
- `{screening_summary}` - Previous stage output

## Notes

- All prompt files use UTF-8 encoding
- Keep prompts concise but comprehensive
- Use markdown formatting in output instructions
- Include specific examples in agent backstories for better performance
- Test prompt changes against existing golden outputs
