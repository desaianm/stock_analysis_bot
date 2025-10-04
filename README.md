# Financial Analysis Discord Bot

A sophisticated Discord bot powered by AI agents for comprehensive stock market analysis and investment recommendations.

## Features

- **Top 20 Stock Recommendations** (`/top20`)
  - Quantitative screening of market candidates
  - Sector allocation optimization
  - Head-to-head stock tournaments
  - Portfolio optimization and risk analysis
  - Detailed investment thesis for each selection

- **Undervalued Stock Analysis** (`/undervalued`)
  - Screens for undervalued stocks based on multiple criteria
  - Analyzes financial metrics, technical indicators, and market sentiment
  - Provides comprehensive valuation analysis
  - Risk assessment and monitoring guidelines

- **Individual Company Analysis** (`/analyze [company_name]`)
  - Detailed company lookup and ticker symbol identification
  - Comprehensive financial analysis
  - Technical analysis and charts
  - Market sentiment analysis
  - Investment recommendations

## Setup

1. Install required dependencies:


## Project Structure

- `main.py` – Discord bot entrypoint and slash-command handlers.
- `stockbot/flows/` – CrewAI orchestration modules (`single_stock.py`, `recommendations.py`, `undervalued.py`).
- `stockbot/agents/financial.py` – Reusable agent factories for lookup/reporting tasks.
- `stockbot/portfolio/analysis.py` – Portfolio Crew orchestration used by the `/portfolio` command.
- `stockbot/tools/data.py` – QuickFS, market-data, charting, and markdown helper tools.
- `stockbot/tasks/workflows.py` – Company lookup task builder and portfolio image ingestion helper.

## Commands

- `/top20` - Get top 20 stock recommendations
- `/undervalued` - Get undervalued stock analysis
- `/analyze [company_name]` - Analyze a specific company
- `/help` - Show available commands

## Dependencies

Key dependencies include:
- crewai
- langchain
- discord.py
- quickfs
- yfinance
- matplotlib
- pandas
- numpy

## Output

Analysis reports are saved in the `reports/` directory in Markdown format.

## Notes

- The bot uses multiple AI agents working together to provide comprehensive analysis
- Analysis may take several minutes to complete due to the depth of research
- All financial data is sourced from reputable providers through official APIs

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

