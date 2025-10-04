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

- `main.py` - Main Discord bot implementation and command handlers
- `agents.py` - AI agent definitions and configurations
- `tasks.py` - Task definitions for AI agents
- `tools.py` - Custom tools for financial analysis and data retrieval
- `flow.py` - Analysis workflow definitions
- `stock_recomendation_flow.py` - Top 20 stocks analysis implementation
- `stock_under.py` - Undervalued stocks analysis implementation

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


