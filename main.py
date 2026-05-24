"""Discord bot entrypoint. Routes slash commands to Agno-powered analysis flows."""

import asyncio
import os
import traceback
from datetime import datetime
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from stockbot.flows.company_lookup import lookup_company
from stockbot.flows.recommendations import InvestmentPreferences, Top20StocksFlow
from stockbot.flows.single_stock import SingleStockAnalysisFlow
from stockbot.flows.undervalued import (
    UndervaluedAnalysisFlow,
    ValueScreeningPreferences,
)
from stockbot.scheduler.daily_tracker import DailyPortfolioScheduler

load_dotenv()


def get_default_preferences():
    """Default preferences for both analysis flows."""
    investment_prefs = InvestmentPreferences(
        strategy="balanced",
        risk_tolerance="moderate",
        time_horizon="5",
        min_market_cap=1.0,
        max_position_size=0.10,
        preferred_sectors=[],
        excluded_sectors=[],
        esg_focus=False,
        dividend_focus=False,
        international_exposure=False,
    )
    value_prefs = ValueScreeningPreferences(
        max_price=100.0,
        min_price=5.0,
        min_volume=500000,
        max_pe=25.0,
        min_market_cap=300000000,
        min_current_ratio=1.5,
        max_debt_equity=2.0,
        price_vs_high=0.4,
    )
    return investment_prefs, value_prefs


investment_prefs, value_prefs = get_default_preferences()
under_flow = UndervaluedAnalysisFlow(value_prefs)
stock_recommendation_flow = Top20StocksFlow(investment_prefs)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

portfolio_scheduler = DailyPortfolioScheduler()
daily_update_task = portfolio_scheduler.create_task()

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
        portfolio_scheduler.start(daily_update_task)
        print("Daily portfolio scheduler started")
    except Exception as exc:
        print(f"Failed to sync commands: {exc}")


@bot.tree.command(name="top20", description="Get top 20 stock recommendations")
async def get_top_20(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        await interaction.followup.send("Analyzing top 20 stocks... Please wait.")
        results = await stock_recommendation_flow.execute_portfolio_construction()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"top20_report_{timestamp}.md"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Top 20 Stock Recommendations Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results)

        await interaction.followup.send(
            "Analysis complete! Here's your report:",
            file=discord.File(report_file),
        )
        report_file.unlink()
    except Exception as exc:
        await interaction.followup.send("An error occurred while processing your request.")
        print(f"Error in top20 command: {exc}")
        traceback.print_exc()


@bot.tree.command(name="portfolio", description="View portfolio performance and learning insights")
async def view_portfolio(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        from stockbot.flows.performance_tracker import PerformanceTrackerFlow

        tracker = PerformanceTrackerFlow()
        report = await tracker.generate_performance_report()

        if len(report) <= 1900:
            await interaction.followup.send(f"```markdown\n{report}\n```")
        else:
            chunks = [report[i : i + 1900] for i in range(0, len(report), 1900)]
            await interaction.followup.send(f"```markdown\n{chunks[0]}\n```")
            for chunk in chunks[1:]:
                await interaction.channel.send(f"```markdown\n{chunk}\n```")
    except Exception as exc:
        await interaction.followup.send(f"Error generating portfolio report: {str(exc)}")
        print(f"Error in portfolio command: {exc}")
        traceback.print_exc()


@bot.tree.command(name="undervalued", description="Get undervalued stock analysis")
async def get_undervalued(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        await interaction.followup.send("Analyzing undervalued stocks... Please wait.")
        results = await under_flow.execute_undervalued_analysis()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"undervalued_report_{timestamp}.md"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Undervalued Stocks Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results)

        await interaction.followup.send(
            "Analysis complete! Here's your report:",
            file=discord.File(report_file),
        )
        report_file.unlink()
    except Exception as exc:
        await interaction.followup.send("An error occurred while processing your request.")
        print(f"Error in undervalued command: {exc}")
        traceback.print_exc()


@bot.tree.command(name="analyze", description="Analyze a specific company")
async def analyze_company(interaction: discord.Interaction, company_name: str):
    try:
        await interaction.response.defer()
        await interaction.followup.send(
            f"Searching for '{company_name}' and analyzing... Please wait."
        )

        try:
            company_data = await lookup_company(company_name)
        except Exception as exc:
            await interaction.followup.send(f"Error finding company symbol: {str(exc)}")
            return

        if not company_data or not company_data.get("ticker"):
            await interaction.followup.send(
                f"Could not find ticker symbol for company: {company_name}"
            )
            return

        ticker = company_data["ticker"]
        stock_flow = SingleStockAnalysisFlow(ticker)
        await stock_flow.execute_analysis()

        report_file = f"stock_analysis_{ticker}_{datetime.now().strftime('%Y%m%d')}.md"
        if not os.path.exists(report_file):
            await interaction.followup.send("Analysis failed to generate report.")
            return

        await interaction.followup.send(
            f"Analysis complete for {company_name} ({ticker})! Here's your report:",
            file=discord.File(report_file),
        )
        Path(report_file).unlink()
    except Exception as exc:
        await interaction.followup.send("An error occurred while processing your request.")
        print(f"Error in analyze command: {str(exc)}")
        traceback.print_exc()


@bot.tree.command(name="help", description="Show available commands")
async def help_command(interaction: discord.Interaction):
    help_text = (
        "Available commands:\n"
        "/top20 - Get top 20 stock recommendations\n"
        "/undervalued - Get undervalued stock analysis\n"
        "/analyze <company_name> - Single-company investment report\n"
        "/portfolio - View tracked portfolio performance and learning insights"
    )
    await interaction.response.send_message(help_text)


def main():
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("Discord token not found in environment variables")
    bot.run(token)


if __name__ == "__main__":
    main()
