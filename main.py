import asyncio
from stock_recomendation_flow import Top20StocksFlow, InvestmentPreferences
from stock_under import UndervaluedAnalysisFlow, ValueScreeningPreferences
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime


load_dotenv()

def get_default_preferences():
    """Get default preferences for both analysis flows"""
    # Default investment preferences for Top20StocksFlow
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
        international_exposure=False
    )
    
    # Default value screening preferences for UndervaluedAnalysisFlow
    value_prefs = ValueScreeningPreferences(
        max_price=100.0,
        min_price=5.0,
        min_volume=500000,
        max_pe=25.0,
        min_market_cap=300000000,
        min_current_ratio=1.5,
        max_debt_equity=2.0,
        price_vs_high=0.4
    )
    
    return investment_prefs, value_prefs

# Initialize flows with preferences
investment_prefs, value_prefs =  get_default_preferences()
under_flow = UndervaluedAnalysisFlow(value_prefs)
stock_recomendation_flow = Top20StocksFlow(investment_prefs)

# Set up bot with command prefix and intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Create a directory for reports if it doesn't exist
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='top20')
async def get_top_20(ctx):
    """Get top 20 stock recommendations"""
    try:
        await ctx.send("Analyzing top 20 stocks... Please wait.")
        results = await stock_recomendation_flow.execute_portfolio_construction()
        
        # Create report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"top20_report_{timestamp}.md"
        
        with open(report_file, "w", encoding='utf-8') as f:
            f.write("# Top 20 Stock Recommendations Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results)
        
        # Send report as attachment
        await ctx.send(
            "Analysis complete! Here's your report:",
            file=discord.File(report_file)
        )
        
    except Exception as e:
        await ctx.send("An error occurred while processing your request.")
        print(f"Error in top20 command: {e}")

@bot.command(name='undervalued')
async def get_undervalued(ctx):
    """Get undervalued stock analysis"""
    try:
        await ctx.send("Analyzing undervalued stocks... Please wait.")
        results = await under_flow.execute_undervalued_analysis()
        
        # Create report file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"undervalued_report_{timestamp}.md"
        
        with open(report_file, "w", encoding='utf-8') as f:
            f.write("# Undervalued Stocks Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results)
        
        # Send report as attachment
        await ctx.send(
            "Analysis complete! Here's your report:",
            file=discord.File(report_file)
        )
        
    except Exception as e:
        await ctx.send("An error occurred while processing your request.")
        print(f"Error in undervalued command: {e}")

def main():
    # Get token from environment variable
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise ValueError("Discord token not found in environment variables")
    
    # Run the bot
    bot.run(token)

if __name__ == "__main__":
    main()




