import asyncio
import json
import traceback

from crewai import Crew
from stockbot.agents.financial import FinancialResearchAgents
from stockbot.portfolio.analysis import PortfolioAnalysisCrew
from stockbot.flows.recommendations import Top20StocksFlow, InvestmentPreferences
from stockbot.flows.undervalued import UndervaluedAnalysisFlow, ValueScreeningPreferences
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from discord import app_commands
from stockbot.flows.single_stock import EnhancedStockAnalysisFlow
from stockbot.tasks.workflows import MarkdownReportCreationTasks, update_portfolio
from stockbot.tools.data import CompanyInfoTool



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


async def company_lookup(company_name):
        agents = FinancialResearchAgents()
        tasks = MarkdownReportCreationTasks()
        company_lookup_agent = agents.company_lookup_agent()
        company_lookup_task = await tasks.company_lookup_task(company_lookup_agent, company_name)
        crew = Crew(
            agents=[company_lookup_agent], 
            tasks=[company_lookup_task],
            verbose=True,
            memory=True
        )
        result = await crew.kickoff_async()
        company_data = result.to_dict()
        return company_data

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.tree.command(name="top20", description="Get top 20 stock recommendations")
async def get_top_20(interaction: discord.Interaction):
    """Get top 20 stock recommendations"""
    try:
        await interaction.response.defer()
        await interaction.followup.send("Analyzing top 20 stocks... Please wait.")
        results = await stock_recomendation_flow.execute_portfolio_construction()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"top20_report_{timestamp}.md"
        
        with open(report_file, "w", encoding='utf-8') as f:
            f.write("# Top 20 Stock Recommendations Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results)
        
        await interaction.followup.send(
            "Analysis complete! Here's your report:",
            file=discord.File(report_file)
        )
        
        # Remove the file after sending
        report_file.unlink()
        
    except Exception as e:
        await interaction.followup.send("An error occurred while processing your request.")
        print(f"Error in top20 command: {e}")


@bot.tree.command(name="portfolio", description="Get portfolio analysis")
async def get_portfolio(interaction: discord.Interaction):
    """Get portfolio analysis"""
    await interaction.response.defer()
    await interaction.followup.send("Analyzing portfolio... Please wait.")
    try:
        # Load portfolio data
        with open("portfolio.json", "r") as f:
            portfolio_data = json.load(f)

        # Create and run portfolio analysis
        analyzer = PortfolioAnalysisCrew()
        analysis_report = analyzer.analyze_portfolio(portfolio_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"portfolio_analysis_{timestamp}.md"
        
        with open(report_file, "w", encoding='utf-8') as f:
            f.write(analysis_report)
            
        await interaction.followup.send(
            "Analysis complete! Here's your portfolio report:",
            file=discord.File(report_file)
        )
        
        # Remove the file after sending
        report_file.unlink()
        
    except Exception as e:
        await interaction.followup.send("An error occurred while analyzing the portfolio.")
        print(f"Error in portfolio command: {e}")

@bot.tree.command(name="undervalued", description="Get undervalued stock analysis")
async def get_undervalued(interaction: discord.Interaction):
    """Get undervalued stock analysis"""
    try:
        await interaction.response.defer()
        await interaction.followup.send("Analyzing undervalued stocks... Please wait.")
        results = await under_flow.execute_undervalued_analysis()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"undervalued_report_{timestamp}.md"
        
        with open(report_file, "w", encoding='utf-8') as f:
            f.write("# Undervalued Stocks Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(results)
        
        await interaction.followup.send(
            "Analysis complete! Here's your report:",
            file=discord.File(report_file)
        )
        
        # Remove the file after sending
        report_file.unlink()
        
    except Exception as e:
        await interaction.followup.send("An error occurred while processing your request.")
        print(f"Error in undervalued command: {e}")

@bot.tree.command(name="help", description="Show available commands")
async def help_command(interaction: discord.Interaction):
    """Show help for the bot"""
    help_text = "Available commands:\n" \
                "/top20 - Get top 20 stock recommendations\n" \
                "/undervalued - Get undervalued stock analysis"
    await interaction.response.send_message(help_text)

@bot.tree.command(name="analyze", description="Analyze a specific company")
async def analyze_company(interaction: discord.Interaction, company_name: str):
    """Analyze a specific company by name"""
    try:
        await interaction.response.defer()
        await interaction.followup.send(f"Searching for company '{company_name}' and analyzing... Please wait.")

        
        
        # Try to find the ticker symbol
        try:
            company_data = await company_lookup(company_name)
            if not company_data or 'ticker' not in company_data:
                await interaction.followup.send(f"Could not find ticker symbol for company: {company_name}")
                return
            ticker = company_data['ticker']
        except Exception as e:
            await interaction.followup.send(f"Error finding company symbol: {str(e)}")
            return

        # Initialize and run the enhanced stock analysis flow
        stock_flow = EnhancedStockAnalysisFlow()
        result = await stock_flow.kickoff_async(inputs={"ticker": ticker})

        # Get the generated report filename
        report_file = f"stock_analysis_{ticker}_{datetime.now().strftime('%Y%m%d')}.md"
        
        if not os.path.exists(report_file):
            await interaction.followup.send("Analysis failed to generate report.")
            return

        await interaction.followup.send(
            f"Analysis complete for {company_name} ({ticker})! Here's your report:",
            file=discord.File(report_file)
        )
        
        # Remove the file after sending
        Path(report_file).unlink()

    except Exception as e:
        await interaction.followup.send("An error occurred while processing your request.")
        print(f"Error in analyze command: {str(e)}")
        traceback.print_exc()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    print(f"Received message: {message.content}")

    if message.content.startswith("/update_portfolio"):
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    message_content = await update_portfolio(attachment)

                    await message.channel.send(message_content)
                    break
            else:
                await message.channel.send("Please attach a valid image file (PNG, JPG, JPEG, or GIF).")
        else:
            await message.channel.send("Please attach an image with your portfolio update.")
    
    await bot.process_commands(message)

def main():
    #Get token from environment variable
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise ValueError("Discord token not found in environment variables")
    
    # Run the bot
    bot.run(token)

   


if __name__ == "__main__":
    main()



