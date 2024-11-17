import asyncio
from stock_recomendation_flow import Top20StocksFlow
from stock_under import UndervaluedAnalysisFlow
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv


load_dotenv()

# Initialize flows
under_flow = UndervaluedAnalysisFlow()
stock_recomendation_flow = Top20StocksFlow()

# Set up bot with command prefix and intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='top20')
async def get_top_20(ctx):
    """Get top 20 stock recommendations"""
    try:
        await ctx.send("Analyzing top 20 stocks... Please wait.")
        results = await stock_recomendation_flow.analyze()
        await ctx.send(f"Top 20 Stock Recommendations:\n{results}")
    except Exception as e:
        await ctx.send("An error occurred while processing your request.")
        print(f"Error in top20 command: {e}")

@bot.command(name='undervalued')
async def get_undervalued(ctx):
    """Get undervalued stock analysis"""
    try:
        await ctx.send("Analyzing undervalued stocks... Please wait.")
        results = await under_flow.analyze()
        await ctx.send(f"Undervalued Stocks Analysis:\n{results}")
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




