from crewai import Agent, LLM
from textwrap import dedent
from dotenv import load_dotenv
import os

from stockbot.tools.data import (
    ExtractionTool,
    DataFetchingTool,
    ChartingTool,
    FinancialReportTool,
    MarkdownTool,
    ChatAnalysisTool,
    StockPriceDataTool,
    RealTimeQuoteTool,
    OptionsChainTool,
    AnalystRecommendationsTool,
    StockNewsTool,
    CompanyInfoTool,
    TavilySearchTool,
)

load_dotenv()

oai_api_key = os.getenv("OPENAI_API_KEY")

extraction_tool = ExtractionTool()
data_fetching_tool = DataFetchingTool()
financial_report_tool = FinancialReportTool()
markdown_tool = MarkdownTool()
charting_tool = ChartingTool()
chat_analysis_tool = ChatAnalysisTool()
stock_price_data_tool = StockPriceDataTool()
real_time_quote_tool = RealTimeQuoteTool()
options_chain_tool = OptionsChainTool()
analyst_recommendations_tool = AnalystRecommendationsTool()
stock_news_tool = StockNewsTool()
company_info_tool = CompanyInfoTool()
tavily_search_tool = TavilySearchTool()



class FinancialResearchAgents:
    def __init__(self):
        self.openai_llm = LLM(
            model="gpt-4.1",
            temperature=1,
            base_url="https://api.openai.com/v1",
            api_key=oai_api_key,
        )
        self.claude_llm = LLM(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=1,
        )
        
    def markdown_report_creator(self):
        return Agent(
            role="Markdown Report Creator",
            goal=dedent(f"""Retrieve accurate data of the metrics requested for a particular symbol."""),
            backstory=dedent(f"""Expert in creating markdown reports. The best at using tools to gather data from an API. You retrieve **EVERY** metric from QuickFS when asked and never miss a single one."""),
            tools=[
                extraction_tool, 
                data_fetching_tool,
                financial_report_tool,
                chat_analysis_tool
            ],
            verbose=True,
            llm=self.openai_llm,
        )
    
    def stock_analysis_agent(self):
        return Agent(
            role="Stock Analysis Agent",
            goal=dedent(f"""Fetch all the information using the given tools and create a detailed report on given stock symbol in markdown syntax."""),
            backstory=dedent(f"""Expert in analyzing stock data. You are known for receiving a list of data points and meticulously analyzing them to provide a summary of the key financial metrics to analyze the company's financial health."""),
            tools=[
                stock_price_data_tool,
                real_time_quote_tool,
                options_chain_tool,
                analyst_recommendations_tool,
                stock_news_tool,
                company_info_tool
            ],
            verbose=True,
            llm=self.openai_llm,
        )

    def chart_creator(self):
        return Agent(
            role="Chart Creator",
            goal=dedent(f"""Create a chart of the data provided using the tool."""),
            backstory=dedent(f"""Expert in creating charts. You are known for receiving a list of data points and meticulously creating an accurate chart. You must use the tool provided. """),
            tools=[
                charting_tool
            ] ,
            verbose=True,
            llm=self.openai_llm,
        )

    
    def markdown_writer(self):
        return Agent(
            role="Data Report Creator",
            goal=dedent(f"""Use *.png files in same directory to add the correct syntax a markdown file."""),
            backstory=dedent(f"""Expert in writing text inside a markdown file. You take a text input and write the contents to a markdown file in the same directory. You always add a new line after inserting into the markdown file. **YOU USE MARKDOWN SYNTAX AT ALL TIMES NO MATTER WHAT** YOU NEVER INSERT ANYTHING INTO THE report.md FILE THAT ISN'T MARKDOWN SYNTAX. """),
            tools=[markdown_tool,chat_analysis_tool],
            verbose=True,
            llm=self.openai_llm,
        )

    def company_research_agent(self):
        return Agent(
            role="Company Research Agent",
            goal=dedent(f"""Research a company and provide a detailed report on how company sentiment, news and other factors are affecting the stock price."""),
            backstory=dedent(f"""Expert in finding the sentiment of a company. You are known for providing a detailed report on how company sentiment, news and other factors are affecting the stock price."""),
            tools=[tavily_search_tool],
            verbose=True,
            llm=self.claude_llm,
        )

    def company_lookup_agent(self):
        return Agent(
            role="Company Lookup Agent",
            goal=dedent(f"""Search for a company by name and find all the information required for the company."""),
            backstory=dedent(f"""Expert in finding company information. You are known for providing a detailed company information or any information required for the company."""),
            tools=[tavily_search_tool],
            verbose=True,
            llm=self.openai_llm,
        )

     
