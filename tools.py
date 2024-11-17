import base64
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from quickfs import QuickFS
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Type
from langchain_community.tools import TavilySearchResults
import random
import matplotlib.pyplot as plt
from crewai.tools import BaseTool
from langchain_core.messages import HumanMessage
from datetime import datetime
from crewai.tools import tool
load_dotenv()

tavily_search = TavilySearchResults(max_results=10,
    search_depth="advanced",
    include_raw_content=True,)
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearch."""
    query: str = Field(..., description="The query to search the web with")

class TavilySearchTool(BaseTool):
    name: str = "Tavily Search Tool"
    description: str = """Useful to search the web for information"""
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> str:
        return tavily_search.run(query)

class ExtractionToolInput(BaseModel):
    """Input schema for ExtractionTool."""
    data: str = Field(..., description="Input string containing symbol and metrics (e.g., 'AAPL revenue net_income eps')")

class ExtractionTool(BaseTool):
    name: str = "Extract symbol and metrics"
    description: str = """Useful to extract the relevant information from the input string. Parses string and extracts the symbol and all of the relevant metrics requested.
        Example:
        - Input: "AAPL revenue net_income eps"
        - Output: [
            {"symbol": "AAPL", "metric": "revenue"},
            {"symbol": "AAPL", "metric": "net_income"},
            {"symbol": "AAPL", "metric": "eps"}
        ]
        """
    args_schema: Type[BaseModel] = ExtractionToolInput

    def _run(self, data: str) -> List[dict]:
        words = data.split()
        symbol = words[0]
        result_list = [{"symbol": symbol, "metric": metric} for metric in words[1:]]
        return result_list

class DataFetchingToolInput(BaseModel):
    """Input schema for DataFetchingTool."""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    metric: str = Field(..., description="Financial metric to retrieve (e.g., 'revenue')")

class DataFetchingTool(BaseTool):
    name: str = "Retrieve metric data from QuickFS API"
    description: str = """Useful to retrieve data from the QuickFS API based on the given symbol and metric.
        :param symbol: str, only one symbol
        :param metric: str, only one metric
        :return value: list, A list containing the data points retrieved
        Return value example: [...data_points]

        Example:
        - Input: symbol="AAPL", metric="revenue"
        - Output: [265595000000, 274515000000, 365817000000, 394328000000, 383285000000]
        """
    args_schema: Type[BaseModel] = DataFetchingToolInput

    def _run(self, symbol: str, metric: str) -> List:
        api_key = os.environ.get("QUICKFS_API_KEY")
        client = QuickFS(api_key)
        res = client.get_data_range(symbol=f'{symbol}:US', metric=metric, period='FY-9:FY')
        return res

class CreateChartInput(BaseModel):
    metric: str
    data: List[float]

class CreateChartOutput(BaseModel):
    file_path: str

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive)
import matplotlib.pyplot as plt
class ChartingToolInput(BaseModel):
    """Input schema for ChartingTool."""
    metric_name: str = Field(..., description="Name of the metric to be visualized")
    data: List[float] = Field(..., description="List of numerical data points")

class ChartingTool(BaseTool):
    name: str = "Create a chart of the data"
    description: str = """Creates a bar chart graphic based on the provided metric and data.

        Parameters:
        - metric_name (str): The name of the metric to be visualized on the chart.
        - data (List[float]): A list of numerical data points representing the metric over time.

        Returns:
        - file_path (str): The file path to the saved chart image.
        
        Example:
        - Input: metric_name='revenue', data=[100000000, 150000000, 120000000, 200000000, 180000000]
        - Output: CreateChartOutput(file_path='./plots/revenue_chart.png')
        """
    args_schema: Type[BaseModel] = ChartingToolInput

    def _run(self, metric_name: str, data: List) -> CreateChartOutput:
        years = list(range(len(data)))
        bar_color = f'#{random.randint(0, 0xFFFFFF):06x}'
        
        plt.figure(figsize=(10, 6))  # Create a new figure
        plt.bar(years, data, color=bar_color)
        plt.xlabel('Years')
        plt.title(metric_name)
        
        file_path = f"plots/{metric_name.replace(' ', '_')}_chart.png"
        plt.savefig(file_path, format='png')
        plt.close()  # Close the figure to free up memory
        
        return CreateChartOutput(file_path=file_path)


class MarkdownToolInput(BaseModel):
    """Input schema for MarkdownTool."""
    text: str = Field(..., description="The markdown text to write to the file")

class MarkdownTool(BaseTool):
    name: str = "Write text to markdown file"
    description: str = """Useful to write markdown text in a *.md file.
           The input to this tool should be a string representing what should used to create markdown syntax. Takes the location of the file as a string and creates the correct syntax thats compatible with an .md file eg report.md

           Example:
           - Input: "# Financial Report\n\n## Revenue\n\n![Revenue Chart](revenue_chart.png)\n\nThe revenue has shown steady growth over the past 5 years."
           - Output: "File written to report.md."
           
           :param text: str, the string to write to the file
           """
    args_schema: Type[BaseModel] = MarkdownToolInput

    def _run(self, text: str) -> str:
        try:
            markdown_file_path = r'report.md'
            with open(markdown_file_path, 'w') as file:
                file.write(text)
            return f"File written to {markdown_file_path}."
        except Exception:
            return "Something has gone wrong writing images to markdown file."

class FinancialReportToolInput(BaseModel):
    """Input schema for FinancialReportTool."""
    symbol: str = Field(..., description="The stock symbol to create a financial report for")

class FinancialReportTool(BaseTool):
    name: str = "Create a financial report"
    description: str = """
    Useful to create a financial report from symbol.

    :param symbol: str, the symbol to create a financial report for
    :return data: dict, a dictionary containing the financial data

    Example:
    - Input: symbol="AAPL"
    - Output: {
        "revenue": [265595000000, 274515000000, 365817000000, 394328000000, 383285000000],
        "net_income": [55256000000, 57411000000, 94680000000, 99803000000, 96995000000],
        "eps": [3.28, 3.31, 5.61, 6.11, 6.15]
    }
    """
    args_schema: Type[BaseModel] = FinancialReportToolInput

    def _run(self, symbol: str) -> str:
        client = QuickFS(os.getenv("QUICKFS_API_KEY"))
        data = client.get_data_full(symbol)
        return data
    
class ChatAnalysisToolInput(BaseModel):
    """Input schema for ChatAnalysisTool."""
    data: str = Field(..., description="The data to analyze and create a report for")

class ChatAnalysisTool(BaseTool):
    name: str = "Chat Analysis Tool"
    description: str = """
    Useful to analyze the charts of a symbol and create a report.

    :param data: str, the data to analyze
    :return report: str, the report in markdown syntax

    Example:
    - Input: "Analyze the revenue and net income charts for AAPL"
    - Output: "# Financial Analysis Report for AAPL

    ## Revenue Analysis
    ![Revenue Chart](revenue_chart.png)

    Apple's revenue has shown consistent growth over the past 5 years, with a notable increase in FY2021...

    ## Net Income Analysis
    ![Net Income Chart](net_income_chart.png)

    The company's net income has followed a similar trend to its revenue, demonstrating strong profitability..."
    """
    args_schema: Type[BaseModel] = ChatAnalysisToolInput

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _run(self, data: str) -> str:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002")

        # Get all image files in the plots directory
        plot_files = [f for f in os.listdir('plots') if f.endswith('.png')]
        
        # Encode each image to base64
        encoded_images = []
        for plot_file in plot_files:
            image_path = os.path.join('plots', plot_file)
            encoded_image = self._encode_image(image_path)
            encoded_images.append({
                "file_name": plot_file,
                "base64_image": encoded_image
            })
        
        # Prepare the prompt for the model
        prompt = f"""Analyze the following financial charts and create a comprehensive report:


        Charts: {', '.join([img['file_name'] for img in encoded_images])}

        Please provide a detailed analysis of the financial performance based on these charts. 
        Include insights on trends, potential risks, and opportunities. 
        Format the report in markdown syntax, including appropriate headers and sections.
        """

        # Call the model with the prompt and images
        response = model.invoke(
            [
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": prompt
                    },
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img['base64_image']}",
                            "detail": "high"
                        }
                    } for img in encoded_images]
                ])
            ]
        )

        return response.content


from crewai_tools import BaseTool
from typing import List
import yfinance as yf
import os

class StockPriceDataToolInput(BaseModel):
    """Input schema for StockPriceDataTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve data for")
    period: str = Field(..., description="Time period for data retrieval (e.g., '5y', '6mo')")

class StockPriceDataTool(BaseTool):
    name: str = "Retrieve historical stock price data"
    description: str = """Useful to retrieve historical stock price data from Yahoo Finance based on the given symbol and time period.

        :param symbol: str, only one symbol
        :param period: str, time period for data retrieval (e.g., '5y' for 5 years, '6mo' for 6 months)
        :return value: list, A list containing the closing prices
        Return value example: [...closing_prices]

        Example:
        - Input: symbol="AAPL", period="5y"
        - Output: [145.85, 147.92, 149.26, 150.65, 148.97, ...]
        """
    args_schema: Type[BaseModel] = StockPriceDataToolInput

    def _run(self, symbol: str, period: str) -> List:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data['Close'].tolist()

class RealTimeQuoteToolInput(BaseModel):
    """Input schema for RealTimeQuoteTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve real-time quote for")

class RealTimeQuoteTool(BaseTool):
    name: str = "Retrieve real-time stock quote"
    description: str = """Useful to retrieve real-time stock quote from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing current price, volume, and market cap
        Return value example: [current_price, volume, market_cap]

        Example:
        - Input: symbol="AAPL"
        - Output: [150.65, 75000000, 2500000000000]
        """
    args_schema: Type[BaseModel] = RealTimeQuoteToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        info = stock.info
        return [info.get('currentPrice'), info.get('volume'), info.get('marketCap')]

class OptionsChainToolInput(BaseModel):
    """Input schema for OptionsChainTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve options data for")
    expiration_date: str = Field(..., description="Expiration date for options (format: 'YYYY-MM-DD')")

class OptionsChainTool(BaseTool):
    name: str = "Retrieve options chain data"
    description: str = """Useful to retrieve options chain data from Yahoo Finance based on the given symbol and expiration date.

        :param symbol: str, only one symbol
        :param expiration_date: str, expiration date for options (format: 'YYYY-MM-DD')
        :return value: list, A list containing calls and puts data, and the actual expiration date used
        Return value example: [[...calls_data], [...puts_data], 'YYYY-MM-DD']

        Example:
        - Input: symbol="AAPL", expiration_date="2023-07-21"
        - Output: [[{'strike': 140, 'lastPrice': 10.5, ...}], [{'strike': 140, 'lastPrice': 0.5, ...}], '2023-07-21']
        """
    args_schema: Type[BaseModel] = OptionsChainToolInput

    def _run(self, symbol: str, expiration_date: str) -> List:
        stock = yf.Ticker(symbol)
        
        # Get available expiration dates
        expirations = stock.options

        if not expirations:
            return [[], [], None]  # No options data available

        # Convert input expiration_date to datetime
        target_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()

        # Find the closest available expiration date
        closest_date = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - target_date))

        # Fetch options data for the closest available date
        options = stock.option_chain(date=closest_date)
        
        return [options.calls.to_dict('records'), options.puts.to_dict('records'), closest_date]

class AnalystRecommendationsToolInput(BaseModel):
    """Input schema for AnalystRecommendationsTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve analyst recommendations for")

class AnalystRecommendationsTool(BaseTool):
    name: str = "Retrieve analyst recommendations"
    description: str = """Useful to retrieve analyst recommendations from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing analyst recommendations
        Return value example: [...recommendations]

        Example:
        - Input: symbol="AAPL"
        - Output: [{'firm': 'Barclays', 'toGrade': 'Overweight', 'fromGrade': 'Equal Weight', 'action': 'up'}, ...]
        """
    args_schema: Type[BaseModel] = AnalystRecommendationsToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations.to_dict('records')

class StockNewsToolInput(BaseModel):
    """Input schema for StockNewsTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve news for")

class StockNewsTool(BaseTool):
    name: str = "Retrieve recent stock news"
    description: str = """Useful to retrieve recent news headlines from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing recent news headlines
        Return value example: [...headlines]

        Example:
        - Input: symbol="AAPL"
        - Output: ["Apple Launches New iPhone", "Apple's Q2 Earnings Beat Expectations", ...]
        """
    args_schema: Type[BaseModel] = StockNewsToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        news = stock.news
        return [item['title'] for item in news[:5]]  # Get the 5 most recent headlines

class CompanyInfoToolInput(BaseModel):
    """Input schema for CompanyInfoTool."""
    symbol: str = Field(..., description="Stock symbol to retrieve company information for")

class CompanyInfoTool(BaseTool):
    name: str = "Retrieve company information"
    description: str = """Useful to retrieve detailed company information from Yahoo Finance based on the given symbol.

        :param symbol: str, only one symbol
        :return value: list, A list containing key company information
        Return value example: [long_name, sector, industry, market_cap, forward_pe, dividend_yield]

        Example:
        - Input: symbol="AAPL"
        - Output: ["Apple Inc.", "Technology", "Consumer Electronics", 2500000000000, 25.5, 0.65]
        """
    args_schema: Type[BaseModel] = CompanyInfoToolInput

    def _run(self, symbol: str) -> List:
        stock = yf.Ticker(symbol)
        info = stock.info
        return [
            info.get('longName'),
            info.get('sector'),
            info.get('industry'),
            info.get('marketCap'),
            info.get('forwardPE'),
            info.get('dividendYield')
        ]

# class CompanySearchToolInput(BaseModel):
#     """Input schema for CompanySearchTool."""
#     company_name: str = Field(..., description="Company name to search for")

# class CompanySearchTool(BaseTool):
#     name: str = "Search for a company by name"
#     description: str = """Useful to search for a company by name and retrieve its symbol."""
#     args_schema: Type[BaseModel] = CompanySearchToolInput

#     def _run(self, company_name: str) -> str:
#         return yf.Ticker.search(company_name)