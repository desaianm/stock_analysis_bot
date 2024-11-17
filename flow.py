from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, router, start
from crewai import LLM, Agent, Task, Crew
from crewai_tools import (
    SerperDevTool, 
    WebsiteSearchTool
)
from tools import (
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
    TavilySearchTool
)

from agents import FinancialResearchAgents
from tasks import MarkdownReportCreationTasks

class StockData(BaseModel):
    ticker: str = ""
    company_name: str = ""
    current_price: float = 0.0
    historical_data: Dict = {}
    sentiment_data: Dict = {}
    financial_metrics: Dict = {}
    technical_analysis: Dict = {}
    prediction: Dict = {}
    analysis_complete: bool = False
    retry_count: int = 0

class EnhancedStockAnalysisFlow(Flow[StockData]):
    def __init__(self):
        super().__init__()
        # Initialize basic tools
        self.search_tool = SerperDevTool()
        self.web_tool = WebsiteSearchTool()
        self.tavily_tool = TavilySearchTool()
        
        # Initialize financial tools
        self.extraction_tool = ExtractionTool()
        self.data_fetching_tool = DataFetchingTool()
        self.charting_tool = ChartingTool()
        self.financial_report_tool = FinancialReportTool()
        self.markdown_tool = MarkdownTool()
        self.chat_analysis_tool = ChatAnalysisTool()
        
        # Initialize market data tools
        self.stock_price_tool = StockPriceDataTool()
        self.real_time_quote_tool = RealTimeQuoteTool()
        self.options_chain_tool = OptionsChainTool()
        self.analyst_recommendations_tool = AnalystRecommendationsTool()
        self.stock_news_tool = StockNewsTool()
        self.company_info_tool = CompanyInfoTool()

        self.claude_llm = LLM(
            model="openai/gpt-4o",
            temperature=1,
            max_tokens=8000,
        )
        
        # Initialize specialized agents
        self.researcher = Agent(
            role='Financial Data Researcher',
            goal='Gather comprehensive financial and market data',
            backstory="""Expert financial researcher with years of experience in gathering
                      and analyzing market data. Access to various financial databases
                      and news sources.""",
            tools=[
                self.search_tool,
                self.web_tool,
                self.stock_price_tool,
                self.real_time_quote_tool,
                self.company_info_tool
            ],
            verbose=True
        )
        
        self.technical_analyst = Agent(
            role='Technical Analysis Specialist',
            goal='Perform technical analysis and identify market patterns',
            backstory="""Expert in technical analysis with deep understanding of market
                      patterns and indicators. Specializes in price action analysis
                      and chart patterns.""",
            tools=[
                self.stock_price_tool,
                self.charting_tool,
                self.options_chain_tool
            ],
            llm=self.claude_llm,
            verbose=True,
            max_retry_limit=5
        )
        
        self.fundamental_analyst = Agent(
            role='Fundamental Analyst',
            goal='Analyze company fundamentals and financial metrics',
            backstory="""Experienced financial analyst specializing in fundamental
                      analysis. Expert in evaluating financial statements and
                      company valuations.""",
            tools=[
                self.data_fetching_tool,
                self.financial_report_tool,
                self.chat_analysis_tool
            ],
            llm=self.claude_llm,
            verbose=True,
            max_retry_limit=5
        )
        
        self.market_sentiment_analyst = Agent(
            role='Market Sentiment Analyst',
            goal='Analyze market sentiment and news impact',
            backstory="""Specialist in analyzing market sentiment and news impact.
                      Expert in natural language processing and sentiment analysis.""",
            tools=[
                self.tavily_tool,
                self.stock_news_tool,
                self.analyst_recommendations_tool
            ],
            llm=self.claude_llm,
            verbose=True,
            max_retry_limit=5
        )
        
        self.report_writer = Agent(
            role='Investment Report Writer',
            goal='Create comprehensive investment reports',
            backstory="""Professional financial writer specializing in creating
                      detailed investment reports. Expert in synthesizing complex
                      financial information into clear recommendations.""",
            tools=[self.markdown_tool, self.chat_analysis_tool],
            llm=self.claude_llm,
            verbose=True,
            max_retry_limit=5
        )

    @start()
    def gather_basic_data(self):
        """Gather initial company and market data"""
        research_task = Task(
            description=f"""Research and gather the following information for {self.state.ticker}:
                        1. Current stock price and basic company information
                        2. Historical stock prices for the past 5 years
                        3. Key financial metrics and ratios
                        4. Recent company news and developments""",
            agent=self.researcher,
            expected_output="json in given schema",
            output_pydantic=StockData
        )
        crew = Crew(agents=[self.researcher], tasks=[research_task])
        result = crew.kickoff()
        
        # Parse and structure the results
        self.state.historical_data = {
            "data": result.raw,
            "timestamp": datetime.now().isoformat()
        }
        return "basic_research_complete"

    @listen(gather_basic_data)
    def perform_technical_analysis(self):
        """Conduct technical analysis"""
        technical_task = Task(
            description=f"""Perform technical analysis for {self.state.ticker}:
                        1. Analyze price patterns and trends
                        2. Calculate key technical indicators
                        3. Evaluate options market data
                        4. Generate technical charts
                        
                        Historical Data:
                        {self.state.historical_data['data']}""",
            agent=self.technical_analyst,
            expected_output="A Report in markdown format"
        )
        crew = Crew(agents=[self.technical_analyst], tasks=[technical_task])
        result = crew.kickoff()
        
        self.state.technical_analysis = {
            "analysis": result.raw,
            "timestamp": datetime.now().isoformat()
        }
        return "technical_analysis_complete"

    @listen(gather_basic_data)
    def analyze_fundamentals(self):
        """Perform fundamental analysis"""
        fundamental_task = Task(
            description=f"""Analyze fundamental data for {self.state.ticker}:
                        1. Evaluate financial statements
                        2. Calculate key financial ratios
                        3. Assess company valuation
                        4. Generate financial charts""",
            agent=self.fundamental_analyst,
            expected_output="A Report in markdown format"
        )
        crew = Crew(agents=[self.fundamental_analyst], tasks=[fundamental_task])
        result = crew.kickoff()
        
        self.state.financial_metrics = {
            "analysis": result.raw,
            "timestamp": datetime.now().isoformat()
        }
        return "fundamental_analysis_complete"

    @listen(gather_basic_data)
    def analyze_sentiment(self):
        """Analyze market sentiment"""
        sentiment_task = Task(
            description=f"""Analyze market sentiment for {self.state.ticker}:
                        1. Review news and social media sentiment
                        2. Analyze analyst recommendations
                        3. Evaluate institutional investor positions
                        4. Assess market perception""",
            agent=self.market_sentiment_analyst,
            expected_output="A Report in markdown format"
        )
        crew = Crew(agents=[self.market_sentiment_analyst], tasks=[sentiment_task])
        result = crew.kickoff()
        
        self.state.sentiment_data = {
            "analysis": result.raw,
            "timestamp": datetime.now().isoformat()
        }
        return "sentiment_analysis_complete"

    @listen(perform_technical_analysis)
    @listen(analyze_fundamentals)
    @listen(analyze_sentiment)
    def compile_report(self):
        """Generate final investment report"""
        if not all([
            self.state.technical_analysis,
            self.state.financial_metrics,
            self.state.sentiment_data
        ]):
            return "wait_for_analysis"
            
        report_task = Task(
            description=f"""
            Create a comprehensive investment report for {self.state.ticker} with the following detailed analysis:

                        1. Executive Summary
                           - Key findings and overall investment thesis
                           - Clear investment recommendation with price targets
                           - Risk rating and investment horizon
                        
                        2. Technical Analysis Deep Dive
                           - Price action and trend analysis
                           - Support/resistance levels and breakout points
                           - Volume analysis and money flow indicators
                           - Moving averages and momentum indicators
                           - Chart patterns and technical signals
                           - Relative strength vs market/sector
                        
                        3. Fundamental Analysis Breakdown
                           - Business model and competitive advantages
                           - Financial statement analysis (Income, Balance Sheet, Cash Flow)
                           - Key performance metrics and ratios
                           - Valuation analysis (DCF, multiples, etc)
                           - Industry comparison and market position
                           - Management assessment
                        
                        4. Market Sentiment Analysis
                           - News flow and media coverage
                           - Social media sentiment trends
                           - Analyst coverage and recommendations
                           - Institutional ownership changes
                           - Options market signals
                           - Short interest and borrowing costs
                        
                        5. Risk Assessment
                           - Company-specific risks
                           - Industry and competitive risks
                           - Macro and market risks
                           - Regulatory and compliance risks
                           - ESG considerations
                        
                        6. Investment Outlook
                           - Growth catalysts and opportunities
                           - Potential headwinds
                           - Price targets (bull/base/bear cases)
                           - Position sizing recommendations
                           - Entry/exit strategies
                           - Monitoring metrics and review points

                        Technical Analysis Data:
                        {self.state.technical_analysis['analysis']}
                        
                        Fundamental Analysis Data:
                        {self.state.financial_metrics['analysis']}
                        
                        Sentiment Analysis Data:
                        {self.state.sentiment_data['analysis']}
            """,
            agent=self.report_writer,
            expected_output="A Report in markdown format"
        )
        crew = Crew(agents=[self.report_writer], tasks=[report_task])
        result = crew.kickoff()
        
        # Save the report
        filename = f"stock_analysis_{self.state.ticker}_{datetime.now().strftime('%Y%m%d')}.md"
        with open(filename, "w") as f:
            f.write(result.raw)
        
        self.state.analysis_complete = True
        return "analysis_complete"
    
    

# def main():
#     # Initialize the flow
#     stock_flow = EnhancedStockAnalysisFlow()
    
#     # Start the flow with initial state
#     result = stock_flow.kickoff(inputs={
#         "ticker": "MRVL"  # Replace with desired stock ticker
#     })
    
#     print("Flow Result:", result)

# if __name__ == "__main__":
#     main()