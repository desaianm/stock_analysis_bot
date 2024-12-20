import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from crewai.tasks.task_output import TaskOutput
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool
)
from tools import (
    StockPriceDataTool,
    RealTimeQuoteTool,
    OptionsChainTool,
    AnalystRecommendationsTool,
    StockNewsTool,
    CompanyInfoTool,
    TavilySearchTool
)

ny_timezone = pytz.timezone('America/New_York')
class PortfolioHolding(BaseModel):
    symbol: str
    shares: float
    total_value: float

class PortfolioData(BaseModel):
    holdings: List[PortfolioHolding]
    total_portfolio_value: float
    last_updated: datetime

class PortfolioAnalysisCrew:
    def __init__(self):
        self.setup_agents()
        self.setup_tools()

    def setup_tools(self):
        """Initialize tools for portfolio analysis"""
        self.stock_price_tool = StockPriceDataTool()
        self.real_time_quote_tool = RealTimeQuoteTool()
        self.options_chain_tool = OptionsChainTool()
        self.analyst_recommendations_tool = AnalystRecommendationsTool()
        self.stock_news_tool = StockNewsTool()
        self.company_info_tool = CompanyInfoTool()
        self.tavily_search_tool = TavilySearchTool()
        self.market_research_tool = SerperDevTool()

    def setup_agents(self):
        """Initialize specialized agents for portfolio analysis"""
        
        # Portfolio Analysis Agent
        self.portfolio_analyst = Agent(
            role="Portfolio Analysis Specialist",
            goal="Analyze current portfolio composition and performance",
            backstory="""You are an experienced portfolio analyst with expertise in 
            evaluating portfolio composition, risk metrics, and suggesting rebalancing 
            strategies. Your analysis considers both individual positions and overall 
            portfolio health.""",
            tools=[
                StockPriceDataTool(),
                RealTimeQuoteTool(),
                CompanyInfoTool()
            ]
        )

        # Risk Management Agent
        self.risk_analyst = Agent(
            role="Risk Management Specialist",
            goal="Evaluate portfolio risks and suggest hedging strategies",
            backstory="""You are a risk management expert specializing in portfolio 
            protection strategies. You analyze various risk metrics and recommend 
            appropriate hedging tactics to protect portfolio value.""",
            tools=[
                OptionsChainTool(),
                StockPriceDataTool(),
                AnalystRecommendationsTool()
            ]
        )

        # Market Research Agent
        self.market_researcher = Agent(
            role="Market Research Analyst",
            goal="Research market conditions and stock-specific opportunities",
            backstory="""You are a market research analyst with deep knowledge of 
            various sectors and market trends. You provide insights on market 
            conditions affecting the portfolio and identify opportunities.""",
            tools=[
                TavilySearchTool(),
                StockNewsTool(),
                SerperDevTool()
            ]
        )

        # Strategy Advisor Agent
        self.strategy_advisor = Agent(
            role="Investment Strategy Advisor",
            goal="Develop comprehensive investment strategies and recommendations",
            backstory="""You are a seasoned investment strategist who synthesizes 
            various analyses to create actionable investment recommendations. You 
            excel at balancing risk and reward while considering client objectives.""",
            tools=[
                CompanyInfoTool(),
                AnalystRecommendationsTool(),
                TavilySearchTool()
            ]
        )

    def create_tasks(self, portfolio_data: PortfolioData) -> List[Task]:
        """Create analysis tasks based on portfolio data"""
        
        # Portfolio Analysis Task
        portfolio_analysis_task = Task(
            description=f"""
            Analyze the current portfolio composition and performance:
            Portfolio Data: {portfolio_data.dict()}
            Current Date and Time: {datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")}
            
            Provide:
            1. Position Analysis
               - Individual position metrics
               - Position weights
               - Performance attribution
            
            2. Portfolio Metrics
               - Diversification analysis
               - Sector exposure
               - Concentration risk
            
            3. Performance Analysis
               - Individual position returns
               - Overall portfolio return
               - Benchmark comparison

            4. Time Contextual Report
               - Current market conditions
               - Historical portfolio performance
               - Forward-looking analysis
               - Current Time Contextual Analysis
            
            Format output in markdown with clear sections and metrics.""",
            agent=self.portfolio_analyst,
            expected_output="A markdown format report"
        )

        # Risk Analysis Task
        risk_analysis_task = Task(
            description=f"""Evaluate portfolio risks and suggest hedging strategies:
            Portfolio Data: {portfolio_data.dict()}
            Current Date and Time: {datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")}
            
            Analyze:
            1. Risk Metrics
               - Beta analysis
               - Volatility measures
               - Correlation analysis
            
            2. Portfolio Vulnerabilities
               - Sector risks
               - Position-specific risks
               - Market condition risks
            
            3. Hedging Recommendations
               - Options strategies
               - Portfolio insurance
               - Position adjustments

            4. Time Contextual Report
               - Current market conditions
               - Historical portfolio performance
               - Forward-looking analysis
               - Current Time Contextual Analysis
            
            Format output in markdown with clear recommendations.""",
            agent=self.risk_analyst,
            context=[portfolio_analysis_task],
            expected_output="A markdown format report"
        )

        # Market Research Task
        market_research_task = Task(
            description=f"""Research market conditions affecting the portfolio:
            Portfolio Data: {portfolio_data.dict()}
            Current Date and Time: {datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")}
            Research:
            1. Market Environment
               - Sector trends
               - Economic indicators
               - Market sentiment
            
            2. Stock-Specific Analysis
               - Company developments
               - Industry dynamics
               - Competitive position
            
            3. Opportunity Identification
               - Growth prospects
               - Valuation opportunities
               - Risk factors

            4. Time Contextual Report
               - Current market conditions
               - Historical portfolio performance
               - Forward-looking analysis
               - Current Time Contextual Analysis
            
            Format output in markdown with actionable insights.""",
            agent=self.market_researcher,
            context=[portfolio_analysis_task, risk_analysis_task],
            expected_output="A markdown format report"
        )

        # Strategy Recommendation Task
        strategy_task = Task(
            description=f"""Develop comprehensive investment strategy recommendations:
            Portfolio Data: {portfolio_data.dict()}
            Current Date and Time: {datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")}
            Previous Analyses Context: Use insights from portfolio analysis, 
            risk assessment, and market research tasks.
            
            Provide:
            1. Portfolio Strategy
               - Position adjustments
               - Target allocations
               - Rebalancing recommendations
            
            2. Action Plan
               - Buy/Sell recommendations
               - Position sizing
               - Entry/exit strategies
            
            3. Implementation Guidelines
               - Timing considerations
               - Risk management tactics
               - Monitoring metrics

            4. Time Contextual Report
               - Current market conditions
               - Historical portfolio performance
               - Forward-looking analysis
               - Current Time Contextual Analysis
            
            Format output as a detailed markdown report.""",
            agent=self.strategy_advisor,
            context=[portfolio_analysis_task, risk_analysis_task, market_research_task],
            expected_output="A markdown format report"
        )

        return [
            portfolio_analysis_task,
            risk_analysis_task,
            market_research_task,
            strategy_task
        ]

    def analyze_portfolio(self, portfolio_data: Dict) -> str:
        """Execute portfolio analysis and generate recommendations"""
        
        # Convert dictionary to PortfolioData model
        if isinstance(portfolio_data, dict):
            portfolio_data = PortfolioData(**portfolio_data)

        # Create analysis crew
        crew = Crew(
            agents=[
                self.portfolio_analyst,
                self.risk_analyst,
                self.market_researcher,
                self.strategy_advisor
            ],
            tasks=self.create_tasks(portfolio_data),
            process=Process.sequential,
            verbose=True,
            memory=True
        )

        # Execute analysis
        result = crew.kickoff()

        # Save the final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"portfolio_analysis_{timestamp}.md"
        
        with open(report_filename, "w") as f:
            f.write(result.raw)
            
        return result.raw

# Usage Example
if __name__ == "__main__":
    # Sample portfolio data
    with open("portfolio.json", "r") as f:
        sample_portfolio = json.load(f)

    # Create and run portfolio analysis
    analyzer = PortfolioAnalysisCrew()
    analysis_report = analyzer.analyze_portfolio(sample_portfolio)

    # Clean up markdown code block markers if present
    analysis_report = analysis_report.strip()
    if analysis_report.startswith("```markdown"):
        analysis_report = analysis_report[12:]  # Remove ```markdown prefix
    if analysis_report.startswith("```"):
        analysis_report = analysis_report[3:]  # Remove ``` prefix
    if analysis_report.endswith("```"):
        analysis_report = analysis_report[:-3]  # Remove ``` suffix
    analysis_report = analysis_report.strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/portfolio_analysis_{timestamp}.md", "w") as f:
        f.write(analysis_report)
