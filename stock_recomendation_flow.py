import asyncio
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import BaseTool
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
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
import pandas as pd
import json
from pathlib import Path

claude =  LLM(
                model="anthropic/claude-3-5-sonnet-20240620",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=8000,
                temperature=1
            )

class StockMetrics(BaseModel):
    ticker: str
    sector: str
    market_cap: float
    beta: float
    revenue_growth: float
    profit_margin: float
    pe_ratio: float
    price_to_book: float
    debt_to_equity: float
    current_ratio: float
    analyst_rating: float
    technical_score: float
    sentiment_score: float
    esg_score: float
    confidence_score: float

class PortfolioState(BaseModel):
    initial_candidates: List[str] = []
    screened_stocks: List[StockMetrics] = []
    sector_allocations: Dict[str, float] = {}
    correlation_matrix: Optional[List[List[float]]] = None
    stock_rankings: Dict[str, float] = {}
    tournament_results: List[Dict] = []
    final_portfolio: List[Dict] = []
    analysis_complete: bool = False

class InvestmentPreferences(BaseModel):
    strategy: str = Field(..., description="Investment strategy: 'growth', 'value', or 'balanced'")
    risk_tolerance: str = Field(..., description="Risk tolerance: 'conservative', 'moderate', or 'aggressive'")
    time_horizon: str = Field(..., description="Investment time horizon in years")
    min_market_cap: float = Field(..., description="Minimum market cap in billions USD")
    max_position_size: float = Field(..., description="Maximum position size as percentage")
    preferred_sectors: List[str] = Field(default=[], description="List of preferred sectors")
    excluded_sectors: List[str] = Field(default=[], description="List of sectors to exclude")
    esg_focus: bool = Field(default=False, description="Whether to focus on ESG factors")
    dividend_focus: bool = Field(default=False, description="Whether to focus on dividend-paying stocks")
    international_exposure: bool = Field(default=False, description="Whether to include international stocks")

class PortfolioConstraints(BaseModel):
    max_sector_weight: float
    min_position_size: float
    max_position_size: float
    min_liquidity: float
    max_volatility: float
    min_sharpe_ratio: float

class Top20StocksFlow:
    def __init__(self, preferences: InvestmentPreferences):
        self.preferences = preferences
        self.constraints = self.generate_constraints()
        self.state = PortfolioState()
        self.tournament_pool = 10000
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.setup_agents()
        self.setup_tasks()

    def generate_constraints(self) -> PortfolioConstraints:
        """Generate portfolio constraints based on user preferences"""
        # ... constraint generation code ...

    def setup_agents(self):
        """Initialize specialized agents for portfolio construction"""
        
        # Initial Screening Specialist
        self.screening_agent = Agent(
            role="Quantitative Screening Specialist",
            goal="Identify the most promising stock candidates based on quantitative metrics",
            backstory="""You are a quantitative analyst who has developed advanced screening models. 
            Your bonus is tied to the performance of stocks you identify. You receive 15% of the 
            tournament pool if your screened stocks outperform the market by 10%. Your screening 
            criteria must balance growth, value, and quality factors.""",
            tools=[
                StockPriceDataTool(),
                CompanyInfoTool(),
                FinancialReportTool()
            ]
        )

        # Sector Allocation Specialist
        self.sector_agent = Agent(
            role="Sector Strategy Expert",
            goal="Optimize sector allocation for maximum risk-adjusted returns",
            backstory="""You are a sector strategist known for identifying emerging sector trends. 
            You earn 15% of the tournament pool if your sector allocation strategy outperforms 
            sector-neutral benchmarks. Your analysis must consider economic cycles, policy impacts, 
            and sector correlations.""",
            tools=[
                SerperDevTool(),
                CompanyInfoTool(),
                DataFetchingTool()
            ]
        )

        # Stock Analysis Tournament Manager
        self.tournament_agent = Agent(
            role="Tournament Analysis Director",
            goal="Conduct head-to-head stock analysis tournaments",
            backstory="""You manage stock analysis tournaments where stocks compete based on multiple 
            criteria. Your bonus depends on the subsequent performance of the tournament winners. 
            You earn 20% of the pool if the selected stocks outperform their peers.""",
            tools=[
                StockPriceDataTool(),
                FinancialReportTool(),
                AnalystRecommendationsTool()
            ]
        )

        # Portfolio Optimization Specialist
        self.portfolio_agent = Agent(
            role="Portfolio Engineering Expert",
            goal="Optimize final portfolio construction for maximum efficiency",
            backstory="""You are a portfolio optimization expert using advanced mathematical models. 
            Your substantial bonus depends on the portfolio's Sharpe ratio and diversification 
            metrics. You earn 25% of the pool if your portfolio achieves top-quartile risk-adjusted 
            returns.""",
            tools=[
                StockPriceDataTool(),
                OptionsChainTool(),
                RealTimeQuoteTool()
            ]
        )

        # Risk Management Specialist
        self.risk_agent = Agent(
            role="Risk Engineering Expert",
            goal="Ensure portfolio risk optimization and stress testing",
            backstory="""You specialize in risk management and portfolio stress testing. Your bonus 
            is tied to downside protection and risk-adjusted returns. You earn 15% of the pool if 
            your risk management prevents significant drawdowns.""",
            tools=[
                OptionsChainTool(),
                StockPriceDataTool(),
                FinancialReportTool()
            ]
        )

        # Final Validation Expert
        self.validation_agent = Agent(
            role="Portfolio Validation Expert",
            goal="Validate final portfolio selection and provide detailed investment thesis",
            backstory="""You are the final authority on portfolio construction, validating all 
            analyses and ensuring alignment with investment objectives. Your bonus depends on the 
            overall portfolio performance and risk management effectiveness.""",
            tools=[
                CompanyInfoTool(),
                FinancialReportTool(),
                SerperDevTool()
            ]
        )

    def setup_tasks(self):
        """Initialize portfolio construction tasks"""
        
        # Initial Universe Screening Task
        self.screening_task = Task(
            description=f"""
            Current Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Screen the entire market to identify the top 20 candidates based on and take time into context:
            1. Quantitative Factors
               - Growth metrics (revenue, earnings, cash flow growth)
               - Value metrics (P/E, P/B, EV/EBITDA)
               - Quality metrics (ROE, debt/equity, profit margins)
               
            2. Technical Factors
               - Price momentum
               - Relative strength
               - Volume trends
               
            3. Market Factors
               - Market cap constraints
               - Liquidity requirements
               - Sector representation
            
            Your compensation depends on:
            - Quality of screened candidates
            - Diversity of factors considered
            - Forward-looking growth potential
            
            Initial Tournament Pool: ${self.tournament_pool:,}
            Your Potential Reward: 15% for market-beating performance""",
            agent=self.screening_agent,
            expected_output="A list of 50 stocks in given schema"
        )

        # Sector Analysis Task
        self.sector_task = Task(
            description=f"""
            Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Analyze sectors and recommend optimal allocations and take current time into context:
            1. Sector Analysis
               - Growth potential
               - Risk factors
               - Economic sensitivity
               
            2. Allocation Strategy
               - Target weights
               - Diversification requirements
               - Risk constraints
               
            Your bonus depends on:
            - Sector allocation efficiency
            - Risk-adjusted sector returns
            - Diversification effectiveness
            
            Tournament Pool: ${self.tournament_pool:,}
            Potential Reward: 15% for optimal sector strategy""",
            agent=self.sector_agent,
            expected_output="A report in markdown"
        )

        # Tournament Analysis Task
        self.tournament_task = Task(
            description=f"""
            Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Conduct head-to-head stock tournaments:
            1. Tournament Structure
               - Multiple rounds of competition
               - Various evaluation criteria
               - Scoring methodology
               
            2. Analysis Criteria
               - Financial strength
               - Growth potential
               - Risk factors
               - Market positioning
               - Latest Information (Based On Time)
               
            Your reward depends on:
            - Tournament effectiveness
            - Winner performance
            - Analysis depth
            
            Tournament Pool: ${self.tournament_pool:,}
            Potential Reward: 20% for successful tournament""",
            agent=self.tournament_agent,
            expected_output="A report in markdown"
        )

    async def execute_portfolio_construction(self):
        """Execute the full portfolio construction process"""
        
        # Phase 1: Initial Screening
        screening_crew = Crew(
            agents=[self.screening_agent],
            tasks=[self.screening_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        screening_result = await screening_crew.kickoff_async()
        self.state.initial_candidates = screening_result.raw
        
        # Phase 2: Sector Analysis
        sector_crew = Crew(
            agents=[self.sector_agent],
            tasks=[self.sector_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        sector_result = await sector_crew.kickoff_async()
        self.state.sector_allocations = sector_result.raw

        # Phase 3: Stock Tournaments
        tournament_crew = Crew(
            agents=[self.tournament_agent],
            tasks=[self.tournament_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        tournament_result = await tournament_crew.kickoff_async()
        self.state.tournament_results = tournament_result.raw
        
        # Phase 4: Portfolio Optimization
        optimization_task = Task(
            description=f"""Optimize final portfolio of 20 stocks based on:
            Tournament Results: {self.state.tournament_results}
            Sector Allocations: {self.state.sector_allocations}
            
            Requirements:
            1. Select exactly 20 stocks
            2. Maintain sector allocation constraints
            3. Optimize for risk-adjusted returns
            4. Ensure proper diversification
            
            Provide for each selected stock:
            1. Investment thesis
            2. Target position size
            3. Entry strategy
            4. Risk management guidelines""",
            agent=self.portfolio_agent,
            expected_output="A report in markdown"
        )
        
        optimization_crew = Crew(
            agents=[self.portfolio_agent],
            tasks=[optimization_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        optimization_result = await optimization_crew.kickoff_async()
        
        # Phase 5: Risk Analysis and Validation
        validation_task = Task(
            description=f"""Validate and finalize portfolio selection:
            Proposed Portfolio: {optimization_result.raw}
            
            Provide:
            1. Final portfolio composition
            2. Position sizes
            3. Risk analysis
            4. Investment strategy
            5. Monitoring guidelines
            
            Your validation must ensure:
            - Proper diversification
            - Risk management
            - Strategy alignment
            - Implementation feasibility""",
            agent=self.validation_agent,
            expected_output="Report in markdown format"
        )
        
        validation_crew = Crew(
            agents=[self.validation_agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        
        final_result = await validation_crew.kickoff_async()
        self.state.final_portfolio = final_result.raw
        self.state.analysis_complete = True
        
        final_result.raw = final_result.raw.strip()
        final_result.raw = final_result.raw[len("```markdown"):].strip()
        if final_result.raw.startswith("```"):
            final_result.raw = final_result.raw[len("```"):].strip()
        if final_result.raw.endswith("```"):
            final_result.raw = final_result.raw[:-3].strip()

        with open(self.output_dir / "final_portfolio.md", "w") as f:
            f.write(final_result.raw)
        return self.state.final_portfolio

async def main():
    """Execute the portfolio construction process with user preferences"""
    try:
        preferences = await get_user_preferences()
        
        print("\nInitializing portfolio construction with your preferences...")
        portfolio_flow = Top20StocksFlow(preferences)
        
        print("\nExecuting portfolio analysis...")
        final_portfolio = await portfolio_flow.execute_portfolio_construction()
        
        print(f"\nPortfolio construction completed successfully!")
        print(f"Results saved in: {portfolio_flow.output_dir}")
        
    except Exception as e:
        print(f"\nError during portfolio construction: {str(e)}")
        raise

async def get_user_preferences() -> InvestmentPreferences:
    """Get investment preferences from user input"""
    
    print("\n=== Investment Strategy Questionnaire ===\n")
    
    # Core strategy
    print("Select your investment strategy:")
    print("1. Growth (focus on high-growth companies)")
    print("2. Value (focus on undervalued companies)")
    print("3. Balanced (mix of growth and value)")
    strategy_map = {1: 'growth', 2: 'value', 3: 'balanced'}
    while True:
        try:
            strategy = strategy_map[int(input("Enter your choice (1-3): "))]
            break
        except (ValueError, KeyError):
            print("Please enter a valid option (1-3)")

    # Risk tolerance
    print("\nSelect your risk tolerance:")
    print("1. Conservative (lower risk, lower potential return)")
    print("2. Moderate (balanced risk and return)")
    print("3. Aggressive (higher risk, higher potential return)")
    risk_map = {1: 'conservative', 2: 'moderate', 3: 'aggressive'}
    while True:
        try:
            risk_tolerance = risk_map[int(input("Enter your choice (1-3): "))]
            break
        except (ValueError, KeyError):
            print("Please enter a valid option (1-3)")

    # Time horizon
    while True:
        try:
            time_horizon = str(int(input("\nEnter your investment time horizon in years: ")))
            if int(time_horizon) > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

    # Market cap preference
    while True:
        try:
            min_market_cap = float(input("\nEnter minimum market cap in billions USD: "))
            if min_market_cap > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Position size
    while True:
        try:
            max_position_size = float(input("\nEnter maximum position size as percentage (5-15): ")) / 100
            if 0.05 <= max_position_size <= 0.15:
                break
            print("Please enter a value between 5 and 15")
        except ValueError:
            print("Please enter a valid number")

    # Sector preferences
    print("\nEnter preferred sectors (comma-separated, press enter for none):")
    print("Example: Technology, Healthcare, Energy")
    preferred_sectors = [s.strip() for s in input().split(',') if s.strip()]
    
    print("\nEnter excluded sectors (comma-separated, press enter for none):")
    excluded_sectors = [s.strip() for s in input().split(',') if s.strip()]

    # Additional preferences
    esg_focus = input("\nConsider ESG factors? (y/n): ").lower() == 'y'
    dividend_focus = input("Focus on dividend-paying stocks? (y/n): ").lower() == 'y'
    international_exposure = input("Include international stocks? (y/n): ").lower() == 'y'

    # Create and return InvestmentPreferences with all required fields
    return InvestmentPreferences(
        strategy=strategy,
        risk_tolerance=risk_tolerance,
        time_horizon=time_horizon,
        min_market_cap=min_market_cap,
        max_position_size=max_position_size,
        preferred_sectors=preferred_sectors,
        excluded_sectors=excluded_sectors,
        esg_focus=esg_focus,
        dividend_focus=dividend_focus,
        international_exposure=international_exposure
    )

if __name__ == "__main__":
    asyncio.run(main())