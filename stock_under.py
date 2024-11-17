from typing import List, Dict, Optional
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
from pathlib import Path
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
class UndervaluedMetrics(BaseModel):
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    price_to_book: float
    price_to_earnings: float
    forward_pe: float
    peg_ratio: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    insider_buying: float
    institutional_ownership: float
    short_interest: float
    rsi_14: float
    revenue_growth: float
    earnings_growth: float
    cash_flow_growth: float
    analyst_targets: List[float]
    potential_catalysts: List[str]

class ValueScreeningPreferences(BaseModel):
    max_price: float = Field(default=100.0, description="Maximum stock price")
    min_price: float = Field(default=5.0, description="Minimum stock price")
    min_volume: float = Field(default=500000, description="Minimum daily trading volume")
    max_pe: float = Field(default=25.0, description="Maximum P/E ratio")
    min_market_cap: float = Field(default=300000000, description="Minimum market cap ($300M)")
    min_current_ratio: float = Field(default=1.5, description="Minimum current ratio")
    max_debt_equity: float = Field(default=2.0, description="Maximum debt/equity ratio")
    price_vs_high: float = Field(default=0.4, description="Maximum decline from 52-week high (40%)")

class UndervaluedAnalysisFlow:
    def __init__(self, preferences: ValueScreeningPreferences):
        self.preferences = preferences
        self.output_dir = Path("outputs/undervalued_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_agents()
        self.setup_tasks()

    def setup_agents(self):
        """Initialize specialized agents for finding undervalued stocks"""
        
        # Value Screening Agent
        self.value_screener = Agent(
            role="Value Stock Screening Specialist",
            goal="Identify fundamentally strong but underperforming stocks under $100",
            backstory="""You are an expert in finding hidden value opportunities. You specialize in 
            identifying stocks that are temporarily undervalued due to market inefficiencies or 
            short-term issues but have strong recovery potential. You earn a significant bonus for 
            identifying stocks that achieve >50% returns within 18 months.""",
            tools=[
                StockPriceDataTool(),
                CompanyInfoTool(),
                FinancialReportTool(),
                RealTimeQuoteTool()
            ]
        )

        # Turnaround Analysis Agent
        self.turnaround_analyst = Agent(
            role="Turnaround Potential Analyst",
            goal="Analyze companies for turnaround potential and catalysts",
            backstory="""You specialize in identifying companies undergoing strategic changes or 
            experiencing temporary setbacks. You have a strong track record of spotting companies on 
            the verge of successful turnarounds. Your bonus is tied to accurately predicting 
            successful business transformations.""",
            tools=[
                SerperDevTool(),
                StockNewsTool(),
                FinancialReportTool()
            ]
        )

        # Insider Activity Analyst
        self.insider_analyst = Agent(
            role="Insider Activity Specialist",
            goal="Analyze insider trading patterns and institutional movements",
            backstory="""You are an expert in tracking and analyzing insider buying patterns and 
            institutional ownership changes. You can identify meaningful insider activity that 
            signals potential turnarounds. Your compensation is linked to identifying stocks where 
            insider buying precedes significant price appreciation.""",
            tools=[
                CompanyInfoTool(),
                SerperDevTool(),
                TavilySearchTool()
            ]
        )

        # Technical Analysis Agent
        self.technical_analyst = Agent(
            role="Technical Recovery Analyst",
            goal="Identify stocks showing technical signs of bottoming",
            backstory="""You specialize in identifying technical patterns that signal a stock has 
            bottomed and is beginning to recover. You have developed unique indicators for identifying 
            accumulation phases. You earn bonuses when you spot technical bottoms that lead to 
            sustained recoveries.""",
            tools=[
                StockPriceDataTool(),
                ChartingTool(),
                OptionsChainTool()
            ]
        )

    def setup_tasks(self):
        """Initialize analysis tasks"""
        
        # Initial Value Screening Task
        self.screening_task = Task(
            description=f"""Screen the market for undervalued stocks matching our criteria:

            Price Constraints:
            - Maximum price: ${self.preferences.max_price}
            - Minimum price: ${self.preferences.min_price}
            - Minimum volume: {self.preferences.min_volume:,.0f} shares daily
            - Maximum decline: {self.preferences.price_vs_high*100}% from 52-week high

            Fundamental Requirements:
            - Maximum P/E: {self.preferences.max_pe}
            - Minimum market cap: ${self.preferences.min_market_cap:,.0f}
            - Minimum current ratio: {self.preferences.min_current_ratio}
            - Maximum debt/equity: {self.preferences.max_debt_equity}

            Additional Criteria:
            1. Strong Balance Sheet Analysis:
               - Adequate cash reserves
               - Manageable debt levels
               - Working capital adequacy

            2. Business Health Indicators:
               - Gross margin trends
               - Operating cash flow analysis
               - Market share stability
               - Customer concentration

            3. Management Quality:
               - Track record
               - Strategic initiatives
               - Capital allocation history

            4. Competitive Position:
               - Industry position
               - Competitive advantages
               - Market opportunities

            For each candidate provide:
            1. Detailed metric breakdown
            2. Key strength factors
            3. Primary risks
            4. Recovery catalysts
            5. Target price scenarios

            Your bonus is tied to finding stocks with >50% upside potential and limited downside risk.
            Focus on quality companies facing temporary challenges rather than structurally impaired businesses.""",
            agent=self.value_screener,
            expected_output="Comprehensive report of undervalued stock candidates"
        )

        # Turnaround Analysis Task
        self.turnaround_task = Task(
            description=f"""Analyze each candidate stock for turnaround potential:

            1. Business Recovery Factors
               - Management change impact
               - Strategic restructuring
               - Cost reduction initiatives
               - Market repositioning
               - New product launches

            2. Financial Recovery Indicators
               - Working capital trends
               - Cash flow dynamics
               - Margin improvement potential
               - Debt reduction ability
               - Revenue growth catalysts

            3. Industry Position Analysis
               - Competitive dynamics
               - Market share trends
               - Industry cycle position
               - Regulatory environment
               - Technology disruption impact

            4. Catalyst Identification
               - Near-term events
               - Management initiatives
               - Industry changes
               - Market recognition factors
               - Technical triggers

            For each stock provide:
            1. Turnaround probability score
            2. Key catalyst timeline
            3. Success indicators to monitor
            4. Risk mitigation factors
            5. Expected recovery timeline

            Focus on identifying tangible catalysts that could drive revaluation within 18 months.""",
            agent=self.turnaround_analyst,
            expected_output="Detailed turnaround analysis report"
        )

        # Additional tasks...

    async def execute_undervalued_analysis(self) -> str:
        """Execute the full undervalued stock analysis process"""
        
        # Phase 1: Initial Value Screening
        print("\nExecuting initial value screening...")
        screening_crew = Crew(
            agents=[self.value_screener],
            tasks=[self.screening_task],
            process=Process.sequential,
            verbose=True
        )
        screening_result = await screening_crew.kickoff_async()
        
        # Save initial screening results
        await self.save_phase_output(
            "initial_screening",
            screening_result.raw,
            "Initial value stock screening results"
        )

        # Phase 2: Turnaround Analysis
        print("\nAnalyzing turnaround potential...")
        turnaround_crew = Crew(
            agents=[self.turnaround_analyst],
            tasks=[self.turnaround_task],
            process=Process.sequential,
            verbose=True
        )
        turnaround_result = await turnaround_crew.kickoff_async()
        
        # Save turnaround analysis
        await self.save_phase_output(
            "turnaround_analysis",
            turnaround_result.raw,
            "Detailed turnaround potential analysis"
        )

        # Create final summary report
        final_report = self.create_final_report(
            screening_result.raw,
            turnaround_result.raw
        )
        
        # Save final report
        await self.save_phase_output(
            "final_report",
            final_report,
            "Final undervalued stocks analysis"
        )
        
        return final_report

    def create_final_report(self, screening_data: str, turnaround_data: str) -> str:
        """Create final analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Undervalued Stocks Analysis Report
Generated: {timestamp}

## Executive Summary
This analysis identifies fundamentally strong but underperforming stocks under ${self.preferences.max_price} 
with significant recovery potential.

## Screening Criteria
- Price Range: ${self.preferences.min_price} - ${self.preferences.max_price}
- Minimum Volume: {self.preferences.min_volume:,.0f} shares/day
- Maximum P/E: {self.preferences.max_pe}
- Minimum Market Cap: ${self.preferences.min_market_cap:,.0f}

## Initial Screening Results
{screening_data}

## Turnaround Analysis
{turnaround_data}

## Implementation Strategy
1. Position Sizing
   - Initial Position: 3-4% of portfolio
   - Maximum Position: 5% at cost
   - Average Down Strategy: Add 1% at 10% below entry

2. Risk Management
   - Stop Loss: 25% below entry
   - Position Review: Any fundamental thesis violation
   - Catalyst Monitoring: Quarterly review of turnaround progress

3. Profit Taking
   - First Target: 25% of position at 50% gain
   - Second Target: 25% of position at 100% gain
   - Hold Remainder: Until thesis completion or violation

## Monitoring Guidelines
1. Weekly:
   - Price and volume action
   - News and catalyst updates
   - Insider activity

2. Monthly:
   - Technical trend analysis
   - Industry group strength
   - Institutional ownership changes

3. Quarterly:
   - Financial results review
   - Management execution analysis
   - Catalyst progression
   - Thesis validation

## Risk Warnings
- Use strict position sizing
- Monitor stop levels
- Review thesis quarterly
- Track catalyst timeline
"""
        return report

    async def save_phase_output(self, phase: str, content: str, description: str):
        """Save phase output to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{phase}_{timestamp}.md"
        
        with open(filename, "w") as f:
            f.write(f"# {description}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(content)

async def get_value_preferences() -> ValueScreeningPreferences:
    """Get randomized value screening preferences within reasonable ranges"""
    import random

    # Define ranges for each parameter (min, max)
    ranges = {
        'max_price': (50, 200),
        'min_price': (2, 10),
        'min_volume': (300000, 1000000),
        'max_pe': (15, 30),
        'min_market_cap_millions': (100, 500),
        'min_current_ratio': (1.2, 2.0),
        'max_debt_equity': (1.5, 2.5),
        'price_vs_high_percent': (30, 50)
    }

    # Generate random values within ranges
    max_price = random.uniform(*ranges['max_price'])
    min_price = random.uniform(*ranges['min_price'])
    min_volume = random.uniform(*ranges['min_volume'])
    max_pe = random.uniform(*ranges['max_pe'])
    min_market_cap = random.uniform(*ranges['min_market_cap_millions']) * 1000000
    min_current_ratio = random.uniform(*ranges['min_current_ratio'])
    max_debt_equity = random.uniform(*ranges['max_debt_equity'])
    price_vs_high = random.uniform(*ranges['price_vs_high_percent']) / 100

    # Print selected values for transparency
    print("\n=== Randomly Selected Screening Parameters ===")
    print(f"Maximum Price: ${max_price:.2f}")
    print(f"Minimum Price: ${min_price:.2f}")
    print(f"Minimum Volume: {min_volume:,.0f}")
    print(f"Maximum P/E: {max_pe:.1f}")
    print(f"Minimum Market Cap: ${min_market_cap/1000000:.1f}M")
    print(f"Minimum Current Ratio: {min_current_ratio:.2f}")
    print(f"Maximum Debt/Equity: {max_debt_equity:.2f}")
    print(f"Maximum Decline from 52-week High: {price_vs_high*100:.1f}%\n")

    return ValueScreeningPreferences(
        max_price=max_price,
        min_price=min_price,
        min_volume=min_volume,
        max_pe=max_pe,
        min_market_cap=min_market_cap,
        min_current_ratio=min_current_ratio,
        max_debt_equity=max_debt_equity,
        price_vs_high=price_vs_high
    )

async def main():
    """Execute the undervalued stock analysis 5 times"""
    try:
        for run_number in range(1, 6):
            print(f"\n=== Starting Analysis Run #{run_number} ===")
            
            preferences = await get_value_preferences()
            
            # Create a unique output directory for each run
            analysis_flow = UndervaluedAnalysisFlow(preferences)
            original_output_dir = analysis_flow.output_dir
            new_run_number = run_number+5
            analysis_flow.output_dir = original_output_dir / f"run_{new_run_number}"
            analysis_flow.output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nInitializing undervalued stock analysis for run #{new_run_number}...")
            final_report = await analysis_flow.execute_undervalued_analysis()
            
            print(f"\nRun #{new_run_number} completed successfully!")
            print(f"Results saved in: {analysis_flow.output_dir}")
            
            # Add a small delay between runs to ensure unique timestamps
            await asyncio.sleep(1)
            
        print("\nAll 5 analysis runs completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())