from dataclasses import Field
from crewai import Task
from textwrap import dedent
from typing import List
from pydantic import BaseModel
from datetime import datetime   
import pytz

class CompanyDataOutput(BaseModel):
    ticker: str 
    company_name: str 
    company_info: str 


class MarkdownReportCreationTasks:
    async def get_ny_time(self):
        ny_timezone = pytz.timezone('America/New_York')
        ny_time = datetime.now(ny_timezone)
        date_time = ny_time.strftime('%Y-%m-%d %I:%M:%S %p %Z')
        day = ny_time.strftime('%A')
        time = f"Current Day: {day} Current Date and Time: {date_time}"
        return time

    def __tip_section(self):
        return "If you do your BEST WORK and return exactly what I ask, I'll give you a $10,000 commission for every task you complete!"

    async def parse_input(self, agent, data: str):
        return Task(
               description=dedent(f"""
            Extract and validate financial data input.
            Analyze the input string to extract the company
            symbol and all requested financial metrics. Ensure the symbol is valid
            and the metrics are relevant for comprehensive stock analysis.

            **Parameters**: 
            - data: {data}

            **Additional Instructions**:
            1. Verify the company symbol against a list of publicly traded companies.
            2. Categorize the requested metrics (e.g., profitability, liquidity, valuation).
            3. If any crucial metrics for stock analysis are missing, suggest including them.

            **Notes**
            {self.__tip_section()}
            """
        ),
            agent=agent,
            expected_output="""A validated list of dictionaries containing the symbol and metrics, categorized by analysis type.
            Example output: `[
                {'symbol': 'MSTR', 'category': 'Profitability', 'metric': 'net_income'},
                {'symbol': 'MSTR', 'category': 'Cash Flow', 'metric': 'fcf'},
                {'symbol': 'MSTR', 'category': 'Valuation', 'metric': 'pe_ratio'}
            ]`"""
        )

    async def get_data_from_api(self, agent, context):
        return Task(
               description=dedent(f"""
            Comprehensive Financial Data Retrieval and Analysis
            For each metric identified, retrieve historical data using QuickFS API. 
            Perform a thorough analysis of the data, including:
            1. Calculating year-over-year growth rates
            2. Identifying trends and patterns
            3. Comparing against industry averages
            4. Highlighting any anomalies or red flags

            **Additional Instructions**:
            - Retrieve at least 5 years of historical data for each metric
            - For metrics not directly available, construct them using available data
            - Provide context for each metric (e.g., what's considered good/bad in the industry)

            **Notes**
            Ensure you retrieve and analyze EVERY metric requested. Your analysis will be crucial for investment decision-making.
            {self.__tip_section()}
            """
        ),
            agent=agent,
            context=context,
            expected_output="""A comprehensive analysis of each metric, including:
            1. Raw data points
            2. Calculated growth rates and trends
            3. Industry comparisons
            4. Anomalies and potential red flags
            
            Example output: [
                {
                    'metric': 'net_income',
                    'data': [...historical_data_points],
                    'growth_rates': [...yearly_growth_rates],
                    'trend': "Upward trend with 15% CAGR over 5 years",
                    'industry_comparison': "20% above industry average",
                    'anomalies': "Significant drop in 2020 due to pandemic"
                },
                {...}
            ]"""
        )

    async def create_charts(self, agent, context) -> Task:
        return Task(
            description=dedent(f"""
                Create Comprehensive Financial Visualization Suite:
                Take company symbol from previous task
                Develop a series of advanced charts and graphs that visually represent the company's financial metrics and performance over time. 
                Your visualizations should:

                1. Clearly illustrate trends and patterns in each metric
                2. Include comparative analysis (e.g., vs. industry averages, competitors)
                3. Highlight key inflection points or significant events
                4. Use appropriate chart types for different metrics (e.g., line charts for time series, bar charts for comparisons)
                5. Ensure all charts are clearly labeled and include a brief explanatory caption

                **Additional Instructions**:
                - Create a summary dashboard that combines key metrics
                - Use color coding to indicate positive/negative performance
                - Include forward-looking projections where applicable
                - Ensure charts are accessible and easy to interpret for non-financial experts

                DO NOT alter the metric names when creating chart titles. Maintain consistency with the original data.

                {self.__tip_section()}
            """),
            agent=agent,
            context=context,
            expected_output="""
                A list of file locations for the created charts, including:
                1. Individual metric charts
                2. Comparative charts
                3. Summary dashboard
                
                Example output: [
                    'charts/net_income_trend.png',
                    'charts/fcf_vs_industry.png',
                    'charts/valuation_metrics_summary.png',
                    'charts/financial_performance_dashboard.png'
                ]
                """
        )

    async def write_markdown(self, agent, context,company_symbol):
        return Task(
            description=dedent(f"""
                Produce a Comprehensive Stock Analysis Report
                Create an in-depth, professional-grade stock analysis report in markdown syntax. 
                Your report should provide a clear, data-driven investment thesis for the given company symbol.
                Company Symbol: {company_symbol}

                The report must include:

                1. Executive Summary
                   - Brief company overview
                   - Key findings and investment recommendation

                2. Business Model Analysis
                   - Core products/services
                   - Revenue streams
                   - Competitive landscape

                3. Financial Performance Analysis
                   - Detailed analysis of all requested metrics
                   - Historical performance trends
                   - Comparison to industry benchmarks

                4. Risk Assessment
                   - Identification of key risks (market, financial, operational)
                   - Potential mitigating factors

                5. Valuation Analysis
                   - Current valuation metrics
                   - Comparison to peers
                   - Fair value estimate

                6. Future Outlook
                   - Growth prospects
                   - Upcoming catalysts or challenges

                7. Investment Recommendation
                   - Clear buy/hold/sell recommendation
                   - Supporting rationale
                   - Potential upside/downside scenarios

                8. Appendix
                   - All charts and graphs created in the previous task
                   - Data sources and methodology

                **Additional Instructions**:
                - Use markdown syntax consistently throughout the report
                - Incorporate all charts and data visualizations created in the previous task
                - Provide balanced analysis, acknowledging both positive and negative factors
                - Use headers, subheaders, and bullet points for clear organization
                - Include hyperlinks to relevant external sources for additional context

                {self.__tip_section()}
            """
        ),
            agent=agent,
            context = context,
            expected_output=""" A in detail report in markdown syntax.
                """,
            
        )
    
    async def stock_analysis(self, agent,context):
        return Task(
            description=dedent(f"""
            You are an advanced stock analysis AI with access to a suite of powerful financial tools. 
            Your task is to produce a detailed, well-structured report to help investors make informed decisions about whether to invest in a specific stock. You should use all available tools to gather comprehensive data and provide insightful analysis.
            Current Date and Time: {await self.get_ny_time()}
            Company Symbol: {context}


            Report Structure
            Your report should include the following sections:

            Executive Summary
            Company Overview
            Stock Price Analysis
            Financial Health Assessment
            Market Sentiment and News Analysis
            Options Market Analysis
            Analyst Recommendations
            Risk Assessment
            Investment Thesis
            Conclusion and Recommendation

            Data Gathering and Analysis Process
            For each section of the report, utilize the appropriate tools to gather relevant data. Analyze this data thoroughly, considering both quantitative and qualitative factors. Your analysis should be objective, balanced, and supported by the data you've collected.
            1. Executive Summary

            Provide a brief overview of the company and its stock performance
            Summarize key findings and your investment recommendation

            2. Company Overview
            Use the CompanyInfoTool to gather essential information about the company:

            Company name, sector, and industry
            Market capitalization
            Forward P/E ratio
            Dividend yield (if applicable)
            Briefly describe the company's business model and main products/services

            3. Stock Price Analysis
            Utilize the StockPriceDataTool to retrieve historical stock price data:

            Analyze price trends over various time periods (1 month, 6 months, 1 year, 5 years)
            Calculate key technical indicators (e.g., moving averages, relative strength index)
            Identify support and resistance levels

            Use the RealTimeQuoteTool to get current market data:

            Current stock price
            Trading volume
            Compare current price to historical averages

            4. Financial Health Assessment
            Based on the CompanyInfoTool data and your analysis:

            Evaluate the company's financial ratios (P/E, P/B, debt-to-equity)
            Assess revenue and earnings growth trends
            Compare the company's financial metrics to industry averages

            5. Market Sentiment and News Analysis
            Use the StockNewsTool to gather recent news headlines:

            Summarize key news items and their potential impact on the stock
            Analyze overall market sentiment towards the company
            Identify any upcoming events or catalysts that could affect the stock price

            6. Options Market Analysis
            Employ the OptionsChainTool to retrieve options data:

            Analyze the put/call ratio
            Identify any unusual options activity
            Assess implied volatility levels
            Determine what the options market suggests about future stock price movements

            7. Analyst Recommendations
            Utilize the AnalystRecommendationsTool to gather recent analyst opinions:

            Summarize recent rating changes and actions
            Calculate the consensus rating and price target
            Compare current price to analyst price targets
            Analyze the reasoning behind significant rating changes

            8. Risk Assessment
            Based on all gathered data and your analysis:

            Identify and explain key risks facing the company
            Assess industry-specific risks
            Evaluate market risks that could impact the stock
            Consider any regulatory or legal risks

            9. Investment Thesis
            Develop a comprehensive investment thesis:

            Outline the main reasons to consider investing in the stock
            Discuss potential catalysts for future growth
            Address how the company is positioned to handle identified risks

            10. Conclusion and Recommendation
            Synthesize all analyzed information to provide:

            A clear investment recommendation (Buy, Hold, or Sell)
            Specific price targets (entry point, stop-loss, and profit-taking levels)
            Time horizon for the investment
            Potential return expectations
            Key metrics or events to monitor going forward

            Additional Guidelines

            Use clear, concise language throughout the report
            Support all claims and analyses with data from the provided tools
            Provide balanced perspectives, discussing both bullish and bearish arguments
            Clearly state any assumptions made in your analysis
            Acknowledge areas of uncertainty or where more information would be beneficial
            Tailor the depth of analysis to the complexity of the company and its industry

            Remember, your goal is to provide a comprehensive, unbiased report that enables investors to make well-informed decisions. 
            Your report should be thorough enough for experienced investors while remaining accessible to those with less financial expertise.
            """),
            agent=agent,
            
            expected_output="""A text report in markdown format""",
            
        )
    
    async def company_research_task(self, agent,context):
        return Task(
            description=dedent(f"""
                Current Date and Time: {await self.get_ny_time()}
                Company Info : {context}
                Conduct Comprehensive Company Research and Sentiment Analysis below:

                1. Latest News Analysis
                - Find and analyze the most recent news articles about the company
                - Identify key events, announcements, or developments
                - Evaluate the impact of news on stock price movements
                - Determine overall media sentiment (positive, negative, neutral)

                2. Market Sentiment Assessment
                - Analyze social media discussions and retail investor sentiment
                - Review recent analyst reports and commentary
                - Evaluate institutional investor positions and changes
                - Gauge overall market perception and confidence

                3. Company Performance Context
                - Research recent earnings reports and financial performance
                - Track key business initiatives and strategic moves
                - Analyze competitive position and market share trends
                - Identify any significant management changes or corporate actions

                4. Industry and Market Environment
                - Assess current industry trends and dynamics
                - Evaluate impact of macroeconomic factors
                - Compare performance against key competitors
                - Identify potential opportunities and threats

                5. Future Outlook
                - Research upcoming catalysts or events
                - Analyze growth projections and expansion plans
                - Evaluate potential risks and challenges
                - Consider regulatory or policy changes that could impact the company

                Provide a detailed synthesis of all findings, clearly explaining:
                - How current sentiment could affect stock performance
                - Key drivers of positive or negative sentiment
                - Whether market perception aligns with fundamental performance
                - Potential shifts in sentiment that could impact the stock

                Focus on providing actionable insights that can inform investment decisions.
                Make sure the report takes current date and time into account.
                Current Date and Time: {await self.get_ny_time()}
                Support all conclusions with specific evidence and sources as much as possible.
            """),
            agent=agent,
            expected_output="""
               A markdown report structured as per the instructions above.
            """
        )
    
    def test_task(self, agent):
        return Task(
            description="""
            print hello in output
            """,
            agent=agent,
            expected_output="hello",
        )
    
    async def company_lookup_task(self, agent, name ):
        return Task(
            description=f"""
            Company Name: {name}
            Find the ticker symbol for the company with the given name.
            Find company information for the company with the given name.
            Make sure it is a publicly traded company and listed on a major exchange like NYSE, NASDAQ, AMEX, etc.
            """ ,
            agent=agent,
            expected_output="""Json output with given schema""",
            output_pydantic=CompanyDataOutput,
        )