from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
import pandas as pd
import requests
import os
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM


load_dotenv()

# ollam_llm = 
# Initialize LLMs
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=1)

# State definition
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    stock_data: Dict = Field(default_factory=dict)
    technical_analysis: Dict = Field(default_factory=dict)
    fundamental_analysis: Dict = Field(default_factory=dict)
    risk_assessment: Dict = Field(default_factory=dict)
    final_recommendation: Dict = Field(default_factory=dict)

def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock data using Financial Datasets API"""
    headers = {"X-API-KEY": os.getenv("FINANCIAL_DATASETS_API_KEY")}
    
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
        
    data = response.json()
    prices = data.get("prices")
    if not prices:
        raise ValueError("No price data returned")
        
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    
    # Ensure numeric data types
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df.sort_index(inplace=True)
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """Calculate technical indicators"""
    # Calculate moving averages
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate average volume
    df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + (df['close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['BB_middle'] - (df['close'].rolling(window=20).std() * 2)
    
    latest = df.iloc[-1]
    return {
        'current_price': latest['close'],
        'sma20': latest['SMA20'],
        'sma50': latest['SMA50'],
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'signal_line': latest['Signal_Line'],
        'volume': latest['volume'],
        'volume_ma20': latest['Volume_MA20'],
        'bb_upper': latest['BB_upper'],
        'bb_middle': latest['BB_middle'],
        'bb_lower': latest['BB_lower'],
        'daily_change': ((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100
    }

def get_fundamentals(ticker: str) -> Dict:
    """Fetch fundamental data using Financial Datasets API"""
    headers = {"X-API-KEY": os.getenv("FINANCIAL_DATASETS_API_KEY")}
    
    url = f"https://api.financialdatasets.ai/fundamentals/?ticker={ticker}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching fundamentals: {response.status_code} - {response.text}")
        
    return response.json()

def data_collection_agent(state: AgentState) -> Dict:
    """Agent responsible for collecting stock data and calculating indicators"""
    messages = state.messages
    params = messages[-1].additional_kwargs
    
    try:
        # Fetch stock data
        df = get_price_data(params['symbol'], params['start_date'], params['end_date'])
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(df)
        
        # Update state with complete data structure
        state.stock_data = {
            'symbol': params['symbol'],
            'data': df.to_dict(),
            'indicators': indicators
        }
        
        message = HumanMessage(
            content=f"""Stock data collected for {params['symbol']}:
            Current Price: ${indicators['current_price']:.2f}
            Daily Change: {indicators['daily_change']:.2f}%
            SMA20: ${indicators['sma20']:.2f}
            SMA50: ${indicators['sma50']:.2f}
            RSI: {indicators['rsi']:.2f}
            MACD: {indicators['macd']:.2f}
            Signal Line: {indicators['signal_line']:.2f}
            Volume: {indicators['volume']}
            Volume MA20: {indicators['volume_ma20']}
            Bollinger Bands:
            - Upper: ${indicators['bb_upper']:.2f}
            - Middle: ${indicators['bb_middle']:.2f}
            - Lower: ${indicators['bb_lower']:.2f}""",
            name="data_collector"
        )
        
        return {"messages": state.messages + [message], "stock_data": state.stock_data}
        
    except Exception as e:
        message = HumanMessage(
            content=f"Error collecting data: {str(e)}",
            name="data_collector"
        )
        return {"messages": state.messages + [message], "stock_data": {}}

def technical_analysis_agent(state: AgentState) -> Dict:
    """Agent responsible for technical analysis"""
    messages = state.messages
    
    # Check if stock_data exists and has indicators
    if not state.stock_data or 'indicators' not in state.stock_data:
        message = HumanMessage(
            content="Error: No stock data or indicators available for analysis",
            name="technical_analyst"
        )
        return {"messages": state.messages + [message]}
    
    indicators = state.stock_data['indicators']
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a technical analysis expert. As you analyze the provided technical indicators, think through each step of your analysis process carefully and explain your reasoning.
        
        Think step by step through this Analysis Process and use <thinking> tags to guide your thought process:
        1. First, assess the overall trend using moving averages:
           - Compare current price to SMA20 and SMA50
           - Analyze the relationship between SMA20 and SMA50
           - Consider the slope and direction of these averages
           
        2. Next, evaluate momentum using RSI:
           - Check if RSI indicates overbought (>70) or oversold (<30)
           - Look for any divergences with price action
           - Consider recent RSI trends
           
        3. Then, analyze MACD signals:
           - Identify current MACD position relative to signal line
           - Note any recent or pending crossovers
           - Assess MACD histogram pattern
           
        4. Examine volume patterns:
           - Compare current volume to 20-day average
           - Look for volume confirmation of price moves
           - Identify any unusual volume spikes
           
        5. Study Bollinger Bands position:
           - Note price position relative to bands
           - Check for any band squeezes or expansions
           - Identify potential mean reversion opportunities
           
        6. Finally, synthesize all indicators:
           - Look for confirming signals across indicators
           - Identify any contradicting signals
           - Weigh the relative importance of each signal
        
        Your output should include:
        1. Technical Outlook (BULLISH/BEARISH/NEUTRAL)
        2. Confidence Score (0-1)
        3. Key Technical Signals:
           - Trend Analysis (Moving Averages)
           - Momentum (RSI)
           - Volume Analysis
           - MACD Signal
           - Bollinger Bands Position
        4. Support and Resistance Levels
        5. Risk Levels
        
        For each conclusion, explain your step-by-step reasoning and how different indicators support your analysis.
        """),
        ("human", f"""
        Analyze these technical indicators for {state.stock_data['symbol']}:
        
        Price Action:
        - Current Price: ${indicators['current_price']:.2f}
        - Daily Change: {indicators['daily_change']:.2f}%
        
        Moving Averages:
        - SMA20: ${indicators['sma20']:.2f}
        - SMA50: ${indicators['sma50']:.2f}
        
        Momentum/Trend:
        - RSI: {indicators['rsi']:.2f}
        - MACD: {indicators['macd']:.2f}
        - Signal Line: {indicators['signal_line']:.2f}
        
        Volume:
        - Current: {indicators['volume']}
        - 20-day Average: {indicators['volume_ma20']}
        
        Bollinger Bands:
        - Upper: ${indicators['bb_upper']:.2f}
        - Middle: ${indicators['bb_middle']:.2f}
        - Lower: ${indicators['bb_lower']:.2f}
        """)
    ])
    
    response = llm.invoke(analysis_prompt.format_messages())
    
    message = HumanMessage(
        content=response.content,
        name="technical_analyst"
    )
    
    return {"messages": state.messages + [message]}

def fundamental_analysis_agent(state: AgentState) -> Dict:
    """Agent responsible for fundamental analysis"""
    messages = state.messages
    symbol = state.stock_data['symbol']
    
    try:
        # Get fundamental data
        fundamentals = get_fundamentals(symbol)
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are a fundamental analysis expert. Walk through each aspect of the company's fundamentals systematically to build a comprehensive analysis.
        
            Think step by step through the following analysis process use <thinking> tags to guide your thought process:
            1. Begin with Profitability Analysis:
            - Calculate and trend key margins
            - Compare with industry averages
            - Identify margin expansion/contraction
            
            2. Examine Growth Metrics:
            - Analyze revenue growth rates
            - Study earnings growth patterns
            - Evaluate cash flow growth
            - Consider growth sustainability
            
            3. Assess Valuation Metrics:
            - Compare multiple valuation methods
            - Consider historical valuation ranges
            - Evaluate against peer valuations
            - Factor in growth rates
            
            4. Review Financial Health:
            - Analyze debt levels and coverage
            - Study working capital efficiency
            - Evaluate cash flow adequacy
            - Consider balance sheet strength
            
            5. Consider Competitive Position:
            - Evaluate market share trends
            - Assess competitive advantages
            - Study industry dynamics
            - Consider regulatory environment
            
            6. Synthesize All Factors:
            - Weigh relative importance
            - Consider interdependencies
            - Identify key drivers
            - Note potential red flags
            
            Your output should include:
            1. Fundamental Outlook (BULLISH/BEARISH/NEUTRAL)
            2. Confidence Score (0-1)
            3. Key Metrics Analysis:
            - Profitability Metrics
            - Growth Metrics
            - Valuation Metrics
            - Financial Health
            4. Industry Comparison
            5. Key Risks and Opportunities
            
            For each conclusion, explain your reasoning process and how different metrics support your analysis.
            """),
            ("human", f"""
            Analyze these fundamental metrics for {symbol}:
            
            Financial Data:
            {json.dumps(fundamentals, indent=2)}
            """)
        ])
        
        response = llm.invoke(analysis_prompt.format_messages())
        
    except Exception as e:
        response = AIMessage(content=f"Error analyzing fundamentals: {str(e)}")
    
    message = HumanMessage(
        content=response.content,
        name="fundamental_analyst"
    )
    
    return {"messages": state.messages + [message]}

def risk_assessment_agent(state: AgentState) -> Dict:
    """Agent responsible for risk assessment"""
    messages = state.messages
    technical_analysis = messages[-2].content
    fundamental_analysis = messages[-1].content
    
    risk_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a portfolio manager making the final investment decision.
        Walk through each aspect of the investment case systematically.
        
        Think step by step through the following decision process use <thinking> tags to guide your thought process:
        1. First, Review Technical Setup:
           - Evaluate current price action
           - Consider technical signals
           - Assess entry timing
           - Review risk levels
           
        2. Examine Fundamental Foundation:
           - Analyze business quality
           - Consider valuation context
           - Evaluate growth prospects
           - Review financial health
           
        3. Assess Risk Profile:
           - Consider total risk exposure
           - Evaluate risk/reward ratio
           - Review correlation impacts
           - Assess portfolio fit
           
        4. Define Investment Parameters:
           - Calculate position size
           - Set price targets
           - Establish stop levels
           - Plan entry strategy
           
        5. Develop Monitoring Framework:
           - Identify key metrics
           - Set review triggers
           - Plan adjustment criteria
           - Design exit strategy
           
        6. Synthesize Final Decision:
           - Weigh all factors
           - Consider alternatives
           - Evaluate conviction level
           - Finalize action plan
        
        Your output should include:
        1. Investment Action (BUY/SELL/HOLD)
        2. Conviction Level (LOW/MEDIUM/HIGH)
        3. Price Targets:
           - Entry Price Range
           - Target Price
           - Stop Loss
        4. Position Sizing
        5. Investment Thesis
        6. Monitoring Points
        
        For each aspect of your recommendation, explain your decision process and how different analyses support your conclusion.
        """),
        ("human", f"""
        Based on the following analyses, provide a comprehensive risk assessment:
        
        Technical Analysis:
        {technical_analysis}
        
        Fundamental Analysis:
        {fundamental_analysis}
        """)
    ])
    
    response = llm.invoke(risk_prompt.format_messages())
    
    message = HumanMessage(
        content=response.content,
        name="risk_manager"
    )
    
    return {"messages": state.messages + [message]}

def final_recommendation_agent(state: AgentState) -> Dict:
    """Agent responsible for making final investment recommendation"""
    messages = state.messages
    stock_data = state.stock_data
    
    recommendation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a portfolio manager making the final investment decision.
        Walk through each aspect of the investment case systematically.
        
        
        Think step by step through the following decision process use <thinking> tags to guide your thought process:
        1. First, Review Technical Setup:
           - Evaluate current price action
           - Consider technical signals
           - Assess entry timing
           - Review risk levels
           
        2. Examine Fundamental Foundation:
           - Analyze business quality
           - Consider valuation context
           - Evaluate growth prospects
           - Review financial health
           
        3. Assess Risk Profile:
           - Consider total risk exposure
           - Evaluate risk/reward ratio
           - Review correlation impacts
           - Assess portfolio fit
           
        4. Define Investment Parameters:
           - Calculate position size
           - Set price targets
           - Establish stop levels
           - Plan entry strategy
           
        5. Develop Monitoring Framework:
           - Identify key metrics
           - Set review triggers
           - Plan adjustment criteria
           - Design exit strategy
           
        6. Synthesize Final Decision:
           - Weigh all factors
           - Consider alternatives
           - Evaluate conviction level
           - Finalize action plan
        
        Your output should include:
        1. Investment Action (BUY/SELL/HOLD)
        2. Conviction Level (LOW/MEDIUM/HIGH)
        3. Price Targets:
           - Entry Price Range
           - Target Price
           - Stop Loss
        4. Position Sizing
        5. Investment Thesis
        6. Monitoring Points
        
        For each aspect of your recommendation, explain your decision process and how different analyses support your conclusion."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", f"""
        Current Stock Information:
        Symbol: {stock_data['symbol']}
        Price: ${stock_data['indicators']['current_price']:.2f}
        
        Provide your final recommendation in depth report  based on all preceding analyses. 
        """)
    ])
    
    response = llm.invoke(recommendation_prompt.format_messages(messages=messages))
    
    message = HumanMessage(
        content=response.content,
        name="portfolio_manager"
    )
    
    return {"messages": [message]}

# Create the graph
def create_stock_analysis_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("data_collector", data_collection_agent)
    workflow.add_node("technical_analyst", technical_analysis_agent)
    workflow.add_node("fundamental_analyst", fundamental_analysis_agent)
    workflow.add_node("risk_manager", risk_assessment_agent)
    workflow.add_node("portfolio_manager", final_recommendation_agent)
    
    # Add edges
    workflow.set_entry_point("data_collector")
    workflow.add_edge('data_collector', 'technical_analyst')
    workflow.add_edge('technical_analyst', 'fundamental_analyst')
    workflow.add_edge('fundamental_analyst', 'risk_manager')
    workflow.add_edge('risk_manager', 'portfolio_manager')
    workflow.add_edge('portfolio_manager', END)
    
    return workflow.compile()

def analyze_stock(symbol: str, start_date: str, end_date: str) -> Dict:
    """
    Run complete stock analysis using the agent graph
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Analysis start date (YYYY-MM-DD)
        end_date: Analysis end date (YYYY-MM-DD)
    
    Returns:
        Dict containing the complete analysis results
    """
    graph = create_stock_analysis_graph()
    
    initial_state = {
        "messages": [
            HumanMessage(
                content="Analyze this stock and provide investment recommendation.",
                additional_kwargs={
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
        ],
        "stock_data": {},
        "technical_analysis": {},
        "fundamental_analysis": {},
        "risk_assessment": {},
        "final_recommendation": {}
    }
    
    final_state = graph.invoke(initial_state)
    return final_state

# Example usage
if __name__ == "__main__":
    try:
        # Validate API key
        if not os.getenv("FINANCIAL_DATASETS_API_KEY"):
            raise ValueError("FINANCIAL_DATASETS_API_KEY not found in environment variables")
        
        # Example analysis for a stock
        symbol = "ACHR"
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\nStarting analysis for {symbol}")
        print(f"Analysis period: {start_date} to {end_date}")
        print("-" * 100)
        
        # Run the analysis
        results = analyze_stock(symbol, start_date, end_date)
        
        # Print results from each agent
        print("\nAnalysis Results:")
        print("=" * 80)
        
        for message in results["messages"]:
            print(f"\n{message.name.upper()} REPORT:")
            print("-" * 50)
            print(message.content)
            print("=" * 80)
            
        # Save results to file
        output_file = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, "w") as f:
            f.write(f"Stock Analysis Report for {symbol}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: {start_date} to {end_date}\n")
            f.write("=" * 80 + "\n\n")
            
            for message in results["messages"]:
                f.write(f"{message.name.upper()} REPORT:\n")
                f.write("-" * 50 + "\n")
                f.write(message.content + "\n")
                f.write("=" * 80 + "\n\n")
        
        print(f"\nDetailed analysis has been saved to {output_file}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        print("\nDetailed error traceback:")
        print(traceback.format_exc())