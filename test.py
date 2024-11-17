import asyncio
import os
from dotenv import load_dotenv
from quickfs import QuickFS
from tools import DataFetchingTool,ChatAnalysisTool, StockPriceDataTool,RealTimeQuoteTool,OptionsChainTool,AnalystRecommendationsTool,StockNewsTool,CompanyInfoTool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

FetchData = DataFetchingTool()
chat_analysis = ChatAnalysisTool()
stock_price_data = StockPriceDataTool()
real_time_quote = RealTimeQuoteTool()
options_chain = OptionsChainTool()
analyst_recommendations = AnalystRecommendationsTool()
stock_news = StockNewsTool()
company_info = CompanyInfoTool()



async def main():
    
    stock_price_data_result =  stock_price_data.run("NVDA", "1y")
    print(stock_price_data_result)
    real_time_quote_result =  real_time_quote.run("NVDA")
    print(real_time_quote_result)
    options_chain_result = options_chain.run("NVDA", "2024-10-25")
    print(options_chain_result)
    analyst_recommendations_result = analyst_recommendations.run("NVDA")
    print(analyst_recommendations_result)
    stock_news_result = stock_news.run("NVDA")
    print(stock_news_result)
    company_info_result = company_info.run("NVDA")
    print(company_info_result)



if __name__ == "__main__":
    asyncio.run(main())


