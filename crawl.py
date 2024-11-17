from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os
from crewai_tools import FirecrawlCrawlWebsiteTool,FirecrawlScrapeWebsiteTool

load_dotenv()

app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Scrape a website:
# scrape_status = app.scrape_url(
#   'https://docs.crewai.com/concepts/flows#adding-crews-to-flows', 
#   params={'formats': ['markdown', 'html']}
# )
# print(scrape_status)

# Crawl a website:
crawl_status = app.async_crawl_url(
  'https://docs.crewai.com/', 
  params={
    'limit': 500, 
    'scrapeOptions': {'formats': ['markdown', 'html']}
  }
)
print(crawl_status)
import time

result = None

def wait_for_crawl_completion(app, job_id, max_attempts=10, delay=5):
    for _ in range(max_attempts):
        crawl_status = app.check_crawl_status(job_id)
        if crawl_status['status'] == 'completed':
            return crawl_status
        time.sleep(delay)
    return None

crawl_status = wait_for_crawl_completion(app, crawl_status['id'])
if crawl_status and 'data' in crawl_status:
    print(crawl_status['data'])
    result = crawl_status['data']
else:
    print("Crawl did not complete in time or encountered an error")

# # Initialize FirecrawlScrapeWebsiteTool
# scrape_tool = FirecrawlScrapeWebsiteTool(api_key=os.getenv("FIRECRAWL_API_KEY"))

# # Example usage of the scrape tool
# url_to_scrape = 'https://docs.crewai.com/concepts/flows#adding-crews-to-flows'


# # Scrape a website:
# scrape_result = app.scrape_url(url_to_scrape, params={'formats': ['markdown', 'html']})
# print(scrape_result)

with open("scrape_result.md", "w") as f:
    print("Writing to file")
    f.write(str(result))
