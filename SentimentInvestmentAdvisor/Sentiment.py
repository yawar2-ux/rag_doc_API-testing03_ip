import os
from fastapi import  HTTPException, Form, APIRouter
from pydantic import BaseModel, HttpUrl
from typing import List, Set
import requests
import urllib.parse
from praw import Reddit
import httpx
from groq import Groq
from bs4 import BeautifulSoup
import re

# API Configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REDDIT_CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "username": os.getenv("REDDIT_USERNAME"),
    "password": os.getenv("REDDIT_PASSWORD"),
    "user_agent": os.getenv("REDDIT_USER_AGENT")
}

print(NEWSAPI_KEY, GROQ_API_KEY, REDDIT_CONFIG)

router = APIRouter()

class HeadlineAnalysis(BaseModel):
    headline: str
    sentiment: str
    recommendation: str
    explanation: str




class UrlInput(BaseModel):
    urls: List[HttpUrl] #This class is used to pass a list of URLs into a system.



class HeadlineAnalyzer:
    def __init__(self):
        self.reddit = Reddit(**REDDIT_CONFIG)
        self.http_client = httpx.Client()
        self.groq_client = Groq(api_key=GROQ_API_KEY, http_client=self.http_client)
        self.seen_headlines: Set[str] = set()


    async def analyze_reddit_headline(self, headline: str, post_data: dict = None) -> HeadlineAnalysis:
        """Special analysis method for Reddit posts with enhanced explanation"""
        prompt = f"""
        Analyze the 10 financial headline as an investor point of view from reddit
        "{headline}"

        Additional context (if available):
        Upvotes: {post_data.get('score', 'N/A')}
        Comments: {post_data.get('num_comments', 'N/A')}
        1.Fetch only investor-focused headlines from Reddit, strictly related to finance, Bitcoin trading, and stock market investments.
        2.The headlines must provide insights on profit/loss, market trends, , price fluctuations, and financial impacts.
        3.Focus only on investor viewpoints, such as potential gains, losses, downturns, uptrends, and economic factors influencing investment decisions."
        4.headline should be well-structed , unique and meaningfull
        5.No political news should be displayed .Only news related to investor should be there.


        If it's not a quality financial headline, respond with "NOT_QUALITY".

        If it is a quality headline, provide an analysis with:
        1. Market impact and potential consequences
        2. Key factors or drivers behind the news
        3. Possible investor implications
        4.headline should be well structured,unique and meaningfull
        5..No political news should be displayed .Only news related to investor should be there.
        Respond in exactly this format:
        Quality: [YES/NOT_QUALITY]
        Sentiment: [strictly only Positive/Negative/Neutral]
        Recommendation: [strictly only Buy/Sell/Hold]
      Explanation: Provide a brief, focused explanation compulsory for all headlines in 2-3 sentences. Include the main point, key implications, and end with a clear conclusion.
CRITICAL: Keep your explanation BRIEF and COMPLETE with proper ending punctuation.Do not leave it empty. Do not write long explanations that risk being cut off.
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            response = completion.choices[0].message.content.strip()
            lines = response.split('\n')

            quality = next((line.replace("Quality:", "").strip()
                          for line in lines if line.startswith("Quality:")), "NOT_QUALITY")

            if quality == "NOT_QUALITY":
                return None

            sentiment = next((line.replace("Sentiment:", "").strip()
                            for line in lines if line.startswith("Sentiment:")), "Neutral")
            recommendation = next((line.replace("Recommendation:", "").strip()
                                for line in lines if line.startswith("Recommendation:")), "Hold")
            explanation = next((line.replace("Explanation:", "").strip()
                             for line in lines if line.startswith("Explanation:")), "")

            return HeadlineAnalysis(
                headline=headline,
                sentiment=sentiment,
                recommendation=recommendation,
                explanation=explanation
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def analyze_headline(self, headline: str) -> HeadlineAnalysis:
        """Standard analysis method for non-Reddit sources"""
        prompt = f"""
        Analyze the top 10  financial headline as an investor point of view.
        "{headline}"
      1.Fetch only investor-focused headlines, strictly related to finance, Bitcoin trading, and stock market investments.
        2.The headlines must provide insights on profit/loss, market trends, investment risks, price fluctuations, and financial impacts.
        3.Focus only on investor viewpoints, such as potential gains, losses, downturns, uptrends, and economic factors influencing investment decisions."
        4.headline should be well-structed , unique and meaningfull.
        5..No political news should be displayed .Only news related to investor should be there.

        If it's not a quality financial headline, respond with "NOT_QUALITY".

        If it is a quality headline, provide analysis in exactly this format:
        Quality: [YES/NOT_QUALITY]
        Sentiment: [strictly only Positive/Negative/Neutral]
        Recommendation: [strictly only Buy/Sell/Hold]
        Explanation: Provide a brief, focused explanation compulsory for all headlines in 2-3 sentences. Include the main point, key implications, and end with a clear conclusion.
CRITICAL: Keep your explanation BRIEF and COMPLETE with proper ending punctuation.Do not leave it empty. Do not write long explanations that risk being cut off.
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            response = completion.choices[0].message.content.strip()
            lines = response.split('\n')

            quality = next((line.replace("Quality:", "").strip()
                          for line in lines if line.startswith("Quality:")), "NOT_QUALITY")

            if quality == "NOT_QUALITY":
                return None

            sentiment = next((line.replace("Sentiment:", "").strip()
                            for line in lines if line.startswith("Sentiment:")), "Neutral")
            recommendation = next((line.replace("Recommendation:", "").strip()
                                for line in lines if line.startswith("Recommendation:")), "Hold")
            explanation = next((line.replace("Explanation:", "").strip()
                             for line in lines if line.startswith("Explanation:")), "")

            return HeadlineAnalysis(
                headline=headline,
                sentiment=sentiment,
                recommendation=recommendation,
                explanation=explanation
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def is_quality_headline(self, text: str) -> bool:
        """Basic filtering of obviously bad headlines"""
        skip_patterns = [
            r'(Daily|Weekly|Monthly|Quarterly) (Discussion|Thread|Chat|Megathread)',
            r'(Rate My|What Are Your|Thoughts On|How Do You)',
            r'Thread|Megathread|Discussion',
            r'[A-Z]{3,5}\s+Earnings\s+Thread',
            r'DD|YOLO|Loss|Gain Porn',
            r'\?$'
        ]

        if any(re.search(pattern, text, re.I) for pattern in skip_patterns):
            return False

        word_count = len(text.split())
        return 8 <= word_count <= 50

analyzer = HeadlineAnalyzer()

@router.post("/reddit-headlines", response_model=List[HeadlineAnalysis])
async def get_reddit_headlines(subreddits: str = Form(...)):
    subreddit_list = [s.strip() for s in subreddits.split(',') if s.strip()]
    headlines = []
    analyzer.seen_headlines.clear()

    # Hardcoded financial keywords
    financial_keywords = {"Interest Rates", "Federal Reserve", "Economic Indicators", "Bond Yields",
"Inflation", "Stock Buybacks", "Market Correction", "Bearish", "Bullish",
"Portfolio Diversification", "Recession", "Financial Stability", "SEC Regulations",
"Debt Ceiling", "Market Liquidity", "Tech Stocks", "Energy Stocks",
 "Investment Strategy", "Blue-Chip Stocks","Stock markets","stocks","revenue","profit/loss","stocks",'stock', 'market', 'shares', 'earnings', 'revenue', 'profit', 'losses',
            'investment', 'investors'
        }

    for subreddit_name in subreddit_list:
        try:
            subreddit = analyzer.reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=70):
                if len(headlines) >= 40:  # Limit to 40 total headlines
                    break

                # Check if post contains any hardcoded financial keyword
                if post.score >= 200 and any(keyword.lower() in post.title.lower() for keyword in financial_keywords):
                    post_data = {
                        'score': post.score,
                        'num_comments': post.num_comments
                    }
                    analysis = await analyzer.analyze_reddit_headline(post.title, post_data)
                    if analysis:  # Only add if it passed quality check
                        headlines.append(analysis)
                        analyzer.seen_headlines.add(post.title)

        except Exception as e:
            print(f"Error processing subreddit {subreddit_name}: {str(e)}")
            continue

    if not headlines:
        raise HTTPException(status_code=404, detail="No quality headlines found")

    return headlines

@router.post("/web-headlines", response_model=List[HeadlineAnalysis])
async def analyze_web_headlines(urls: UrlInput):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

    headlines = []
    analyzer.seen_headlines.clear()
    invalid_urls = []

    # Expanded finance-specific keywords for content validation
    finance_keywords = [
"Interest Rates", "Federal Reserve", "Economic Indicators", "Bond Yields",
"Inflation", "Stock Buybacks", "Market Correction", "Bearish", "Bullish",
"Portfolio Diversification", "Recession", "Financial Stability", "SEC Regulations",
"Debt Ceiling", "Market Liquidity", "Tech Stocks", "Energy Stocks",
 "Investment Strategy", "Blue-Chip Stocks","Stock markets","stocks","revenue","profit/loss","stocks",'stock', 'market', 'shares', 'earnings', 'revenue', 'profit', 'losses',
            'investment', 'investors'
    ]

    # Special handling for known sites
    known_sites = {
        'google.com/finance': {
            'selectors': ['.Yfwt5', '.AoCdqe', '.F86jBf', '.yY2yhb', '.Vfqaje', '.LjnzHe'],
            'container_selectors': ['.yWl69', '.P2Bjpe', '.bLLb2d', '.cA9K0c', '.FxVSWd']
        },
        'finance.yahoo.com': {
            'selectors': ['.js-content-viewer h3', '.Fw(b)', '.Mb(5px)', '.Fz(18px)'],
            'container_selectors': ['.Pos(r)', '.Cf', '.D(ib)', '.Ov(h)']
        },
        'investing.com': {
            'selectors': ['.articleItem', '.title', '.article__title'],
            'container_selectors': ['.largeTitle', '.midTitle', '.articleItemBox']
        },
        'marketwatch.com': {
            'selectors': ['.article_headline', '.headline', '.story_headline'],
            'container_selectors': ['.collection__elements', '.column--primary']
        }
    }

    for url in urls.urls:
        try:
            # Convert URL to string explicitly before using
            url_str = str(url)
            url_lower = url_str.lower()

            # Basic URL format validation
            if not url_str.startswith(('http://', 'https://')):
                invalid_urls.append(f"{url_str} - Invalid URL format")
                continue

            print(f"Fetching content from: {url_str}")

            # Check if this is a known site that needs special handling
            site_specific_selectors = None
            site_specific_containers = None

            for known_site, config in known_sites.items():
                if known_site in url_lower:
                    site_specific_selectors = config['selectors']
                    site_specific_containers = config['container_selectors']
                    print(f"Using specialized selectors for {known_site}")
                    break

            response = requests.get(url_str, headers=headers, timeout=15)

            # Check status code
            if response.status_code != 200:
                invalid_urls.append(f"{url_str} - HTTP Error: {response.status_code}")
                continue

            # Save HTML content for debugging if needed
            #with open('debug_page.html', 'w', encoding='utf-8') as f:
            #    f.write(response.text)

            soup = BeautifulSoup(response.content, 'html.parser')

            # Use site-specific selectors if available
            elements = []
            if site_specific_selectors:
                # First try containers to narrow down where to look for headlines
                containers = []
                for container in site_specific_containers:
                    containers.extend(soup.select(container))

                # If containers found, look for headlines within them
                if containers:
                    for container in containers:
                        for selector in site_specific_selectors:
                            elements.extend(container.select(selector))

                # If no headlines found in containers, try direct selectors
                if not elements:
                    for selector in site_specific_selectors:
                        elements.extend(soup.select(selector))

                print(f"Found {len(elements)} elements using site-specific selectors")

            # If site-specific selectors didn't work, fall back to generic selectors
            if not elements:
                # Standard headline selectors
                headline_selectors = [
                    # Financial news specific
                    'h1.article-title', 'h2.article-title', 'h3.article-title',
                    '.headline', '.article-headline', '.story-heading', '.article__headline',
                    '.news-headline', '.story-title', '.finance-news h2', '.market-news-item',
                    '.stock-news-headline', '.article_title', '.article-name', '.post-title',

                    # General news selectors
                    'h1.title', 'h2.title', 'h3.title', 'h1.entry-title', 'h2.entry-title',
                    '.entry-title', 'article h1', 'article h2', '.article h1', '.article h2',
                    '.post h1', '.post h2', '.story h1', '.story h2'
                ]

                for selector in headline_selectors:
                    elements.extend(soup.select(selector))

                # If still not enough elements, try very generic selectors
                if len(elements) < 5:
                    for selector in ['h1', 'h2', 'h3', 'a[href*="article"]', 'a[href*="news"]', 'a.title']:
                        elements.extend(soup.select(selector))

                print(f"Found {len(elements)} elements using generic selectors")

            # For Google Finance specifically, try finding news items via text content
            if 'google.com/finance' in url_lower and not elements:
                print("Special handling for Google Finance")
                # Try to find all link elements that might contain news
                link_elements = soup.find_all('a')
                for link in link_elements:
                    # Look for links with reasonable length text that might be headlines
                    text = link.get_text().strip()
                    if text and 20 <= len(text) <= 200:
                        elements.append(link)
                print(f"Found {len(elements)} potential news links on Google Finance")

            url_headlines = 0
            finance_headlines_found = False

            for element in elements:
                if url_headlines >= 7:
                    break

                text = element.get_text().strip()
                # Skip empty, very short, or excessively long text
                if not text or len(text) < 20 or len(text) > 200:
                    continue

                # Process this potential headline
                print(f"Potential headline: {text[:60]}...")

                # For Google Finance, we know the content is finance-related, so skip keyword check
                skip_keyword_check = 'google.com/finance' in url_lower

                # Check headline for finance keywords (unless we're on a known finance site)
                contains_finance_keyword = skip_keyword_check or any(keyword in text.lower() for keyword in finance_keywords)

                if contains_finance_keyword and analyzer.is_quality_headline(text) and text not in analyzer.seen_headlines:
                    analyzer.seen_headlines.add(text)

                    analysis = await analyzer.analyze_headline(text)
                    if analysis:
                        headlines.append(analysis)
                        url_headlines += 1
                        finance_headlines_found = True
                        print(f"✓ Added quality headline ({url_headlines}/10)")
                    else:
                        print("✗ Failed LLM quality check")

            if not finance_headlines_found:
                invalid_urls.append(f"{url_str} - No finance headlines found in content")

        except requests.exceptions.RequestException as e:
            invalid_urls.append(f"{url_str} - Connection error: {str(e)}")
            print(f"Request error for {url}: {str(e)}")
            continue
        except Exception as e:
            invalid_urls.append(f"{url_str} - Processing error: {str(e)}")
            print(f"Error processing URL {url}: {str(e)}")
            continue

    # Prepare response
    if not headlines:
        detail = "No finance-related headlines found."
        if invalid_urls:
            detail += f" Issues with URLs: {'; '.join(invalid_urls)}"
        raise HTTPException(status_code=404, detail=detail)

    print(f"Successfully found {len(headlines)} quality finance headlines")
    return headlines


@router.post("/news-headlines", response_model=List[HeadlineAnalysis])
async def analyze_news_headlines():
    # Predefined financial keywords
    financial_terms = [
        "stocks", "market", "finance", "investment", "economy",
        "business", "trading", "crypto", "bitcoin", "earnings",
        "financial", "nasdaq", "dow jones", "Capital Markets",
        "Shareholders", "Stock Prices", "Banking Investment"
    ]

    try:
        async with httpx.AsyncClient() as client:
            # Dynamic URL approaches that don't require manual date updates
            query_approaches = [
                # Approach 1: Prioritize recent business news
                f"https://newsapi.org/v2/top-headlines?category=business&sortBy=publishedAt&pageSize=50&apiKey={NEWSAPI_KEY}",

                # Approach 2: Combined financial keywords
                f"https://newsapi.org/v2/top-headlines?q={urllib.parse.quote(' OR '.join(financial_terms))}&sortBy=publishedAt&pageSize=50&apiKey={NEWSAPI_KEY}",

                # Approach 3: Broad financial insights
                f"https://newsapi.org/v2/top-headlines?q=investor+OR+investment+OR+financial news+trends&sortBy=publishedAt&pageSize=50&apiKey={NEWSAPI_KEY}"
            ]

            for url in query_approaches:
                print(f"Attempting URL: {url}")  # Debug print

                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    # Parse JSON response
                    response_json = response.json()
                    articles = response_json.get("articles", [])

                    print(f"Total articles found: {len(articles)}")  # Debug print

                    headlines = []
                    analyzer.seen_headlines.clear()

                    for article in articles:
                        if len(headlines) >= 15:  # Limit to 10 headlines
                            break

                        headline = article.get("title", "").strip()
                        published_at = article.get("publishedAt", "")

                        # Relaxed quality checks
                        if headline and len(headline) > 7:
                            try:
                                analysis = await analyzer.analyze_headline(headline)
                                if analysis:
                                    headlines.append(analysis)
                                    analyzer.seen_headlines.add(headline)
                                    print(f"Added headline: {headline}")  # Debug print
                            except Exception as analysis_error:
                                print(f"Analysis failed for headline: {headline}. Error: {analysis_error}")

                    # If we found headlines, return them
                    if headlines:
                        return headlines

                except Exception as approach_err:
                    print(f"Error with this approach: {approach_err}")
                    continue

            # If no headlines found after all approaches
            raise HTTPException(status_code=404, detail="No financial headlines found across multiple query approaches")

    except Exception as e:
        print(f"Full error details: {str(e)}")  # Comprehensive error logging
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")
