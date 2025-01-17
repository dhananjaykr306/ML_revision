# @Author: Dhananjay Kumar
# @Date: 17-01-2025
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 17-01-2025
# @Title: Python program to make a basic financial agent and news agent.


from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.groq import GroqModel
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("GROQ_API_KEY").strip().replace('"', '').replace("'", "")

class StockPriceResult(BaseModel):
    symbol: str
    price: float
    currency: str = "USD"
    message: str

model = GroqModel('llama-3.1-70b-versatile', api_key= api_key)
#model = GroqModel('llama-3.1-70b-versatile', api_key="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")
#model = GroqModel('llama-3.1-70b-versatile')
stock_agent = Agent(
    model,
    result_type=StockPriceResult,
    system_prompt="You are a helpful financial assistant that can look up stock prices. Use the get_stock_price tool to fetch current data."
)

news_agent = Agent(
    model,
    result_type=str,
    system_prompt="You are a helpful financial assistant that can provide news articles. Use the get_news tool to fetch news articles."
)

@stock_agent.tool_plain
def get_stock_price(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price
    return {
        "price": round(price, 2),
        "currency": "USD"
    }

x = input("Enter a stock name:")
result = stock_agent.run_sync(f"What is {x} current stock price?")
print(f"Stock Price: ${result.data.price:.2f} {result.data.currency}")
print(f"Message: {result.data.message}")
z = news_agent.run_sync(f"What are the latest news articles on {x}?")
print(z.data)
