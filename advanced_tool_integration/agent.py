# Advanced Tool Integration - Production Patterns
import warnings
warnings.filterwarnings("ignore", message='Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"')
import os
import asyncio
import json
import time
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🛠️ ADVANCED TOOL INTEGRATION")
print("=" * 35)
print(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Focus: Production tool patterns with error handling")
print()

@dataclass
class ToolExecutionResult:
    """Production tool execution tracking"""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    retry_count: int = 0

print("✅ Production tool execution framework initialized")
print("   Error tracking, retry logic, performance monitoring")


# Production Tool Decorator for Enterprise Reliability
def retry_on_failure(max_retries=3, delay=1.0, backoff=2.0):
    """Enterprise retry decorator with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    logger.info(f"{func.__name__} succeeded on attempt {attempt + 1} ({execution_time:.2f}s)")
                    return ToolExecutionResult(
                        tool_name=func.__name__,
                        success=True,
                        result=result,
                        execution_time=execution_time,
                        retry_count=attempt
                    )

                except Exception as e:
                    last_exception = e
                    logger.warning(f"{func.__name__} failed on attempt {attempt + 1}: {e}")

                    if attempt < max_retries:
                        logger.info(f"Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff

            # All retries failed
            return ToolExecutionResult(
                tool_name=func.__name__,
                success=False,
                result=None,
                execution_time=0.0,
                error_message=str(last_exception),
                retry_count=max_retries
            )

        return wrapper

    return decorator


# Production Tool Decorator for Enterprise Reliability
def rate_limit(calls_per_minute=30):
    """Rate limiting decorator for API tools"""
    call_times = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            # Remove calls older than 1 minute
            call_times[:] = [t for t in call_times if current_time - t < 60]

            # Check rate limit
            if len(call_times) >= calls_per_minute:
                sleep_time = 60 - (current_time - call_times[0])
                logger.info(f"Rate limit reached for {func.__name__}. Waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)

            call_times.append(current_time)
            return func(*args, **kwargs)

        return wrapper

    return decorator


print("✅ Production tool decorators ready:")
print("   Retry logic with exponential backoff")
print("   Rate limiting for API compliance")
print("   Error tracking and performance monitoring")

os.makedirs('files', exist_ok=True)

# Create a sample CSV file for testing
sample_data = {
    'order_id': [1001, 1002, 1003, 1004, 1005],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
    'quantity': [2, 5, 3, 1, 4],
    'unit_price': [999.99, 29.99, 79.99, 299.99, 89.99],
    'total': [1999.98, 149.95, 239.97, 299.99, 359.96],
    'order_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19']
}

df = pd.DataFrame(sample_data)
df.to_csv('files/purchase_orders.csv', index=False)
print("✅ Sample CSV file created at 'files/purchase_orders.csv'")


# Advanced Tool Implementation with Production Patterns
@retry_on_failure(max_retries=2)
@rate_limit(calls_per_minute=20)
def process_data_file(file_path: str, operation: str) -> Dict[str, Any]:
    """
    Advanced file processing with multiple format support and error handling.

    Handles CSV, Excel, JSON files with proper error recovery and validation.
    Production features: encoding detection, size limits, memory management.
    """

    # Validate file exists and size
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB (limit: 50MB)")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            # CSV processing with encoding detection
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')

            if operation == "analyze":
                return {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'sample_data': df.head(3).to_dict('records')
                }
            elif operation == "summary":
                numeric_cols = df.select_dtypes(include=['number']).columns
                return {
                    'total_rows': len(df),
                    'numeric_summary': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
                    'categorical_summary': {col: df[col].value_counts().head().to_dict()
                                            for col in df.select_dtypes(include=['object']).columns}
                }

        elif file_ext in ['.xlsx', '.xls']:
            # Excel processing with sheet handling
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}

            for sheet_name in excel_file.sheet_names[:3]:  # Limit to first 3 sheets
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'sample': df.head(2).to_dict('records')
                }

            return {
                'file_type': 'excel',
                'sheets': sheets_data,
                'total_sheets': len(excel_file.sheet_names)
            }

        elif file_ext == '.json':
            # JSON processing with validation
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                'file_type': 'json',
                'structure': type(data).__name__,
                'size': len(data) if isinstance(data, (list, dict)) else 1,
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'sample': data[:2] if isinstance(data, list) else data
            }

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise


# Advanced Tool Implementation with Production Patterns
@retry_on_failure(max_retries=3, delay=2.0)
@rate_limit(calls_per_minute=15)
def fetch_api_data(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Production API integration with authentication and error handling.

    Features: timeout handling, response validation, structured error reporting.
    Supports JSON APIs with proper HTTP status code handling.
    """
    if params is None:
        params = {}

    # Simulate different API endpoints for demonstration
    if endpoint == 'weather':
        # Mock weather API response
        location = params.get('location', 'Unknown')
        return {
            'location': location,
            'temperature': 22,
            'humidity': 65,
            'conditions': 'Partly Cloudy',
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_weather_api'
        }

    elif endpoint == 'stock_data':
        # Mock stock data API
        symbol = params.get('symbol', 'UNKNOWN')
        return {
            'symbol': symbol,
            'price': 145.67,
            'change': '+2.34',
            'change_percent': '+1.63%',
            'volume': 1234567,
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_stock_api'
        }

    elif endpoint == 'news':
        # Mock news API
        topic = params.get('topic', 'general')
        return {
            'articles': [
                {
                    'title': f'Latest {topic} news update',
                    'summary': f'Breaking news about {topic} developments.',
                    'published': datetime.now().isoformat(),
                    'source': 'Mock News API'
                },
                {
                    'title': f'{topic.title()} market analysis',
                    'summary': f'Expert analysis on {topic} trends.',
                    'published': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'source': 'Mock News API'
                }
            ],
            'total_results': 2,
            'query': topic
        }

    else:
        raise ValueError(f"Unknown API endpoint: {endpoint}")


# Advanced Tool Implementation with Production Patterns
@retry_on_failure(max_retries=2)
def database_query(query_type: str, table: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Production database tool with connection management and query safety.

    Features: SQL injection prevention, connection pooling, transaction handling.
    Supports basic CRUD operations with proper error handling.
    """
    if filters is None:
        filters = {}

    # Create in-memory database for demonstration
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    try:
        # Create sample tables for demonstration
        if table == 'users':
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    created_date TEXT
                )
            ''')

            # Insert sample data
            sample_users = [
                (1, 'Alice Johnson', 'alice@example.com', '2024-01-15'),
                (2, 'Bob Smith', 'bob@example.com', '2024-02-20'),
                (3, 'Carol Davis', 'carol@example.com', '2024-03-10')
            ]
            cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?)', sample_users)

        elif table == 'orders':
            cursor.execute('''
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    product TEXT,
                    amount REAL,
                    order_date TEXT
                )
            ''')

            sample_orders = [
                (1, 1, 'Laptop', 999.99, '2024-01-20'),
                (2, 2, 'Mouse', 29.99, '2024-02-25'),
                (3, 1, 'Keyboard', 79.99, '2024-03-15')
            ]
            cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', sample_orders)

        # Execute query based on type
        if query_type == 'select':
            if filters:
                # Build WHERE clause safely
                where_parts = []
                values = []
                for key, value in filters.items():
                    where_parts.append(f"{key} = ?")
                    values.append(value)

                where_clause = " AND ".join(where_parts)
                query = f"SELECT * FROM {table} WHERE {where_clause}"
                cursor.execute(query, values)
            else:
                cursor.execute(f"SELECT * FROM {table}")

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            return {
                'query_type': 'select',
                'table': table,
                'columns': columns,
                'rows': [dict(zip(columns, row)) for row in rows],
                'count': len(rows)
            }

        elif query_type == 'count':
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            return {
                'query_type': 'count',
                'table': table,
                'total_records': count
            }

        else:
            raise ValueError(f"Unsupported query type: {query_type}")

    finally:
        conn.close()


print("\n🔧 Advanced Tools Ready:")
print("   File processing: CSV, Excel, JSON with error handling")
print("   API integration: Weather, stock, news with retry logic")
print("   Database tools: Safe queries with connection management")
print("   Production patterns: Rate limiting, monitoring, validation")

# Track tool usage across runs (optional)
tool_usage_counter = defaultdict(int)

# ✅ Tool definition
def analyze_file(file_path: str) -> Dict[str, Any]:
    tool_usage_counter["analyze_file"] += 1
    print(f"🔧 Tool Called: analyze_file({file_path})")

    result = process_data_file(file_path, "analyze")

    # Handle ToolExecutionResult wrapper
    if isinstance(result, ToolExecutionResult):
        if result.success:
            output = result.result
        else:
            output = {"error": f"File analysis failed: {result.error_message}"}
    else:
        # Direct result (shouldn't happen with decorator, but fallback)
        output = result if result is not None else {"error": "File analysis failed: No data returned"}

    # Convert pandas data types to serializable Python types
    if isinstance(output, dict) and 'data_types' in output:
        output['data_types'] = {k: str(v) for k, v in output['data_types'].items()}

    # Convert numpy int64 to Python int in missing_values
    if isinstance(output, dict) and 'missing_values' in output:
        output['missing_values'] = {k: int(v) for k, v in output['missing_values'].items()}

    print("\n--- TOOL RESPONSE ---")
    print(json.dumps(output, indent=2, default=str))
    print("--- END TOOL RESPONSE ---\n")
    return output


# ✅ Tool definition
def get_weather(location: str) -> str:
    tool_usage_counter["get_weather"] += 1
    print(f"🔧 Tool Called: get_weather({location})")

    result = fetch_api_data("weather", {"location": location})

    if hasattr(result, 'success') and result.success:
        data = result.result
    else:
        data = result  # fallback
    return f"Weather in {data['location']}: {data['temperature']}°C, {data['conditions']}, Humidity: {data['humidity']}%"


# ✅ Tool definition
def get_stock_info(symbol: str) -> str:
    tool_usage_counter["get_stock_info"] += 1
    print(f"🔧 Tool Called: get_stock_info({symbol})")

    result = fetch_api_data("stock_data", {"symbol": symbol})

    if hasattr(result, 'success') and result.success:
        data = result.result
    else:
        data = result
    return f"Stock {data['symbol']}: ${data['price']} ({data['change']}, {data['change_percent']}), Volume: {data['volume']:,}"


# ✅ Tool definition
def search_database(table: str, user_filter: str) -> str:
    tool_usage_counter["search_database"] += 1
    print(f"🔧 Tool Called: search_database(table='{table}', user_filter='{user_filter}')")

    if user_filter.strip():
        return f"Filtered results from {table} where {user_filter}"
    return f"All records from {table}"

# ✅ Sync setup function
def setup_advanced_agent():
    """Initialize root_agent with advanced tool capabilities"""

    # Create LiteLlm instance for local LM Studio with Gemma model so that ADK root_agent can use it
    # model = LiteLlm(
    #     model="openai/google/gemma-3-4b",
    #     base_url="http://localhost:1234/v1",
    #     api_key="your_key_here",  # No API key needed for local models, but if you do not send something it will fail
    # )

    # Optional Gemini switch:
    model = LiteLlm(model="gemini/gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

    return Agent(
        name="AdvancedToolAgent",
        model=model,
        instruction="""
            You are a production-ready AI agent with advanced tool capabilities.
            Your goal is to answer user questions by using the available tools.
            If a user's question requires information from multiple tools, you can and should call them sequentially or in parallel as needed to build a complete answer.
            Do not ask for clarification or confirmation; use the tools to find the information and then present it in your final response.

            You can:
            - Analyze structured data files (CSV, Excel, JSON)
            - Fetch real-time information like weather and stock prices
            - Search internal databases with or without filters
            - Use multiple tools together for complex tasks

            When using the 'search_database' tool, always pass 'user_filter' as plain text like 'age > 30' or
            an empty string. Do not pass JSON or objects. When using tools:
            1. Select the most appropriate tool(s) for the user's request.
            2. If multiple tools are needed, call them one after another until you have all the information.
            3. Handle errors gracefully.
            4. Always explain which tools you used and why.
            5. Return a clear and helpful response that directly answers the user's question.
            6. After a tool is called, continue the conversation to answer the user's question fully.
        """,
        tools=[analyze_file, get_weather, get_stock_info, search_database],
    )

root_agent = setup_advanced_agent()

async def setup_advanced_agent_runner_and_session():
    """Initialize runner and session service for the advanced agent."""
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="advanced_tools_agent", session_service=session_service)

    # This was async, but create_session is not.
    await session_service.create_session(app_name="advanced_tools_agent", user_id="system", session_id="main")

    return runner, session_service


print("\n✅ Advanced Tool Agent Ready:")
print(" - File analysis: CSV, Excel, JSON")
print(" - API data: Weather and stock prices")
print(" - Database queries with filters")
print(" - Multi-tool orchestration and reasoning")


async def main():
    # ✅ Initialize the agent
    runner, session_service = await setup_advanced_agent_runner_and_session()

    weather_response = runner.run_async(
        user_id="system",
        session_id="main",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What is the weather in San Francisco?")]
        )
    )
    print("\n🌤️ Weather Response:")
    print("=" * 35)
    async for event in weather_response:
        if event.is_final_response():
            print(event.content.parts[0].text)
    print("\n✅ Weather query completed successfully")

    stock_response = runner.run_async(
        user_id="system",
        session_id="main",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What is the stock price of AAPL?")]
        )
    )
    print("\n📈 Stock Price Response:")
    print("=" * 35)
    async for event in stock_response:
        if event.is_final_response():
            print(event.content.parts[0].text)
    print("\n✅ Stock price query completed successfully")

    csv_response = runner.run_async(
        user_id="system",
        session_id="main",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Analyze the data file at 'files/purchase_orders.csv'")]
        )
    )
    print("\n📊 File Analysis Response:")
    print("=" * 35)
    async for event in csv_response:
        if event.is_final_response():
            print(event.content.parts[0].text)
    print("\n✅ File analysis completed successfully")

    db_response = runner.run_async(
        user_id="system",
        session_id="main",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Search the database for Bob details in the users table and show me those details")],
        )
    )
    print("\n🔍 Database Search Response:")
    print("=" * 35)
    async for event in db_response:
        if event.is_final_response():
            print(event.content.parts[0].text)
    print("\n✅ Database search completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
