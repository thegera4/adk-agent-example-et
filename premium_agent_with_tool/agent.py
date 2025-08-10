# Premium Google ADK Agent (using DeepSeek API)
import warnings
warnings.filterwarnings("ignore", message="there are non-text parts in the response")
warnings.filterwarnings("ignore", message='Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"')
import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from datetime import datetime

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Define simple tool functions
def get_current_time():
    """Returns the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_simple_math(expression: str):
    """Safely calculate simple math expressions like '2 + 3 * 4'"""
    try:
        allowed = '0123456789+-*/(). '
        if all(c in allowed for c in expression):
            result = eval(expression)
            return {"result": result}
        return {"error": "Only basic math allowed."}
    except:
        return {"error": "Failed to calculate."}


print("🛠️ Creating an upgraded agent with tools...")

root_agent = Agent(
    name="SmartADKAgent",
    model="gemini-1.5-flash",
    instruction="""
        You are a helpful AI assistant with two tools:
        - get_current_time(): returns current timestamp
        - calculate_simple_math(expr): returns basic arithmetic
        Use them when asked, and explain your steps.
    """,
    tools=[get_current_time, calculate_simple_math],
    description="Smart agent with time and math tools"
)

# Set up the session service and runner
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="smart_tool_agent", session_service=session_service)

print("✅ Smart agent created with tools!")
print(f"🧰 Tools available: {[fn.__name__ for fn in root_agent.tools]}")


async def test_smart_agent(question):
    print(f"💬 You: {question}")
    print("🤖 Smart Agent is thinking and may invoke tools...")

    content = types.Content(role="user", parts=[types.Part(text=question)])
    async for event in runner.run_async(user_id="user", session_id="tool_session", new_message=content):
        if event.content and event.content.parts:
            part = event.content.parts[0]
            if part.function_call:
                print(f"Tool call: {part.function_call.name}")
            elif part.text and event.content.role == 'model':
                print(f"🤖 Smart Agent: {part.text.strip()}")


print("🧪 TESTING SMART AGENT WITH TOOLS")
print("=" * 40)


async def main():
    await session_service.create_session(app_name="smart_tool_agent", user_id="user", session_id="tool_session")

    print("\n⏰ Test 1: Current Time")
    await test_smart_agent("What time is it right now?")

    print("\n🧮 Test 2: Math Calculation")
    await test_smart_agent("Can you calculate 15 * 7 + 23 for me?")



# Run it
if __name__ == "__main__":
    asyncio.run(main())