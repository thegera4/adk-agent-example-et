# Premium Google ADK Agent (using DeepSeek API)
import warnings
warnings.filterwarnings("ignore", message='Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"')
import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

print("🚀 Creating Premium Google ADK Agent with DeepSeek...")
# Initialize DeepSeek model with LiteLlm (See docs: https://docs.litellm.ai/docs/providers/deepseek)
deepseek_model = LiteLlm(model="deepseek/deepseek-chat", api_key=DEEPSEEK_API_KEY,)

# 1. Create Google ADK agent
deepseek_agent = Agent(
    name="PremiumDeepSeekAgent",
    model=deepseek_model,
    instruction="""
        You are a sophisticated AI assistant powered by DeepSeek AI.
        Your role is to provide advanced reasoning, technical explanations, and professional insights.
        You excel at complex tasks, including:
        - Analyzing and summarizing technical documents
        - Providing detailed explanations of AI concepts
        - Offering professional advice on enterprise applications
        You have advanced reasoning capabilities and access to cutting-edge AI technology.
        Be professional, insightful, and highlight your advanced capabilities when appropriate.
    """,
    description="Premium agent using DeepSeek model"
)

print("✅ Premium DeepSeek Agent Created!")
print(f" 🤖 Name: {deepseek_agent.name}")
print(" 🧠 Model: deepseek-chat")
print(" 💰 Cost: $0.07 per million tokens (50% discount if used during UTC 16:30 - 00:30 = ⚠️$0.035⚠️ per million tokens)")
print(" 🌟 Capabilities: Fast, reliable, production-ready")

# 2. Set up runner and session service (ADK 1.3.0 pattern)
session_service = InMemorySessionService()
runner = Runner(agent=deepseek_agent, app_name="premium_agent_demo", session_service=session_service)


# 3. Async helper to run and get response
async def ask_premium_agent(question):
    """Ask the agent a question and get the response"""
    input_msg = types.Content(role="user", parts=[types.Part(text=question)])

    async for event in runner.run_async(user_id="student", session_id="premium_demo", new_message=input_msg):
        if event.is_final_response():
            return event.content.parts[0].text

    return "No response received"


# 4. Async main function to test
async def main():
    # Initialize session
    await session_service.create_session(app_name="premium_agent_demo", user_id="student", session_id="premium_demo")

    print("✅ ADK Runtime ready!")

    # Test 1: Complex reasoning task
    print("\n🧪 TEST 1: Advanced Reasoning")
    print("=" * 45)

    # complex_question = "Explain how AI agents could transform enterprise customer service, including 3 specific use cases and potential ROI."
    complex_question = "I am a fullstack developer. Can you give me 3 ideas for projects that I can monetize using AI agents? Please include the tech stack, potential users, and how to monetize them."
    print(f"💬 Question: {complex_question}")
    print("🧠 DeepSeek thinking...")

    response1 = await ask_premium_agent(complex_question)
    print(f"🐳 DeepSeek Agent: {response1}")

    # Test 2: Technical explanation
    print("\n🧪 TEST 2: Technical Knowledge")
    print("=" * 45)

    # tech_question = "Compare Google ADK to LangChain and CrewAI and explain why enterprises choose ADK for production systems."
    tech_question = "Which AI agent framework is better for freelancing projects: LangChain or Google ADK? Please explain the pros and cons of each framework."
    print(f"💬 Question: {tech_question}")
    print("🧠 DeepSeek thinking...")

    response2 = await ask_premium_agent(tech_question)
    print(f"🐳 DeepSeek Agent: {response2}")

    print("\n🌟 Premium DeepSeek Agent showcasing advanced capabilities!")
    print("💡 Perfect for complex reasoning and enterprise applications")
    print("✅ Using Google ADK production patterns!")


# 5. Run it
asyncio.run(main())