# Free Google ADK Agent (using LM Studio, LiteLLM and Gemma model)
import warnings
warnings.filterwarnings("ignore", message='Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"')
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

# Create LiteLlm instance for local LM Studio with Gemma model so that ADK agent can use it
llm = LiteLlm(
    model="openai/google/gemma-3-4b",
    base_url="http://localhost:1234/v1",
    api_key="your_key_here",  # No API key needed for local models, but if you do not send something it will fail
)

# Create Google ADK agent
root_agent = Agent(
    name="LocalLLMAgent",
    model=llm,
    instruction="""
        You are a sophisticated AI assistant powered by LM Studio and the Google Gemma model.
        Your role is to provide advanced reasoning, technical explanations, and professional insights.
        Be professional, insightful, and highlight your advanced capabilities when appropriate.
    """,
    description="Free Local agent using LM Studio with Google Gemma model"
)

# Runtime setup
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="local_agent_demo", session_service=session_service)


# Ask function
async def ask_local_agent(question):
    """Ask the agent a question and get the response"""
    content = types.Content(role="user", parts=[types.Part(text=question)])
    async for event in runner.run_async(user_id="local_user", session_id="local_session", new_message=content):
        if event.is_final_response():
            return event.content.parts[0].text
    return "❌ No response received"


async def main():
    await session_service.create_session(app_name="local_agent_demo", user_id="local_user", session_id="local_session")
    question = "Why are local Gemma models userful for developers?"
    print(f"❓ Question: {question}")
    response = await ask_local_agent(question)
    print(f"🤖 Response: {response}")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
