import warnings
warnings.filterwarnings("ignore", message='Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"')
import os
from dotenv import load_dotenv

print("🔍 Checking Environment...")

# Option 1: FREE Local LLM (LM Studio)
print("\n🆓 Option 1: FREE Local LLM (LM Studio)")
try:
    import requests
    import json

    response = requests.get("http://localhost:1234/v1/models")

    if response.status_code == 200:
        models = response.json()
        available_models = [model['id'] for model in models['data']]
        print(f"✅ LM Studio is running with these available models: {', '.join(available_models)}")
        lmstudio_available = True
        recommended_model = [model for model in available_models if "google/gemma-3-4b" in model]
        print(f"💡 Recommended model: {', '.join(recommended_model)}")
    else:
        print("❌ LM Studio not responding...")
        lmstudio_available = False
except:
    print("❌ LM Studio not found or not running.")
    print("💡 Install: https://lmstudio.ai/")
    lmstudio_available = False

# Option 2: PAID DeepSeek API
print("\n🐳 Option 2: PAID DeepSeek API")
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_api_key:
    print("✅ DeepSeek API key found.")
    print("💰 Cost per million tokens: $0.07 for deepseek-chat, $0.14 for deepseek-reasoner")
    print("💡 Recommended model for general purposes: deepseek-chat. Advanced recommendation: deepseek-reasoner")
    deepseek_available = True
else:
    print("❌ DeepSeek API key not found.")
    print("💡 Get your API key: https://deepseek.com/ and set it in your env variables as DEEPSEEK_API_KEY='your_api_key'")
    deepseek_available = False

# Show Recommendation
print("\n🎯 Recommendation:")
if lmstudio_available:
    print("✅ Start with FREE LM Studio - perfect for learning and practicing!")
if deepseek_available:
    print("✅ Upgrade to DeepSeek API for production features and advanced capabilities!")
if not (lmstudio_available or deepseek_available):
    print("️❌ No options available. Please set up at least one option to proceed.")