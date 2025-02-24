from llm_module import LLMInterface

# load api key from file
with open("config.data", "r") as f:
    api_key = f.read().strip()

# Initialize the LLM interface
llm_interface = LLMInterface(
    api_key=api_key,
    base_url="https://api.yesapikey.com/v1",
    model_name="gpt-4o-mini",
    temperature=1
)

# Example: Call the model directly using OpenAI
prompt = "Tell me a joke."
response = llm_interface.call_model(prompt)
print("\nDirect Model Response:", response)

# Example: Call the model using LangChain with a formatted template
template = "You are a friendly assistant. Answer the following question: {prompt}"
response = llm_interface.call_model_with_langchain(prompt, template=template)
print("\nLangChain Formatted Response:", response)