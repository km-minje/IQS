from llms.CustomChatModel import CustomChatModel
from llms.OllamaChatModel import OllamaChatModel

def create_llm_model(config, all_config):
    if config["type"] == "customchatmodel":
        return CustomChatModel(config, all_config)
    elif config["type"] == "ollamachatmodel":
        return OllamaChatModel(config)

    else:
        raise ValueError(f"Unknown embedding model type: {config['type']}")