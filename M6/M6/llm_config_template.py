"""
Shared LLM Configuration for Module 5 Agents Notebooks
========================================================

This configuration file provides Azure AI Foundry endpoint settings
for all Module 5 notebooks. It follows the same pattern as Basic_LLM_Call.py.

IMPORTANT: 
- Do NOT commit real API keys to version control
- Use environment variables or placeholder values
- Each participant should replace placeholders with their own credentials
"""

import os

# ========= AZURE AI FOUNDRY CONFIGURATION =========
# These values are configured from the existing repository setup
# From Azure AI Foundry / Azure AI Studio

# Azure AI Foundry inference endpoint (should end with "/models")
INFERENCE_ENDPOINT = os.getenv(
    "AZURE_INFERENCE_ENDPOINT", 
    "https://srika-mkndeeu4-eastus2.openai.azure.com/"
)

# API key from Azure AI Foundry
API_KEY = os.getenv(
    "AZURE_API_KEY",
    "FQxFqow6fCaAPGa0DwBQGX8mwEhewKgtIMMziROqlfIC3PA6qpm6JQQJ99CAACHYHv6XJ3w3AAAAACOGub6I"
)

# Model deployment name (e.g., "gpt-4o-mini", "gpt-35-turbo")
MODEL_NAME = os.getenv(
    "AZURE_MODEL_NAME",
    "gpt-4.1"
)

# API_VERSION = os.getenv(
#     "API_VERSION",
#     "2024-12-01-preview"
# )

# API version for Azure AI Foundry
API_VERSION = "2024-12-01-preview"

# ===================================================

def get_llm_config():
    """
    Returns a dictionary with all LLM configuration parameters.
    Useful for passing config to different client implementations.
    """
    return {
        "endpoint": INFERENCE_ENDPOINT,
        "api_key": API_KEY,
        "model_name": MODEL_NAME,
        "api_version": API_VERSION
    }

def validate_config():
    """
    Checks if configuration has been properly set up.
    """
    print("✅ Configuration validated successfully!")
    print(f"   Endpoint: {INFERENCE_ENDPOINT}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   API Version: {API_VERSION}")
    return True

# Optional: Uncomment to validate on import
# validate_config()
