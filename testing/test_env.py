#!/usr/bin/env python3
"""
Test if .env file is loaded correctly
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (override existing ones)
load_dotenv(override=True)

def test_env_loading():
    """Test if the .env file is loaded correctly"""
    
    print("🔧 Testing .env File Loading")
    print("="*40)
    
    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        # Don't print the full key for security
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"✅ OPENAI_API_KEY loaded: {masked_key}")
        print(f"   Key length: {len(api_key)} characters")
        print(f"   Starts with 'sk-': {api_key.startswith('sk-')}")
    else:
        print("❌ OPENAI_API_KEY not found!")
        
    # Check other environment variables
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    print(f"🤖 OPENAI_MODEL: {model}")
    
    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        print(f"🏢 OPENAI_ORG_ID: {org_id}")
    else:
        print("🏢 OPENAI_ORG_ID: Not set")
    
    print()
    print("📁 Environment file locations:")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   .env file exists: {os.path.exists('.env')}")
    
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            lines = f.readlines()
        print(f"   .env file has {len(lines)} lines")
        
        # Check if OPENAI_API_KEY is in the file
        has_api_key = any('OPENAI_API_KEY' in line and not line.strip().startswith('#') for line in lines)
        print(f"   Contains OPENAI_API_KEY: {has_api_key}")
    
    return api_key is not None

if __name__ == "__main__":
    success = test_env_loading()
    
    if success:
        print("\n🎉 Environment setup successful!")
        print("   You can now run the LLM-enhanced classifier")
    else:
        print("\n❌ Environment setup failed!")
        print("   Check your .env file and make sure OPENAI_API_KEY is set")
