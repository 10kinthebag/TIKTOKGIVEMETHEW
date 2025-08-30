#!/usr/bin/env python3
"""
Diagnose OpenAI API issues
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load environment variables
load_dotenv(override=True)

def diagnose_openai():
    """Diagnose OpenAI API status"""
    
    print("🔍 OpenAI API Diagnostic")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"   Length: {len(api_key)} characters")
    
    # Initialize client
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return
    
    # Test with a minimal completion
    print("\n🧪 Testing minimal API call...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model
            messages=[
                {"role": "user", "content": "Say 'test'"}
            ],
            max_tokens=1
        )
        print("✅ API call successful!")
        print(f"   Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        
        # Parse the error
        error_str = str(e)
        if "429" in error_str:
            print("\n💡 Error Analysis:")
            print("   - This is a rate limit/quota error")
            print("   - Your account may need billing setup")
            print("   - Free trial credits may be exhausted")
        elif "401" in error_str:
            print("\n💡 Error Analysis:")
            print("   - Invalid API key")
            print("   - Check if key is correctly formatted")
        else:
            print(f"\n💡 Unexpected error: {error_str}")
    
    # Try to get account info
    print("\n🏢 Checking account status...")
    try:
        # This endpoint doesn't exist in newer API versions
        # but we can try to get models which requires valid auth
        models = client.models.list()
        print(f"✅ Account active - {len(models.data)} models available")
        
        # Check if gpt-4 is available
        gpt4_available = any(model.id.startswith('gpt-4') for model in models.data)
        print(f"   GPT-4 access: {'✅ Yes' if gpt4_available else '❌ No'}")
        
    except Exception as e:
        print(f"❌ Account check failed: {e}")

if __name__ == "__main__":
    diagnose_openai()
