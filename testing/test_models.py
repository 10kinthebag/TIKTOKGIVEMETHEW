#!/usr/bin/env python3
"""
Model Testing Summary - Quick overview of all your testing options
"""

print("""
🚀 MODEL TESTING COMMAND SUMMARY
=================================

📂 You have 5 models available:
   1. combined_training_model ⭐ (NEWEST - 97.73% accuracy)
   2. final_model ⭐ (Previous best)
   3. roberta_policy_based_model ⭐ (Policy-specific)
   4. policy_based_model (Baseline)
   5. intial_model (Historical)

🎯 QUICK TESTING COMMANDS:

1️⃣ EASIEST - Interactive Menu:
   python testing/quick_test.py
   (Works from any directory!)

2️⃣ DIRECT MODEL TESTING:
   python testing/quick_test.py combined_training_model
   python testing/quick_test.py final_model
   python testing/quick_test.py roberta_policy_based_model

3️⃣ INTERACTIVE MODE:
   python testing/quick_test.py combined_training_model --interactive

4️⃣ FULL FEATURES:
   python testing/multi_model_tester.py
   python testing/multi_model_tester.py --model combined_training_model

🎮 TEST MODES:
   • Batch: Quick test with sample reviews
   • Interactive: Type your own reviews
   • Both: Complete testing experience

💡 RECOMMENDATIONS:
   • Start with: combined_training_model (best performance)
   • Use quick menu: python testing/quick_test.py
   • Compare models: Test same review on different models

📖 Full guide: testing/README.md
""")

if __name__ == "__main__":
    import subprocess
    import sys
    
    print("🎯 Would you like to start testing now? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        try:
            subprocess.run([sys.executable, "testing/quick_test.py"], check=True)
        except Exception as e:
            print(f"Error: {e}")
            print("Try running: python testing/quick_test.py")
    else:
        print("👋 Happy testing!")
