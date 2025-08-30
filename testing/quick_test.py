#!/usr/bin/env python3
"""
Quick Model Selector - Easy way to choose and test your models
Usage: python quick_test.py [model_name] [--interactive]
"""

import sys
import os
import json
from pathlib import Path

# Add the testing directory to path
sys.path.append(str(Path(__file__).parent))

from multi_model_tester import ModelManager, run_interactive_testing, run_batch_testing


def load_config():
    """Load model configuration"""
    config_path = Path(__file__).parent / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def show_quick_menu():
    """Show quick selection menu"""
    config = load_config()
    models_config = config.get('models', {})
    
    print("🚀 Quick Model Tester")
    print("=" * 40)
    print("\nAvailable models:")
    
    # Sort by priority
    sorted_models = sorted(
        models_config.items(), 
        key=lambda x: x[1].get('priority', 99)
    )
    
    for i, (name, info) in enumerate(sorted_models, 1):
        status = "⭐ RECOMMENDED" if info.get('recommended', False) else ""
        print(f"{i}. {name} {status}")
        print(f"   {info.get('description', 'No description')}")
    
    print(f"\n{len(sorted_models) + 1}. Show all models (detailed view)")
    print(f"{len(sorted_models) + 2}. Exit")
    
    return [name for name, _ in sorted_models]


def main():
    """Main function"""
    args = sys.argv[1:]
    
    # Determine the correct models directory path
    current_dir = Path.cwd()
    if current_dir.name == "testing":
        models_dir = "../models"
    else:
        models_dir = "./models"
    
    # Check for command line model name
    if args and not args[0].startswith('--'):
        model_name = args[0]
        interactive_mode = '--interactive' in args
        
        # Quick test with specified model
        model_manager = ModelManager(models_dir)
        if model_manager.load_model(model_name):
            if interactive_mode:
                run_interactive_testing(model_manager)
            else:
                run_batch_testing(model_manager)
        return
    
    # Interactive menu
    model_names = show_quick_menu()
    
    try:
        choice = input(f"\nSelect option (1-{len(model_names) + 2}): ").strip()
        
        if choice.isdigit():
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(model_names):
                # Quick test selected model
                model_name = model_names[choice_num - 1]
                print(f"\n🎯 Testing model: {model_name}")
                
                model_manager = ModelManager(models_dir)
                if model_manager.load_model(model_name):
                    
                    # Ask for test mode
                    print("\nTest mode:")
                    print("1. Quick batch test")
                    print("2. Interactive testing")
                    print("3. Both")
                    
                    mode_choice = input("Choose mode (1-3): ").strip()
                    
                    if mode_choice == "1":
                        run_batch_testing(model_manager)
                    elif mode_choice == "2":
                        run_interactive_testing(model_manager)
                    else:
                        run_batch_testing(model_manager)
                        run_interactive_testing(model_manager)
            
            elif choice_num == len(model_names) + 1:
                # Show detailed view
                from multi_model_tester import main as detailed_main
                detailed_main()
            
            elif choice_num == len(model_names) + 2:
                print("👋 Goodbye!")
            
            else:
                print("❌ Invalid choice")
        
        else:
            print("❌ Please enter a number")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
