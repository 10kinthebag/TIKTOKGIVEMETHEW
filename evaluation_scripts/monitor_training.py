"""
Training Progress Monitor
Monitor the ongoing training and provide real-time updates.
"""

import os
import time
import json
import pandas as pd
from datetime import datetime


def monitor_training_logs(log_dir="./roberta_policy_logs"):
    """Monitor training progress from logs."""
    print("📊 Training Progress Monitor")
    print("=" * 50)
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    print(f"📂 Monitoring logs in: {log_dir}")
    print("🔄 Refreshing every 30 seconds...")
    print("⚡ Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            # Check for training logs
            log_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
            
            if log_files:
                latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(log_dir, f)))
                log_path = os.path.join(log_dir, latest_log)
                
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Latest log: {latest_log}")
                
                # Try to find trainer_state.json for detailed progress
                state_files = [f for f in os.listdir("./roberta_policy_model_results") 
                              if f == "trainer_state.json" and os.path.exists("./roberta_policy_model_results")]
                
                if state_files:
                    try:
                        with open("./roberta_policy_model_results/trainer_state.json", 'r') as f:
                            state = json.load(f)
                            
                        current_epoch = state.get('epoch', 0)
                        best_metric = state.get('best_metric', None)
                        
                        print(f"📈 Current epoch: {current_epoch:.2f}/3.0")
                        if best_metric:
                            print(f"🎯 Best accuracy so far: {best_metric:.1%}")
                        
                        # Show recent log history
                        log_history = state.get('log_history', [])
                        if len(log_history) > 0:
                            recent_logs = log_history[-3:]  # Last 3 entries
                            print("📊 Recent training steps:")
                            for log_entry in recent_logs:
                                if 'loss' in log_entry:
                                    epoch = log_entry.get('epoch', 0)
                                    loss = log_entry.get('loss', 0)
                                    lr = log_entry.get('learning_rate', 0)
                                    print(f"   Epoch {epoch:.1f}: Loss = {loss:.4f}, LR = {lr:.2e}")
                                elif 'eval_accuracy' in log_entry:
                                    epoch = log_entry.get('epoch', 0)
                                    acc = log_entry.get('eval_accuracy', 0)
                                    loss = log_entry.get('eval_loss', 0)
                                    print(f"   📊 Eval Epoch {epoch:.1f}: Accuracy = {acc:.1%}, Loss = {loss:.4f}")
                    
                    except Exception as e:
                        print(f"⚠️ Could not parse trainer state: {e}")
                
            else:
                print(f"⏳ Waiting for training logs to appear...")
            
            print("-" * 50)
            time.sleep(30)  # Wait 30 seconds before next check
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")


def check_training_status():
    """Quick check of current training status."""
    print("🔍 Quick Training Status Check")
    print("=" * 40)
    
    # Check if training is likely running
    model_dirs = ["./roberta_policy_model_results", "./roberta_policy_logs"]
    
    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            recent_files = []
            
            for file in files:
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    # Check if file was modified in last hour
                    mod_time = os.path.getmtime(file_path)
                    if time.time() - mod_time < 3600:  # 1 hour
                        recent_files.append(file)
            
            print(f"📁 {dir_path}:")
            if recent_files:
                print(f"   ✅ {len(recent_files)} recently updated files")
                print(f"   📝 Latest: {recent_files[-1] if recent_files else 'None'}")
            else:
                print(f"   ⚠️ No recent activity")
        else:
            print(f"📁 {dir_path}: ❌ Not found")
    
    # Check final model
    final_model_path = "./models/roberta_policy_based_model"
    if os.path.exists(final_model_path):
        print(f"\n🎉 Final model exists: {final_model_path}")
        model_files = os.listdir(final_model_path)
        print(f"   📁 Contains {len(model_files)} files")
        
        # Check model size
        config_path = os.path.join(final_model_path, "config.json")
        if os.path.exists(config_path):
            print(f"   ✅ Model configuration found")
        
        pytorch_model = os.path.join(final_model_path, "pytorch_model.bin")
        safetensors_model = os.path.join(final_model_path, "model.safetensors")
        
        if os.path.exists(pytorch_model):
            size_mb = os.path.getsize(pytorch_model) / (1024 * 1024)
            print(f"   📊 Model size: {size_mb:.1f} MB")
        elif os.path.exists(safetensors_model):
            size_mb = os.path.getsize(safetensors_model) / (1024 * 1024)
            print(f"   📊 Model size: {size_mb:.1f} MB")
            
        print(f"   🚀 Ready for testing!")
    else:
        print(f"\n⏳ Final model not ready yet: {final_model_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor_training_logs()
    else:
        check_training_status()
        print("\n💡 Use 'python monitor_training.py --monitor' for live monitoring")
