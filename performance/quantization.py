import torch
from transformers import AutoModelForSequenceClassification
import os


def quantize_model(model_path="./final_model"):
    """Attempt to quantize model for faster CPU inference."""
    try:
        print("🔄 Loading model for quantization...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print("🔧 Attempting dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Save quantized model
        quantized_path = f"{model_path}/quantized_model.pth"
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"✅ Model quantized and saved to {quantized_path}")
        
        # Calculate size reduction
        original_size = os.path.getsize(f"{model_path}/pytorch_model.bin") / (1024 * 1024)
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"📊 Size reduction: {original_size:.1f}MB → {quantized_size:.1f}MB")
        
        return quantized_model
        
    except RuntimeError as e:
        if "NoQEngine" in str(e):
            print("⚠️ Quantization not available on this system (common on macOS)")
            print("💡 Alternative optimizations available:")
            print("   1. Use torch.jit.script() for JIT compilation")
            print("   2. Enable mixed precision training")
            print("   3. Use smaller model variants")
            return None
        else:
            print(f"❌ Quantization failed: {e}")
            return None
    except Exception as e:
        print(f"❌ Unexpected error during quantization: {e}")
        return None


def alternative_optimization(model_path="./final_model"):
    """Alternative optimization methods when quantization isn't available."""
    try:
        print("🔄 Attempting JIT compilation...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create a dummy input for tracing
        dummy_input = torch.randint(0, 1000, (1, 256))  # batch_size=1, seq_len=256
        attention_mask = torch.ones_like(dummy_input)
        
        # Trace the model
        traced_model = torch.jit.trace(model, (dummy_input, attention_mask))
        
        # Save traced model
        traced_path = f"{model_path}/traced_model.pt"
        torch.jit.save(traced_model, traced_path)
        print(f"✅ JIT-compiled model saved to {traced_path}")
        
        return traced_model
        
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Starting model optimization...")
    
    # Try quantization first
    quantized = quantize_model()
    
    if quantized is None:
        print("\n🔄 Falling back to alternative optimization...")
        alternative_optimization()
    
    print("\n✅ Optimization process completed!")


