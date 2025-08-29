import time
try:
    # Try relative import first (when run from training_scripts dir)
    from trainer_setup import get_trainer
except ImportError:
    # Fall back to absolute import (when run from root dir)
    from training_scripts.trainer_setup import get_trainer


def main():
    print("🚀 Starting training...")
    start_time = time.time()

    trainer = get_trainer()
    trainer.train()

    end_time = time.time()
    print(f"✅ Training completed in {(end_time - start_time)/60:.2f} minutes")

    trainer.save_model("./final_model")
    if trainer.tokenizer:
        trainer.tokenizer.save_pretrained("./final_model")
    print("✅ Model saved to ./final_model")


if __name__ == "__main__":
    main()


