import time
from hybrid_pipeline import ReviewClassificationPipeline


def benchmark_pipeline(pipeline, test_texts, n=100):
    start = time.time()
    for text in test_texts[:n]:
        pipeline.classify(text)
    total = time.time() - start
    print(f"Processed {n} items in {total:.3f}s -> {total/n*1000:.2f} ms/item")


if __name__ == "__main__":
    pipeline = ReviewClassificationPipeline()
    examples = [
        "Great food and excellent service!",
        "Visit www.example.com for deals",
        "Good",
        "The staff was friendly and the atmosphere cozy",
        "Click here now!",
    ] * 40
    benchmark_pipeline(pipeline, examples, n=100)


