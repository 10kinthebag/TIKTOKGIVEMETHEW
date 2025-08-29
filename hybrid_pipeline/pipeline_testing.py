from hybrid_pipeline import ReviewClassificationPipeline


def main():
    pipeline = ReviewClassificationPipeline()
    test_reviews = [
        "Visit my website www.example.com for great deals!",
        "Never been here but heard it's bad",
        "Click here for amazing discounts!!!",
        "Great food and excellent service. The pasta was delicious!",
        "Had a wonderful experience. Staff was friendly and atmosphere was cozy.",
        "Good",
        "The restaurant was amazing! Visit www.spam.com",
    ]

    print("=== Pipeline Testing ===")
    for i, review in enumerate(test_reviews):
        result = pipeline.classify(review)
        print(f"\n{i+1}. Review: {review}")
        print(f"   Result: {'✅ Valid' if result['is_valid'] else '❌ Invalid'}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Method: {result['method']}")
        print(f"   Reason: {result['reason']}")


if __name__ == "__main__":
    main()


