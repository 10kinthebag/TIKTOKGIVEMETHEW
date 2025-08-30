from flask import Flask, request, jsonify
from inference import ReviewClassificationPipeline


app = Flask(__name__)
pipeline = ReviewClassificationPipeline()


@app.route("/classify", methods=["POST"])
def classify_review():
    data = request.get_json(force=True)
    text = data.get("text", "")
    result = pipeline.classify(text)
    return jsonify({"status": "success", "result": result})


@app.route("/batch_classify", methods=["POST"])
def batch_classify_reviews():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    results = pipeline.batch_classify(texts)
    return jsonify({"status": "success", "results": results})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model": "loaded"})


if __name__ == "__main__":
    print("🚀 Starting Review Classification API...")
    print("📡 API will be available at: http://localhost:5001")
    print("🔌 Endpoints:")
    print("   POST /classify - Single review classification")
    print("   POST /batch_classify - Batch review classification")
    print("   GET /health - Health check")
    app.run(debug=True, host="0.0.0.0", port=5001)


