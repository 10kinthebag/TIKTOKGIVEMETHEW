import gradio as gr
from inference import ReviewClassificationPipeline


pipeline = ReviewClassificationPipeline()


def classify_interface(text):
    if not text.strip():
        return "Please enter some text", 0.0, "N/A", "Empty input"
    result = pipeline.classify(text)
    status = "✅ Valid Review" if result["is_valid"] else "❌ Invalid Review"
    return status, float(result["confidence"]), result["method"].replace("_", " ").title(), result["reason"]


demo = gr.Interface(
    fn=classify_interface,
    inputs=[gr.Textbox(label="Review Text", placeholder="Enter a restaurant review...")],
    outputs=[
        gr.Textbox(label="Classification Result"),
        gr.Number(label="Confidence Score"),
        gr.Textbox(label="Method Used"),
        gr.Textbox(label="Reason"),
    ],
    title="🍽️ Restaurant Review Classifier",
    description="Hybrid rules + ML classifier",
)


if __name__ == "__main__":
    demo.launch(share=True)


