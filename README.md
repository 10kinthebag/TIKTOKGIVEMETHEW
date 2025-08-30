## **ReviewGuard AI**

This document provides details about the frontend and backend implementation for our ReviewGuard AI project.

[ReviewGuard AI Frontend](#reviewguard-ai-frontend)  
[ReviewGuard AI Backend](#reviewguard-ai-backend)

## **Project Overview:**

ReviewGuard AI is a multimodal review moderation system developed to automatically assess the quality and relevancy of Google location reviews, improving reliability of review platforms and enhancing user experience. It is built using Python, PyTorch, and Hugging Face models, with a Streamlit and Gradio frontend, leveraging data processing libraries, machine learning pipelines, and web technologies for an interactive and efficient user experience.

## **Setup Instructions:**

1. Clone the repository  
     
   git clone [https://github.com/10kinthebag/TIKTOKGIVEMETHEW.git](https://github.com/10kinthebag/TIKTOKGIVEMETHEW.git)

2. Create and activate a virtual environment

   python \-m venv venv

   source venv/bin/activate    \# Mac/Linux

   venv\\Scripts\\activate       \# Windows

3. Install dependencies

pip install \-r requirements.txt

4. Prepare data

- Place your raw dataset (e.g. `reviews.csv`) inside the `data/` folder.  
- Ensure the file has at least the following columns:  
  * `business_name`  
  * `author_name`  
  * `text` (review text)  
  * `rating`  
  * `rating_category`

**To obtain results:** 

Run the review cleaning pipeline: python policy\_module.py data/reviews.csv

### 

### **Output**

* Cleaned dataset will be saved to:  
  * `data/filteredData/cleaned_reviews_<timestamp>.csv`  
* A debug version (with policy violation flags) will be saved to:  
  * `data/filteredDataWithFlags/cleaned_reviews_<timestamp>.csv`

## **Technological Stack**

* **Development Tools**: VSCode, Python, Git & GitHub  
* **APIs**: Hugging Face Sentence Transformers  
* **Libraries**:  
  * `pandas` (data wrangling)  
  * `re` (regex rules for spam/ad detection)  
  * `ftfy`, `unidecode` (text normalization)  
  * `scikit-learn` (evaluation metrics)  
  * `sentence-transformers` (semantic similarity check)  
* **Dataset Assets**: Google Local Reviews dataset \+ manually labeled samples  
* **Machine Learning & NLP:** PyTorch, Sentence-BERT embeddings,  Trainer API, sklearn.metrics  
* **Optimization & Deployment:** Quantization, TorchScript JIT compilation, Image Classification, Flask API, Gradio Interface  
* **Frontend**: Streamlit, HTML & CSS, Plotly, JavaScript

## **Team Contributions**

1. Chermaine (Data Lead): Download dataset(s), clean basic issues (remove nulls, weird formatting) using Pandas.  
2. Nickson (NLP Lead): Integrated HuggingFace pre-trained models (sentence transformers for relevancy checks).  
3. Cheryl & Xavier (Policy Leads): Designed rule-based filters (regex for advertisements, spam length thresholds, rants detection).  
4. Chee Yoong (Evaluation Lead): Set up evaluation metrics (precision, recall, F1) with scikit-learn; created manually labeled test set.

## **ReviewGuard AI Frontend**

ReviewGuard AI is a web application designed to provide professional-grade review analysis for users and businesses. It leverages an interactive Streamlit interface to visualize, classify, and monitor reviews in real time, integrating advanced machine learning models with a rules-based policy engine.

## **Features**

* Analyze single or batch reviews for validity and policy compliance  
* Display confidence scores, detected violations, and model reasoning  
* Live content analysis with dataset sample loading for quick testing  
* Executive dashboard with key metrics: total reviews, average ratings, analysis counts, and system accuracy  
* Visualize content distribution, quality score distributions, and policy violation trends using interactive charts  
* Advanced analytics center for model performance, processing efficiency, and quality distribution  
* Real-time system monitoring for continuous content intelligence  
* Customizable AI settings: detection sensitivity and policy strictness  
* Gamified experience with detailed session history and visual quality indicators  
* Modern, TikTok-inspired professional UI for a sleek and engaging user experience

## **Technologies & Tools Used**

* Streamlit, HTML/CSS, Plotly  
* Pandas, NumPy, Gradio (for demo interface)  
* VS Code

## **Installation & Usage**

Launch Streamlit frontend: streamlit run app.py

## **Design Process**

Inspired by platforms prioritizing professional analytics and human-centered visualization, the UI emphasizes clarity, context, and immersive interactions. Visual gradients, professional cards, and responsive charts guide users in making informed decisions quickly. It is also designed with TikTok colours in mind\!

## **ReviewGuard AI Backend**

Our backend is designed to ensure high-quality, trustworthy review classification by combining multiple data sources, advanced preprocessing, and progressive model training strategies. The system handles everything from ground truth correction to policy-based filtering and hybrid dataset training, enabling robust and accurate predictions.

## **How it works:**

**1\. Data Cleaning and Label Correction**  
Before training, we correct inconsistencies in human-labeled data to match model expectations. For instance, if a friend labels `1=suspicious` but the model expects `1=valid`, our scripts automatically flip the labels and generate a verified dataset. This ensures alignment between the labels and model output, improving both training efficiency and evaluation accuracy.

**2\. Hybrid Dataset Construction**  
 To maximize learning, we combine:

* Ground truth data: High-quality, human-annotated reviews.  
* Pseudo-labeled data: Larger quantities of rule-based labeled reviews.

We tokenize all text using transformer-specific tokenizers (e.g., RoBERTa, DeBERTa) and split the data into training, validation, and test sets. This hybrid approach leverages the accuracy of human labels and the volume of pseudo-labeled data, enhancing generalization.

**3\. Policy-Based Filtering**

We integrate a policy module that filters reviews based on rules such as:

* Advertisement or promotional content  
* Irrelevant reviews  
* Spam or gibberish  
* Contradictions between sentiment and rating  
* Short or rant-like reviews

Filtered datasets are labeled (`1=valid`, `0=invalid`) and combined for training, allowing our models to learn from high-quality, policy-verified data.

**4\. Evaluation and Metrics**  
During and after training, we calculate metrics including accuracy, precision, recall, and F1-score, and generate detailed per-class reports. This allows for robust assessment of model performance across valid and invalid review categories.

**5\. Deployment Readiness**  
Trained models and tokenizers are saved to disk, ensuring smooth deployment. The backend supports retraining with new data, progressive fine-tuning, and policy-driven updates, making it scalable and maintainable.

## **APIs:**

**Hugging Face**  
Hugging Face pre-trained models, particularly Sentence Transformers, are used to compute semantic similarity between reviews. These models help in evaluating the relevancy and trustworthiness of reviews, forming the backbone of the hybrid rules \+ ML classification pipeline.

**Custom Pipeline**  
A custom `ReviewClassificationPipeline` integrates rule-based filters with ML models to classify reviews. It evaluates inputs for spam, advertisement content, rants, or irrelevant information, returning a classification status, confidence score, reasoning, and the method used.

**Tensorflow Keras**  
A pre-trained image classification model that evaluates the relevancy of the images provided by reviewers to the focus or interest of their review.

## **Datasets & Assets:**

## **Libraries & Frameworks:**

**Backend Framework: Flask**  
Flask serves as the microservice framework for our backend. Its RESTful API architecture handles HTTP requests from frontend interfaces (Gradio and Streamlit) and returns classification results in JSON format. Flask’s modular structure allows isolated testing of ML inference, rules, and evaluation modules.

**Machine Learning & NLP**  
PyTorch is used for loading and running ML models, including quantized models for optimized performance. Scikit-learn provides evaluation metrics such as precision, recall, and F1 score for model validation. The backend also integrates semantic similarity checks via Sentence-BERT embeddings.

**Backend Architecture**  
The backend communicates with the frontend via RESTful API calls. All ML inference, rules evaluation, and scoring logic reside within Flask route methods. Gradio and Streamlit frontends interact with these endpoints to provide an interactive classification interface.

## 
