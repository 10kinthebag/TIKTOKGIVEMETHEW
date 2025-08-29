import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Import your modules with proper error handling
try:
    from src.policy_module import (
        detect_advertisement,
        detect_irrelevant,
        detect_rant_without_visit,
        detect_contradiction,
        detect_short_review,
        detect_spam_content,
        apply_policy_rules
    )
    POLICY_MODULE_LOADED = True
except ImportError as e:
    st.error(f"❌ Could not import policy_module from src/: {str(e)}")
    st.error("Make sure your policy functions are in src/policy_module.py")
    POLICY_MODULE_LOADED = False

try:
    from eval.metrics import (
        calculate_metrics,
        load_ground_truth,
        evaluate_predictions
    )
    METRICS_MODULE_LOADED = True
except ImportError as e:
    st.warning(f"⚠️ Could not import eval.metrics: {str(e)}")
    METRICS_MODULE_LOADED = False

# Page configuration
st.set_page_config(
    page_title="TikTok TechJam - Review Quality Assessment",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff0050, #ff4081, #e91e63);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .policy-violation {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .policy-clean {
        background: linear-gradient(135deg, #51cf66, #40c057);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #ff0050, #ff4081);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the reviews dataset"""
    try:
        df = pd.read_csv('reviews_cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("❌ reviews_cleaned.csv not found! Please make sure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def analyze_review_quality(text, rating=None, location=""):
    """Analyze review quality using your policy module"""
    if not POLICY_MODULE_LOADED:
        return {
            'overall_quality': 'Unknown',
            'confidence': 0.0,
            'violations': ['Policy module not loaded'],
            'policy_scores': {}
        }
    
    results = {
        'overall_quality': 'High',
        'confidence': 0.85,
        'violations': [],
        'policy_scores': {}
    }
    
    try:
        # Use your actual policy detection functions
        is_ad = detect_advertisement(text)
        is_irrelevant = detect_irrelevant(text)
        is_rant = detect_rant_without_visit(text)
        is_spam = detect_spam_content(text)
        is_short = detect_short_review(text)
        
        # Check contradiction if rating is provided
        is_contradiction = False
        if rating is not None:
            is_contradiction = detect_contradiction(text, rating)
        
        # Compile violations
        violations = []
        if is_ad:
            violations.append("Advertisement: Promotional content detected")
        if is_irrelevant:
            violations.append("Irrelevant: Content not related to location")
        if is_rant:
            violations.append("Rant w/o visit: Review from non-visitor detected")
        if is_spam:
            violations.append("Spam: Spam patterns detected")
        if is_short:
            violations.append("Short review: Review too brief")
        if is_contradiction:
            violations.append("Contradiction: Rating doesn't match sentiment")
        
        # Calculate overall quality based on violations
        violation_count = len(violations)
        if violation_count == 0:
            results['overall_quality'] = 'High'
            results['confidence'] = 0.92
        elif violation_count <= 2:
            results['overall_quality'] = 'Medium'
            results['confidence'] = 0.75
        else:
            results['overall_quality'] = 'Low'
            results['confidence'] = 0.88
        
        results['violations'] = violations
        results['policy_scores'] = {
            'No Advertisement': 0.1 if is_ad else 0.95,
            'Relevant Content': 0.2 if is_irrelevant else 0.90,
            'Genuine Visit': 0.15 if is_rant else 0.88,
            'No Spam': 0.1 if is_spam else 0.93,
            'Sufficient Length': 0.3 if is_short else 0.85
        }
        
        if rating is not None:
            results['policy_scores']['Consistent Rating'] = 0.2 if is_contradiction else 0.88
            
    except Exception as e:
        st.error(f"Error in policy analysis: {str(e)}")
        results['violations'] = [f"Analysis error: {str(e)}"]
    
    return results

@st.cache_data
def analyze_dataset(df, text_col, rating_col=None):
    """Analyze the entire dataset using your policy module"""
    if not POLICY_MODULE_LOADED:
        return {
            'total_reviews': len(df),
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': len(df),
            'violations': {k: 0 for k in ['advertisement', 'irrelevant', 'rant', 'spam', 'short', 'contradiction']},
            'policy_scores': {}
        }
    
    results = {
        'total_reviews': len(df),
        'high_quality': 0,
        'medium_quality': 0,
        'low_quality': 0,
        'violations': {
            'advertisement': 0,
            'irrelevant': 0,
            'rant': 0,
            'spam': 0,
            'short': 0,
            'contradiction': 0
        },
        'policy_scores': {
            'No Advertisement': [],
            'Relevant Content': [],
            'Genuine Visit': [],
            'No Spam': [],
            'Sufficient Length': []
        }
    }
    
    if rating_col:
        results['policy_scores']['Consistent Rating'] = []
    
    # Analyze each review
    for idx, row in df.iterrows():
        if pd.isna(row[text_col]):
            results['low_quality'] += 1
            continue
            
        try:
            rating = row[rating_col] if rating_col and rating_col in df.columns else None
            analysis = analyze_review_quality(str(row[text_col]), rating)
            
            # Count quality levels
            if analysis['overall_quality'] == 'High':
                results['high_quality'] += 1
            elif analysis['overall_quality'] == 'Medium':
                results['medium_quality'] += 1
            else:
                results['low_quality'] += 1
            
            # Count violations using actual detection results
            text_str = str(row[text_col])
            if detect_advertisement(text_str):
                results['violations']['advertisement'] += 1
            if detect_irrelevant(text_str):
                results['violations']['irrelevant'] += 1
            if detect_rant_without_visit(text_str):
                results['violations']['rant'] += 1
            if detect_spam_content(text_str):
                results['violations']['spam'] += 1
            if detect_short_review(text_str):
                results['violations']['short'] += 1
            if rating and detect_contradiction(text_str, rating):
                results['violations']['contradiction'] += 1
            
            # Store policy scores
            for policy, score in analysis['policy_scores'].items():
                if policy in results['policy_scores']:
                    results['policy_scores'][policy].append(score)
                    
        except Exception as e:
            st.warning(f"Error analyzing review at index {idx}: {str(e)}")
            results['low_quality'] += 1
    
    return results

def calculate_performance_metrics(df, text_col, rating_col=None):
    """Calculate performance metrics using your eval.metrics module"""
    if not METRICS_MODULE_LOADED:
        return {
            'has_ground_truth': False,
            'error': 'eval.metrics module not available'
        }
    
    try:
        # Load ground truth data using your module
        ground_truth = load_ground_truth()
        
        # Generate predictions using your policy module
        predictions = []
        for idx, row in df.iterrows():
            if pd.isna(row[text_col]):
                predictions.append(0)  # Assume clean if no text
                continue
                
            text_str = str(row[text_col])
            rating = row[rating_col] if rating_col and rating_col in df.columns else None
            
            # Check if any policy violation exists
            has_violation = (
                detect_advertisement(text_str) or
                detect_irrelevant(text_str) or
                detect_rant_without_visit(text_str) or
                detect_spam_content(text_str) or
                detect_short_review(text_str) or
                (rating and detect_contradiction(text_str, rating))
            )
            predictions.append(1 if has_violation else 0)
        
        # Calculate metrics using your eval module
        metrics = evaluate_predictions(ground_truth, predictions)
        
        return {
            'has_ground_truth': True,
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'false_positives': metrics.get('false_positives', 0),
            'false_negatives': metrics.get('false_negatives', 0)
        }
        
    except Exception as e:
        st.warning(f"Could not calculate metrics using eval.metrics: {str(e)}")
        return {
            'has_ground_truth': False,
            'error': str(e)
        }

def main():
    """Main application function"""
    # Check if required modules are loaded
    if not POLICY_MODULE_LOADED:
        st.error("⚠️ Policy module is not loaded. Please check your src/policy_module.py file.")
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 TikTok TechJam 2024</h1>
        <h2>AI-Powered Review Quality Assessment System</h2>
        <p>Detecting Policy Violations • Ensuring Review Authenticity • Enhancing User Trust</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Detect columns
    text_columns = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['review', 'text', 'content', 'comment'])]
    text_col = text_columns[0] if text_columns else df.columns[0]
    
    rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'star' in col.lower()]
    rating_col = rating_cols[0] if rating_cols else None
    
    location_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['location', 'place', 'business', 'name'])]
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox("Choose Demo Section", [
        "📊 Executive Dashboard",
        "🔍 Live Review Analysis", 
        "📈 Policy Analytics",
        "🎯 Batch Processing",
        "🏆 Model Performance"
    ])
    
    # Display dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.info(f"""
    **Total Reviews:** {len(df):,}
    **Columns:** {len(df.columns)}
    **Text Column:** {text_col}
    **Rating Column:** {rating_col or 'None detected'}
    **Data Source:** reviews_cleaned.csv
    """)
    
    # Module status
    st.sidebar.markdown("### 🔧 Module Status")
    st.sidebar.success("✅ Policy Module Loaded" if POLICY_MODULE_LOADED else "❌ Policy Module Error")
    st.sidebar.success("✅ Metrics Module Loaded" if METRICS_MODULE_LOADED else "⚠️ Metrics Module Missing")
    
    # Route to appropriate page
    if page == "📊 Executive Dashboard":
        show_dashboard(df, text_col, rating_col, location_cols)
    elif page == "🔍 Live Review Analysis":
        show_live_analysis(df, text_col, rating_col, location_cols)
    elif page == "📈 Policy Analytics":
        show_policy_analytics(df, text_col, rating_col)
    elif page == "🎯 Batch Processing":
        show_batch_processing(df, text_col, rating_col)
    elif page == "🏆 Model Performance":
        show_model_performance(df, text_col, rating_col)

def show_dashboard(df, text_col, rating_col, location_cols):
    """Executive dashboard view"""
    st.header("📊 Executive Dashboard")
    
    with st.spinner("🔍 Analyzing your dataset..."):
        analysis_results = analyze_dataset(df, text_col, rating_col)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Reviews</h3>
            <h2>{analysis_results['total_reviews']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_length = df[text_col].str.len().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Review Length</h3>
            <h2>{avg_length:.0f} chars</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_quality_pct = (analysis_results['high_quality'] / analysis_results['total_reviews']) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>High Quality</h3>
            <h2>{high_quality_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_violations = sum(analysis_results['violations'].values())
        violation_pct = (total_violations / analysis_results['total_reviews']) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Policy Violations</h3>
            <h2>{violation_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Review Length Distribution")
        lengths = df[text_col].str.len().dropna()
        fig = px.histogram(x=lengths, nbins=50, title="Character Count Distribution")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quality Assessment Results")
        quality_data = {
            'Quality': ['High Quality', 'Medium Quality', 'Low Quality'],
            'Count': [
                analysis_results['high_quality'], 
                analysis_results['medium_quality'], 
                analysis_results['low_quality']
            ]
        }
        fig = px.pie(values=quality_data['Count'], names=quality_data['Quality'],
                    color_discrete_sequence=['#51cf66', '#ffd43b', '#ff8787'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_live_analysis(df, text_col, rating_col, location_cols):
    """Live review analysis view"""
    st.header("🔍 Live Review Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Enter Review for Analysis")
        user_review = st.text_area("Paste a review here:", height=150, 
                                  placeholder="Enter a review to analyze for policy violations...")
        
        location_context = ""
        if location_cols:
            location_context = st.text_input("Location/Business Name (optional):", 
                                           placeholder="e.g., McDonald's, Central Park")
        
        user_rating = st.selectbox("Review Rating (optional):", 
                                 options=[None, 1, 2, 3, 4, 5],
                                 format_func=lambda x: "Not specified" if x is None else f"{x} stars")
    
    with col2:
        st.subheader("🎲 Quick Test Samples")
        sample_reviews = df[text_col].dropna()
        if len(sample_reviews) > 0:
            if st.button("🔄 Random Sample"):
                user_review = sample_reviews.sample(1).iloc[0]
                st.rerun()
    
    if user_review and user_review.strip():
        st.markdown("---")
        
        with st.spinner("🤖 Analyzing review quality..."):
            results = analyze_review_quality(user_review, user_rating, location_context)
        
        # Results display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality_color = {"High": "#51cf66", "Medium": "#ffd43b", "Low": "#ff6b6b"}[results['overall_quality']]
            st.markdown(f"""
            <div style="background: {quality_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                <h3>Overall Quality</h3>
                <h2>{results['overall_quality']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: #667eea; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                <h3>Confidence</h3>
                <h2>{results['confidence']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            violation_count = len(results['violations'])
            violation_color = "#51cf66" if violation_count == 0 else "#ff6b6b"
            st.markdown(f"""
            <div style="background: {violation_color}; padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                <h3>Violations</h3>
                <h2>{violation_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Policy scores
        if results['policy_scores']:
            st.subheader("📋 Policy Compliance Scores")
            for policy, score in results['policy_scores'].items():
                progress_color = "#51cf66" if score > 0.7 else "#ff6b6b"
                st.markdown(f"**{policy}**")
                st.progress(score)
                st.markdown(f"<span style='color: {progress_color}'>Score: {score:.1%}</span>", 
                           unsafe_allow_html=True)
        
        # Violations detail
        if results['violations']:
            st.subheader("⚠️ Policy Violations Detected")
            for violation in results['violations']:
                st.markdown(f"""
                <div class="policy-violation">
                    🚨 {violation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="policy-clean">
                ✅ No policy violations detected! This review appears to be authentic and relevant.
            </div>
            """, unsafe_allow_html=True)

def show_policy_analytics(df, text_col, rating_col):
    """Policy analytics view"""
    st.header("📈 Policy Analytics Deep Dive")
    
    with st.spinner(f"🔍 Analyzing all {len(df)} reviews for policy violations..."):
        analysis_results = analyze_dataset(df, text_col, rating_col)
    
    total_reviews = analysis_results['total_reviews']
    
    # Policy violation rates chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Policy Violation Rates")
        violation_types = ['Advertisement', 'Irrelevant', 'Rant w/o Visit', 'Spam', 'Short Review']
        violation_counts = [
            analysis_results['violations']['advertisement'],
            analysis_results['violations']['irrelevant'],
            analysis_results['violations']['rant'],
            analysis_results['violations']['spam'],
            analysis_results['violations']['short']
        ]
        
        if rating_col:
            violation_types.append('Rating Contradiction')
            violation_counts.append(analysis_results['violations']['contradiction'])
        
        violation_rates = [count / total_reviews * 100 for count in violation_counts]
        
        fig = px.bar(x=violation_types, y=violation_rates,
                    title="Policy Violation Rates (%)",
                    color=violation_types)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Quality Distribution")
        quality_data = {
            'Quality': ['High', 'Medium', 'Low'],
            'Count': [
                analysis_results['high_quality'],
                analysis_results['medium_quality'],
                analysis_results['low_quality']
            ]
        }
        fig = px.pie(values=quality_data['Count'], names=quality_data['Quality'],
                    color_discrete_sequence=['#51cf66', '#ffd43b', '#ff8787'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_batch_processing(df, text_col, rating_col):
    """Batch processing view"""
    st.header("🎯 Batch Processing & Results")
    
    st.info(f"""
    **Current Dataset:** reviews_cleaned.csv
    **Total Reviews:** {len(df):,}
    **Text Column:** {text_col}
    **Rating Column:** {rating_col or 'None detected'}
    **Status:** ✅ Ready for processing
    """)
    
    if st.button("🚀 Process All Reviews", type="primary"):
        with st.spinner("Processing all reviews..."):
            results = analyze_dataset(df, text_col, rating_col)
        
        st.success("✅ Processing completed!")
        
        # Display summary results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Quality", f"{results['high_quality']:,}", 
                     f"{results['high_quality']/len(df)*100:.1f}%")
        
        with col2:
            st.metric("Medium Quality", f"{results['medium_quality']:,}",
                     f"{results['medium_quality']/len(df)*100:.1f}%")
        
        with col3:
            st.metric("Low Quality", f"{results['low_quality']:,}",
                     f"{results['low_quality']/len(df)*100:.1f}%")

def show_model_performance(df, text_col, rating_col):
    """Model performance view"""
    st.header("🏆 Model Performance Metrics")
    
    with st.spinner("📊 Calculating performance metrics..."):
        metrics_results = calculate_performance_metrics(df, text_col, rating_col)
        analysis_results = analyze_dataset(df, text_col, rating_col)
    
    if metrics_results['has_ground_truth']:
        st.success("✅ Using ground truth data from eval.metrics module!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics_results['accuracy']:.1%}")
        
        with col2:
            st.metric("Precision", f"{metrics_results['precision']:.1%}")
        
        with col3:
            st.metric("Recall", f"{metrics_results['recall']:.1%}")
        
        with col4:
            st.metric("F1-Score", f"{metrics_results['f1_score']:.1%}")
        
        # Additional metrics
        st.subheader("📋 Detailed Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**False Positives:** {metrics_results['false_positives']}")
        
        with col2:
            st.info(f"**False Negatives:** {metrics_results['false_negatives']}")
    
    else:
        st.warning("⚠️ Ground truth data not available. Showing violation detection results.")
        
        total_reviews = analysis_results['total_reviews']
        total_violations = sum(analysis_results['violations'].values())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Clean Reviews", f"{total_reviews - total_violations:,}",
                     f"{(total_reviews - total_violations)/total_reviews*100:.1f}%")
        
        with col2:
            st.metric("Violations Detected", f"{total_violations:,}",
                     f"{total_violations/total_reviews*100:.1f}%")

if __name__ == "__main__":
    main()