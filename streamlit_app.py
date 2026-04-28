import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px

# ---- Page config ----
st.set_page_config(
    page_title="Indian Food Classifier",
    page_icon="🍛",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://fastapi:8000"

# ---- Header ----
st.markdown('<p class="main-header">🍛 Indian Food Classifier Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an Indian food image to classify it using EfficientNet-B0</p>',
            unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.title("🍽️ About")
    st.markdown("""
    ### Model Info
    - **Architecture:** EfficientNet-B0
    - **Dataset:** Indian Food Images
    - **Classes:** 15 Indian dishes
    - **Test Accuracy:** 90.91%
    - **Val Accuracy:** 92.61%
    """)

    st.markdown("---")
    st.markdown("### 🍱 Supported Classes")
    classes = [
        '🍚 Idli', '🥞 Masala Dosa', '🟡 Dhokla',
        '🫙 Paani Puri', '🧆 Pakode', '☕ Chai',
        '🥟 Samosa', '🍲 Pav Bhaji', '🍳 Fried Rice',
        '🍩 Jalebi', '🫓 Butter Naan', '🍛 Kadai Paneer',
        '🍦 Kulfi', '🫓 Chapati', '🍛 Chole Bhature'
    ]
    for cls in classes:
        st.markdown(f"- {cls}")

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    comparison_data = {
        'Model': ['MLP', 'CNN', 'EfficientNet'],
        'Test Acc': [13.35, 58.95, 90.91]
    }
    fig_sidebar = px.bar(
        comparison_data,
        x='Model',
        y='Test Acc',
        color='Test Acc',
        color_continuous_scale='Oranges',
        title='Model Accuracy (%)'
    )
    fig_sidebar.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_sidebar, use_container_width=True)

# ---- Main content ----
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Food Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of Indian food"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",
                 use_column_width=True)

        if st.button("🔍 Classify Food",
                     type="primary",
                     use_container_width=True):
            with st.spinner("🤖 Analyzing your food image..."):
                try:
                    img_bytes = uploaded_file.getvalue()
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
                    )
                    result = response.json()
                    st.session_state['result'] = result
                    st.success("✅ Classification complete!")
                except Exception as e:
                    st.error(f"❌ Error connecting to API: {e}")

with col2:
    st.markdown("### 🎯 Prediction Results")

    if 'result' in st.session_state:
        result = st.session_state['result']

        # Main prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🍽️ {result['prediction'].replace('_', ' ').title()}</h2>
            <h3>Confidence: {result['confidence']*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 3 predictions
        st.markdown("#### 🏆 Top 3 Predictions")
        medals = ['🥇', '🥈', '🥉']
        for i, pred in enumerate(result['top3']):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(float(pred['confidence']))
                st.caption(
                    f"{medals[i]} {pred['class'].replace('_', ' ').title()}"
                )
            with col_b:
                st.metric("", f"{pred['confidence']*100:.1f}%")

        st.markdown("---")

        # Full probability chart
        st.markdown("#### 📊 All Class Probabilities")
        all_probs = result['all_probs']
        sorted_probs = dict(
            sorted(all_probs.items(),
                   key=lambda x: x[1], reverse=True)
        )

        fig = go.Figure(go.Bar(
            x=list(sorted_probs.values()),
            y=[k.replace('_', ' ').title()
               for k in sorted_probs.keys()],
            orientation='h',
            marker=dict(
                color=list(sorted_probs.values()),
                colorscale='Oranges',
                showscale=False
            )
        ))
        fig.update_layout(
            title='Confidence Scores for All Classes',
            xaxis_title='Confidence',
            xaxis=dict(range=[0, 1]),
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Upload an image and click 'Classify Food' to see results")
        st.markdown("""
        #### How it works:
        1. 📤 Upload a clear image of Indian food
        2. 🔍 Click the **Classify Food** button
        3. 🎯 See the predicted class with confidence
        4. 📊 Explore probabilities for all 15 classes
        5. 🏆 Check top 3 predictions
        """)

# ---- Footer metrics ----
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Test Accuracy", "90.91%")
with col2:
    st.metric("📈 Val Accuracy", "92.61%")
with col3:
    st.metric("🍽️ Food Classes", "15")
with col4:
    st.metric("🏗️ Architecture", "EfficientNet-B0")