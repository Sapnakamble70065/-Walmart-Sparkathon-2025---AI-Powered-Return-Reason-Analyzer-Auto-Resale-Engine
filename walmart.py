import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'showcase'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Models
try:
    model = joblib.load('return_reason_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please ensure both .pkl files are in the same directory.")
    st.stop()

# Stopwords & Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Streamlit Config
st.set_page_config(
    page_title="Return Analyzer Pro",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    .header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .product-card {
        border-radius: 15px;
        padding: 20px;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        margin-bottom: 20px;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .product-image {
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .next-btn {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        padding: 12px 30px !important;
        border-radius: 30px !important;
        margin-top: 20px !important;
    }
    .back-btn {
        background: #f8f9fa !important;
        color: #6c757d !important;
        border: 1px solid #dee2e6 !important;
        font-weight: bold !important;
        padding: 10px 25px !important;
        border-radius: 30px !important;
    }
</style>
""", unsafe_allow_html=True)

# Product Data with new working URLs
products = [
    {
        "name": "Wireless Headphones",
        "image": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80",
        "price": "₹5,999"
    },
    {
        "name": "Smart Watch",
        "image": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80",
        "price": "₹12,499"
    },
    {
        "name": "Leather Wallet",
        "image": "https://images.unsplash.com/photo-1591561954555-607968c989ab?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80",
        "price": "₹1,299"
    },
    {
        "name": "Running Shoes",
        "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80",
        "price": "₹3,499"
    },
    {
        "name": "Cotton T-Shirt",
        "image": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80",
        "price": "₹799"
    },
    {
        "name": "Laptop Backpack",
        "image": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=400&h=400&q=80",
        "price": "₹2,199"
    }
]

# Navigation Functions
def go_to_analysis(product):
    st.session_state.selected_product = product
    st.session_state.page = 'analysis'

def go_to_showcase():
    st.session_state.page = 'showcase'

# Showcase Page
if st.session_state.page == 'showcase':
    st.markdown("<h1 class='header'>📦 Our Product Collection</h1>", unsafe_allow_html=True)
    
    # Display products in 2 rows of 3 columns
    cols = st.columns(3)
    for i, product in enumerate(products):
        if i % 3 == 0 and i != 0:
            cols = st.columns(3)  # New row
        
        with cols[i % 3]:
            st.markdown(f"""
            <div class="product-card">
                <center>
                    <img src="{product['image']}" class="product-image" width="200">
                    <h3>{product['name']}</h3>
                    <p><b>{product['price']}</b></p>
                </center>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Select for Return", key=f"select_{i}"):
                go_to_analysis(product)

# Analysis Page
elif st.session_state.page == 'analysis':
    st.markdown("<h1 class='header'>📝 Return Reason Analysis</h1>", unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Products", key="back_btn", type="secondary", use_container_width=True):
        go_to_showcase()
    
    # Selected product info
    product = st.session_state.selected_product
    st.markdown(f"""
    <div style="background:#f8f9fa; padding:15px; border-radius:10px; margin:15px 0">
        <h4>Selected Product: <b>{product['name']}</b></h4>
        <p>Original Price: {product['price']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Return reason input
    return_reason = st.text_area(
        "Please describe why you're returning this product:",
        placeholder="Example: The product arrived damaged...",
        height=150
    )
    
    # Analysis button
    if st.button("🔍 Analyze Return Reason", type="primary", use_container_width=True):
        if not return_reason.strip():
            st.warning("Please enter a return reason")
        else:
            with st.spinner('Analyzing your reason...'):
                # Preprocess and predict
                cleaned = preprocess_text(return_reason)
                vector = vectorizer.transform([cleaned])
                prediction = model.predict(vector)[0]
                probs = model.predict_proba(vector)[0]
                
                # Display results
                st.success(f"**Predicted Return Category:** {prediction}")
                
                # Confidence visualization
                st.subheader("Confidence Levels")
                prob_df = pd.DataFrame({
                    'Category': model.classes_,
                    'Probability': probs
                }).sort_values('Probability', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(prob_df['Category'], prob_df['Probability'], color='#6e8efb')
                ax.set_xlim(0, 1)
                st.pyplot(fig)
                
                # Recommended action
                st.subheader("Recommended Resolution")
                actions = {
                    "Size Issue": ("🔄 Offer size exchange with free return shipping", "#FFA500"),
                    "Defective": ("🔧 Process replacement with expedited shipping", "#FF3333"),
                    "Color Issue": ("💰 Offer 25% discount to keep the item", "#AA66CC"),
                    "Delivery Issue": ("🚚 Refund shipping costs + 10% discount", "#33B5E5"),
                    "Wrong Item": ("📦 Send correct item + 15% discount", "#FFBB33"),
                    "Performance Issue": ("🛠 Technical support consultation", "#00C851")
                }
                
                action, color = actions.get(prediction, ("ℹ️ Requires manual review", "#999999"))
                st.markdown(
                    f"""<div style="padding:15px; background:#f8f9fa; border-left:5px solid {color}; border-radius:5px">
                    <h4 style="color:{color}; margin:0">{action}</h4>
                    </div>""",
                    unsafe_allow_html=True
                )

# Footer
st.markdown("---")
st.markdown("<center>© 2024 Return Analyzer Pro | Developed with Streamlit</center>", unsafe_allow_html=True)