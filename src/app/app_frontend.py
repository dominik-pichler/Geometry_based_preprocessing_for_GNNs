import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Enterprise ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
custom_css = """
<style>
/* Full-width layout adjustments */
.css-1d391kg { padding: 0 !important; }
.css-18e3th9 { padding: 0 !important; }
.css-1v3fvcr { max-width: 100%; padding: 0 !important; }

/* Navbar styling */
.sidebar .sidebar-content {
    background-color: #0A2647;
    color: white;
}
.sidebar .sidebar-content a {
    color: white;
    text-decoration: none;
    font-weight: bold;
}
.sidebar .sidebar-content a:hover {
    color: #FFC107;
}

/* Button styling */
.stButton>button {
    background-color: #205295;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #2C74B3;
}

/* Card styling */
.card {
    background-color: #F5F5F5;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")
    st.markdown("[Home](#home)", unsafe_allow_html=True)
    st.markdown("[Features](#features)", unsafe_allow_html=True)
    st.markdown("[About](#about)", unsafe_allow_html=True)
    st.markdown("[Contact](#contact)", unsafe_allow_html=True)

# Hero Section
st.markdown("<h1 style='text-align:center;'>Anti-Money Laundering ML Inference Platform 💸</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Fight (smurf-based) money laundering through modern neuro-symbolic AI.</p>",
    unsafe_allow_html=True,
)

# Main Interface Section
st.markdown("### Model Configuration")
st.write("This application is based on modern graph neural networks ")


st.markdown("#### Select Model Version")
model_version = st.selectbox(
    "Choose a model version:",
    ["Enterprise Small", "Enterprise Medium", "Enterprise Large"],
    index=1,
    key="model_version",
    help="Select the version of the model you want to use."
)
st.markdown("#### Upload Data")
uploaded_file = st.file_uploader(
    "Drag and drop or browse to upload your file",
    type=["csv", "xlsx", "json"],
    key="file_upload",
    help="Upload your dataset for inference.",
    label_visibility="collapsed"
)

st.markdown("### Results Dashboard")

if st.button("Run Inference", key="run_inference"):
    if uploaded_file:
        with st.spinner("Running inference..."):
            import time
            time.sleep(2)  # Simulate processing time
        st.success("Inference completed successfully!")
        st.write("Results will be displayed here.")
    else:
        st.error("Please upload a file to run inference.")

# Features Section
st.markdown("<h2 id='features'>Features</h2>", unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    st.markdown(
        """
        <div class="card">
        <h4>Enterprise Ready</h4>
        <p>Built for scale with enterprise-grade security and compliance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div class="card">
        <h4>Cloud Native</h4>
        <p>Leveraging cloud infrastructure for optimal performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        """
        <div class="card">
        <h4>API First</h4>
        <p>RESTful APIs for seamless integration with your stack.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# About Section
st.markdown("<h2 id='about'>About Our Platform</h2>", unsafe_allow_html=True)
st.write("""
Our Enterprise ML Platform is designed to meet the demanding needs of modern businesses.
With a focus on reliability, security, and performance, we provide a robust solution for enterprise-scale machine learning deployment.
""")

# Footer Section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>&copy; 2024 Enterprise ML Platform. All rights reserved.</p>",
    unsafe_allow_html=True,
)
