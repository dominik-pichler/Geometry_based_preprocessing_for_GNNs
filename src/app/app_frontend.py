import streamlit as st
import pandas as pd
# Set page configuration
st.set_page_config(
    page_title="Money Laundering Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
custom_css = """
<style>

div.block-container {
    max-width: 1200px; /* Adjust this value to control the width */
    padding-left: 2rem;
    padding-right: 2rem;
}
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
st.markdown("<h1 style='text-align:center;'>Money Laundering Detector </h1>", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

# Main Interface Section
st.markdown("### About")
st.markdown("This application aims at identifying money laundering transactions in arbitray large networks of banking transactions. "
            "<br>It is based on modern graph neural networks and geometry-based proprocessing and in it's current form, includes trained models with the following architectures", unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

st.markdown("**GATe** (Graph Attention Network with edge features) / GAT <br>"
            "**GINe** (Graph Isomorphism Network with edge features) <br>"
            "**PNA** (Principal Neighbourhood Aggregation)<br>"
            "**RGCN** (Relational Graph Convolutional Network)",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)


st.markdown("#### Select Model Version")
model_version = st.selectbox(
    "Choose a model version:",
    ["Graph Attention Network with edge features", "Graph Isomorphism Network with edge features", "Principal Neighbourhood Aggregation","Relational Graph Convolutional Network"],
    index=1,
    key="model_version",
    help="Select the version of the model you want to use."
)
st.markdown("<br>",unsafe_allow_html=True)


# Streamlit app layout
st.markdown("#### Upload Data")

st.markdown("""
Please consider that the application can only work with data in the following structure:
""")

# Display the DataFrame in an interactive table
st.markdown("""
| Timestamp       | From Bank | Account    | To Bank | Account    | Amount Received | Receiving Currency | Amount Paid | Payment Currency | Payment Format |
|-----------------|-----------|------------|---------|------------|-----------------|--------------------|-------------|------------------|----------------|
| 01.09.22 0:08   | 11        | 8000ECA90  | 11      | 8000ECA90  | 3195403         | US Dollar          | 3195403     | US Dollar        | Reinvestment   |
| 01.09.22 0:21   | 3402      | 80021DAD0  | 3402    | 80021DAD0  | 1858.96         | US Dollar          | 1858.96     | US Dollar        | Reinvestment   |
| 01.09.22 0:00   | 11        | 8000ECA90  | 1120    | 8006AA910  | 592571          | US Dollar          | 592571      | US Dollar        | Cheque         |
| 01.09.22 0:16   | 3814      | 8006AD080  | 3814    | 8006AD080  | 12.32           | US Dollar          | 12.32       | US Dollar        | Reinvestment   |
| 01.09.22 0:00   | 20        | 8006AD530  | 20      | 8006AD530  | 2941.56         | US Dollar          | 2941.56     | US Dollar        | Reinvestment   |
| 01.09.22 0:24   | 12        | 8006ADD30  | 12      | 8006ADD30  | 6473.62         | US Dollar          | 6473.62     | US Dollar        | Reinvestment   |
| 01.09.22 0:17   | 11        | 800059120  | 1217    | 8006AD4E0  | 60562           | US Dollar          | 60562       | US Dollar        | ACH            |
""")
st.markdown("<br>",unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Drag and drop or browse to upload your file",
    type=["csv"],
    key="file_upload",
    help="Upload your dataset for inference.",
    label_visibility="collapsed"
)


st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

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
    "<p style='text-align:center;'>&copy;  Dominik Pichler (info[at]dominik-pichler[dot]com - All rights reserved.</p>",
    unsafe_allow_html=True,
)
