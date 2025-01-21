import streamlit as st
import pandas as pd
import networkx as nx
import torch
from pyvis.network import Network
from app_backend import (check_and_validate_uploadData, infer_on_new_data)
from data_loader_inference import preprocess_uploaded_data
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
st.markdown("This application aims to identifying money laundering transactions in arbitray large networks of banking transactions. "
            "<br>It is based on modern graph neural networks and geometry-based proprocessing and in it's current form, includes trained models with the following architectures", unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

st.markdown("**GATe** (Graph Attention Network with edge features) / GAT <br>"
            "**GINe** (Graph Isomorphism Network with edge features) <br>"
            "**RGCN** (Relational Graph Convolutional Network)",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)


st.markdown("#### Select Model Version")
model_version = st.selectbox(
    "Choose a model version:",
    ["Graph Attention Network with edge features", "Graph Isomorphism Network with edge features","Relational Graph Convolutional Network"],
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

# Data as a dictionary
sample_data = {
    "Timestamp": ["01.09.22 0:08", "01.09.22 0:21", "01.09.22 0:00", "01.09.22 0:16", "01.09.22 0:00", "01.09.22 0:24", "01.09.22 0:17"],
    "From_Bank": [11, 3402, 11, 3814, 20, 12, 11],
    "From_Account": ["8000ECA90", "80021DAD0", "8000ECA90", "8006AD080", "8006AD530", "8006ADD30", "800059120"],
    "To_Bank": [11, 3402, 1120, 3814, 20,12,1217],
    "To_Account": ["8000ECA90", "80021DAD0", "8006AA910", "8006AD080", "8006AD530","8006ADD30","8006AD4E0"],
    "Amount_Received": [3195403,1858.96,592571,12.32,2941.56,6473.62,60562],
    "Receiving_Currency": ["US Dollar"]*7,
    "Amount_Paid":[3195403,1858.96,592571,12.32,2941.56,6473.62,60562],
    "Payment_Currency":["US Dollar"]*7,
    "Payment_Format":["Reinvestment","Reinvestment","Cheque","Reinvestment","Reinvestment","Reinvestment","ACH"]
}

st.dataframe(pd.DataFrame(sample_data))

uploaded_file = st.file_uploader(
    "Drag and drop or browse to upload your file",
    type=["csv"],
    key="file_upload",
    help="Upload your dataset for inference.",
    label_visibility="collapsed"
)
valid_dataset_uploaded = False

if uploaded_file is not None:
    # Read the uploaded CSV file into a Pandas DataFrame
    try:
        df_uploaded_file = pd.read_csv(uploaded_file)
        valid_flag, validation_error = check_and_validate_uploadData(df_uploaded_file)

        if valid_flag:
            st.write("File successfully uploaded and read into a DataFrame!")
            st.dataframe(df_uploaded_file.head())
            valid_dataset_uploaded = True
        else:
            st.write("File is not formated properly and hence not uploaded! Please make sure to name the columns accordingly to the requirements above.")
            st.error(validation_error)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Please upload a CSV file.")


st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

if st.button("Run Inference", key="run_inference"):
    if valid_dataset_uploaded:
        with st.spinner("Running inference..."):
            node_tensor,edge_tensor, edge_features = preprocess_uploaded_data(df_uploaded_file)


            #out = model(batch.x, batch.edge_index, batch.edge_attr)

            #pred = infer_on_new_data(node_tensor,edge_tensor, edge_features,model_version)
            x = infer_on_new_data(node_tensor,edge_tensor, edge_features,model_version)
        st.success("Inference completed successfully!")
        flattened_tensor = torch.flatten(x)


        data = {
            'Timestamp': [
                '01.09.22 0:08', '01.09.22 0:21', '01.09.22 0:00', '01.09.22 0:16',
                '01.09.22 0:00', '01.09.22 0:24', '01.09.22 0:17',
                '01.09.22 0:44', '01.09.22 0:45', '01.09.22 0:42', '01.09.22 0:38',
                '01.09.22 0:54'
            ],
            'From_Bank': [11, 3402, 11, 3814, 20, 12, 11, 11, 3402, 11, 3814, 20],
            'From_Account': [
                '8000ECA90', '80021DAD0', '8000ECA90', '8006AD080',
                '8006AD530', '8006ADD30', '800059120',
                '8000ECA90', '80021DAD0', '8000ECA90', '8006AD080',
                '8006AD530'
            ],
            'To_Bank': [11, 3402, 1120, 3814, 20, 12, 1217, 11, 3402, 1120, 3814, 20],
            'To_Account': [
                '8000ECA90', '80021DAD0', '8006AA910', '8006AD080',
                '8006AD530', '8006ADD30', '8006AD4E0',
                '8006AA910', '80021DAD0', '8006AD530',
                '8006AD080', '8006ADD30'
            ],
            'Amount_Paid': [
                3195403, 1858.96, 592571, 12.32,
                2941.56, 6473.62, 60562,
                1933.63, 2980.18, 1132.25,
                1375.34, 2072.34
            ]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Create a directed graph using NetworkX
        G = nx.DiGraph()

        # Add edges with attributes for the graph
        for index, row in df.iterrows():
            G.add_edge(
                row["From_Account"],
                row["To_Account"],
                amount=row["Amount_Paid"],
            )

        # Separate edges based on a condition (e.g., Amount Paid > threshold)
        threshold = 3000
        red_edges = [(u, v) for u, v, d in G.edges(data=True) if d["amount"] > threshold]
        default_edges = [(u, v) for u, v in G.edges() if (u, v) not in red_edges]

        # Initialize Pyvis Network
        net = Network(height="750px", width="100%", directed=True)
        net.toggle_physics(True)  # Enable physics for drag-and-drop

        # Add nodes and edges to the Pyvis Network
        for node in G.nodes():
            net.add_node(
                node,
                label=f"Account: {node}",
                title=f"Account: {node}",
                color="lightblue",
            )

        for edge in default_edges:
            net.add_edge(
                edge[0],
                edge[1],
                title=f"Amount Paid: {G.edges[edge]['amount']}",
                color="gray",
            )

        for edge in red_edges:
            net.add_edge(
                edge[0],
                edge[1],
                title=f"Amount Paid: {G.edges[edge]['amount']}",
                color="red",
                width=3,
            )

        # Save the Pyvis graph as an HTML file
        html_path = "transaction_network.html"
        net.save_graph(html_path)

        # Inject custom CSS directly into the saved HTML
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Add your CSS to remove borders or customize styles
        custom_css = """
        <style>

        .card {
        border:none !important;}

        #mynetwork {
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        </style>
        """
        html_content = html_content.replace("</head>", f"{custom_css}</head>")

        # Save the modified HTML back
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Embed the HTML file into Streamlit
        st.markdown("### Results - Visual Exploration")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=750)



    else:
        st.error("Please upload a (valid) file to run inference.")

st.markdown("<br>",unsafe_allow_html=True)



# Features Section
st.markdown("<h2 id='features'>Features</h2>", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    st.markdown(
        """
        <div class="card">
        <h4>Ready to scale</h4>
        <p>Built for scale and enterprise-grade security / compliance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div class="card">
        <h4>Cloud & On-Premise Hosting</h4>
        <p>Leveraging cloud and On-Premise infrastructure for optimal performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        """
        <div class="card">
        <h4>Countless Extensions</h4>
        <p>New or custom trained models can be easily integrated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Footer Section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>&copy;  Dominik Pichler (info[at]dominik-pichler[dot]com - All rights reserved.</p>",
    unsafe_allow_html=True,
)
