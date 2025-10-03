import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from datetime import datetime
import json
import os
import requests
from streamlit_lottie import st_lottie
import base64
import io

# --- Local Imports ---
# These files must be in the same directory as app.py
from utilities import ExoplanetCNN
from cnn_preprocessing import process_single_target

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Exoplanet Research Platform",
    page_icon="🔭",
    layout="wide"
)
#--- VIDEO BANNER WITH TEXT OVERLAY HERE ---

# --- CSS for Custom Tabs ---
st.markdown("""
<style>
    /* Center the tab bar */
    div[data-testid="stTabs"] > div[role="tablist"] {
        display: flex;
        justify-content: center;
    }

    /* Style the individual tab buttons */
    div[data-testid="stTabs"] button {
        background-color: #262730;
        color: #E0E0E0;
        border: 1px solid #4A4A4A;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        margin: 0 5px;
        font-weight: bold;
        transition: all 0.3s ease;
        border-radius: 999px; 
    }

    /* Style for the tab button when hovered over */
    div[data-testid="stTabs"] button:hover {
        background-color: #3A3B44;
        color: #FFFFFF;
        border-color: #6A6B74;
    }

    /* Style for the selected (active) tab button */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #0068C9;
        color: white;
        border-color: #0052A3;
        border-bottom: 2px solid #0068C9;
    }
    
    /* Center all standard buttons */
    .stButton>button {
        display: block;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)
#-- video container
st.markdown(
    """
    <style>
    .video-container {
        position: relative;
        width: 100%;
        padding-bottom: 56.25%; /* 16:9 aspect ratio for video */
    }
    .video-container video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover; /* Ensures video covers the container */
    }
    .overlay-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        z-index: 1;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Load the video file (using a cached function is best practice)
@st.cache_data
def get_video_bytes(video_path):
    try:
        with open(video_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Video file not found. Please ensure 'assets/banner_video.mp4' exists.")
        return None

video_bytes = get_video_bytes('assets/banner_video.mp4')

# 3. Create the HTML structure with the video and text
if video_bytes:
    st.html(f"""
        <div class="video-container">
            <video autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode('ascii')}" type="video/mp4">
            </video>
            <div class="overlay-text">
                Exploring New Worlds
            </div>
        </div>
    """)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'device' not in st.session_state:
        st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'model_config' not in st.session_state:
        # Configuration is now static based on the model in utilities.py
        st.session_state.model_config = {}

init_session_state()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_and_preprocess_data(global_file, local_file):
    """Load CSV files, display a preview, and prepare data for the model."""
    try:
        global_df = pd.read_csv(global_file)
        local_df = pd.read_csv(local_file)
        
        st.write("Data Preview (Global View):")
        st.dataframe(global_df.head())

        if 'label' not in global_df.columns:
            st.warning("No 'label' column found. The app will run in inference-only mode.")
            y = None
            X_global = global_df.values
            X_local = local_df.values
        else:
            label_map = {1: 1, '1': 1, 'planet': 1, 'planet candidate': 1,
                         0: 0, '0': 0, 'false positive': 0, 'fp': 0}
            labels_series = global_df['label'].str.lower() if global_df['label'].dtype == 'object' else global_df['label']
            y = labels_series.map(label_map).values
            
            if pd.isna(y).any():
                st.error("Some labels could not be understood. Please use 0/1 or 'Planet'/'False Positive'.")
                return None, None, None

            X_global = global_df.drop(columns=['label']).values
            X_local = local_df.drop(columns=['label']).values
            
        return X_global, X_local, y
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def evaluate_model(model, X_global, X_local, y_true, device, batch_size=64):
    """Comprehensive model evaluation on a test dataset."""
    model.eval()
    
    dataset = TensorDataset(
        torch.tensor(X_global, dtype=torch.float32).unsqueeze(1),
        torch.tensor(X_local, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_true, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for X_g_batch, X_l_batch, _ in loader:
            X_g_batch, X_l_batch = X_g_batch.to(device), X_l_batch.to(device)
            outputs = model(X_g_batch, X_l_batch)
            all_probs.extend(outputs.cpu().numpy().flatten())
            
    y_pred_probs = np.array(all_probs)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    return {
        'accuracy': np.mean(y_true == y_pred_classes),
        'precision': precision_score(y_true, y_pred_classes, zero_division=0),
        'recall': recall_score(y_true, y_pred_classes, zero_division=0),
        'f1_score': f1_score(y_true, y_pred_classes, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_probs) if len(np.unique(y_true)) > 1 else 0.5,
        'y_true': y_true,
        'y_pred': y_pred_classes,
        'y_probs': y_pred_probs
    }

def get_planet_context(prediction):
    """Provide context about the classification result."""
    if prediction > 0.9:
        confidence = "Very High"
        description = "Strong planetary signal detected. The light curve shows characteristics highly consistent with a transiting exoplanet."
    elif prediction > 0.7:
        confidence = "High"
        description = "Likely exoplanet candidate. The transit signature is clear, but additional verification recommended."
    elif prediction > 0.5:
        confidence = "Moderate"
        description = "Possible planetary candidate. The signal shows some planetary characteristics but requires careful analysis."
    else:
        confidence = "Low"
        description = "Likely a false positive. The signal may be due to stellar activity, instrumental noise, or an eclipsing binary system."
    
    return confidence, description

# ============================================================
# SIDEBAR - MODEL & CONFIG MANAGEMENT
# ============================================================
def render_sidebar():
    """Render the sidebar for model and configuration management."""
    st.sidebar.title("🔭 Exoplanet Research Platform")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Model Management")

    model_file = st.sidebar.file_uploader("Upload a Custom Model (.pth)", type=['pth'])
    if model_file:
        try:
            st.session_state.model = ExoplanetCNN().to(st.session_state.device) 
            st.session_state.model.load_state_dict(torch.load(model_file, map_location=st.session_state.device))
            st.session_state.model.eval()
            st.sidebar.success("Custom model loaded!")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

    if st.session_state.model:
        st.sidebar.success("Model Status: LOADED")
        st.sidebar.info(f"Using device: {str(st.session_state.device).upper()}")
        buffer = io.BytesIO()
        torch.save(st.session_state.model.state_dict(), buffer)
        buffer.seek(0)
        st.sidebar.download_button(
            label="Download Current Model (.pth)",
            data=buffer,  # Pass the buffer directly
            file_name=f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
            mime="application/octet-stream"
        )
    
    else:
        
        st.sidebar.warning("No model loaded.")
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("This platform uses a deep learning model to classify exoplanet candidates from telescope data.")
# ============================================================
# UI TABS
# ============================================================

def render_classification_tab():
    """Render the UI for the classification tab."""
    st.header("Classify Exoplanet Candidates")
    
    # --- Single Target Classification ---
    with st.container(border=True):
        st.subheader("Process Raw Kepler/TESS Data")
        st.markdown("Enter target information to download, preprocess, and classify a light curve.")
        
        with st.expander("Example: Kepler-90g"):
            st.code("Target ID: KIC 11442793\nPeriod: 210.606\nTransit Time (t0): 2455839.2\nDuration: 7.9")

        col1, col2 = st.columns(2)
        with col1:
            target_id = st.text_input("Target ID", "KIC 11442793")
            period = st.number_input("Period (days)", value=210.606, format="%.4f")
        with col2:
            t0 = st.number_input("Transit Time (BJD)", value=2455839.2, format="%.4f")
            duration = st.number_input("Duration (hours)", value=7.9, format="%.2f")
            mission = st.selectbox("Mission", ["Kepler", "K2", "TESS"])

        if st.button("Process & Classify", type="primary", use_container_width=True):
            if not st.session_state.model:
                st.error("Please load or initialize a model first.")
            else:
                try:
                    with st.spinner("Downloading and preprocessing data... This may take a moment."):
                        global_view, local_view = process_single_target(
                            target_id=target_id, period=period, t0=t0, duration=duration, mission=mission
                        )
                    
                    with st.spinner("Running prediction..."):
                        X_g = torch.tensor(global_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(st.session_state.device)
                        X_l = torch.tensor(local_view, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(st.session_state.device)
                        
                        with torch.no_grad():
                            st.session_state.model.eval()
                            prediction = st.session_state.model(X_g, X_l).item()
                    
                    confidence, description = get_planet_context(prediction, target_id)
                    
                    st.success("Classification Complete!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Planet Probability", f"{prediction:.2%}")
                    c2.metric("Classification", "PLANET" if prediction > 0.7 else "FALSE POSITIVE")
                    c3.metric("Confidence Level", confidence)
                    
                    st.info(f"**Analysis:** {description}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=global_view, mode='lines', name='Global View', line=dict(color='royalblue')))
                    fig.update_layout(title='Global View (Full Orbit)', xaxis_title='Bin', yaxis_title='Normalized Flux')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(y=local_view, mode='lines', name='Local View', line=dict(color='firebrick')))
                    fig2.update_layout(title='Local View (Zoomed on Transit)', xaxis_title='Bin', yaxis_title='Normalized Flux')
                    st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with st.container(border=True):
        st.subheader("Upload Preprocessed Data for Batch Classification")
        c1, c2 = st.columns(2)
        global_file = c1.file_uploader("Global View CSV (2001 features)", type=['csv'])
        local_file = c2.file_uploader("Local View CSV (201 features)", type=['csv'])

        if global_file and local_file:
            X_g, X_l, y = load_and_preprocess_data(global_file, local_file)
            if X_g is not None:
                st.info(f"Loaded {len(X_g)} samples for classification.")

def render_training_tab():
    """Render the UI for the training tab."""
    st.header("Train or Fine-Tune the Model")

    # This expander for the preprocessing script is great.
    with st.expander("Process Your Own Raw Data?"):
        # ... (content for downloading the script remains the same) ...
        st.markdown("""
        To ensure your raw data is compatible with our model, you need to process it into our required **global** and **local** view formats. 
        
        1.  Download the Python script below.
        2.  Install the required libraries: `pip install lightkurve pandas numpy argparse`.
        3.  Run the script from your terminal with your target's parameters.
        4.  Upload the two CSV files that the script generates.
        """)
        
        with open("preprocess_data.py", "r") as f:
            script_content = f.read()

        st.download_button(
            label="📄 Download Preprocessing Script",
            data=script_content,
            file_name="preprocess_data.py",
            mime="text/x-python"
        )
        st.code("python preprocess_data.py --target_id \"KIC 11442793\" --period 210.6 --t0 2455839.2 --duration 7.9")

    st.info("Upload your preprocessed and labeled CSV data below.")

    col1, col2 = st.columns(2)
    train_global = col1.file_uploader("Training Global CSV", type=['csv'], key='train_g')
    train_local = col2.file_uploader("Training Local CSV", type=['csv'], key='train_l')

    if train_global and train_local:
        X_g, X_l, y = load_and_preprocess_data(train_global, train_local)
        if X_g is not None and y is not None:
            st.write(f"Loaded {len(X_g)} samples. Class distribution: {np.bincount(y)}")

            with st.form("training_form"):
                st.subheader("Training Configuration")

                # --- NEW: Add the training mode radio button ---
                training_mode = st.radio(
                    label="Select Training Mode",
                    options=("Train a New Model from Scratch", "Fine-Tune the Current Model"),
                    index=1,  # Default to Fine-Tune
                    horizontal=True,
                    help="Choose 'Train from Scratch' to initialize a new model. Choose 'Fine-Tune' to continue training the model that is currently loaded."
                )
                # ----------------------------------------------
                
                c1, c2, c3 = st.columns(3)
                epochs = c1.number_input("Epochs", 1, 100, 10)
                batch_size = c2.number_input("Batch Size", 8, 256, 32)
                learning_rate = c3.number_input("Learning Rate", 0.000001, 0.01, 0.0001, format="%.6f")
                test_split = c1.slider("Validation Split", 0.1, 0.5, 0.2)
                optimizer_choice = c2.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
                
                submitted = st.form_submit_button("Start Training", type="primary")

                if submitted:
                    if not st.session_state.model and training_mode == "Fine-Tune the Current Model":
                        st.error("Please load a model from the sidebar before you can fine-tune it.")
                    else:
                        # Pass the new option to the training loop
                        run_training_loop(X_g, X_l, y, epochs, batch_size, learning_rate, test_split, optimizer_choice, training_mode)
def run_training_loop(X_g, X_l, y, epochs, batch_size, lr, val_split, optimizer_choice, training_mode):
    """The main training loop with an interactive chart and a final summary report."""
    if training_mode == "Train a New Model from Scratch":
        st.info("Initializing a new model with random weights for training...")
        # This line creates a brand new model, discarding any old one
        st.session_state.model = ExoplanetCNN().to(st.session_state.device)
    else: # This is the "Fine-Tune" mode
        st.info("Starting fine-tuning on the currently loaded model...")
    # --------------------------------------------------------
    st.info("Starting training process...")
    # --- Data Setup  ---
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=y, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_g[train_idx], dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(X_l[train_idx], dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1))
    
    val_dataset = TensorDataset(torch.tensor(X_g[val_idx], dtype=torch.float32).unsqueeze(1),
                                torch.tensor(X_l[val_idx], dtype=torch.float32).unsqueeze(1),
                                torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1))
                                
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- Model and Optimizer Setup (same as before) ---
    model = st.session_state.model
    criterion = nn.BCELoss()
    if optimizer_choice == "Adam": optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == "SGD": optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else: optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # --- UI Placeholders for Live Updates ---
    progress_bar = st.progress(0, text="Initializing training...")
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        # ... (Training steps for one epoch are the same) ...
        total_train_loss = 0
        for X_g_b, X_l_b, y_b in train_loader:
            X_g_b, X_l_b, y_b = X_g_b.to(st.session_state.device), X_l_b.to(st.session_state.device), y_b.to(st.session_state.device)
            optimizer.zero_grad()
            outputs = model(X_g_b, X_l_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        # ... (Validation steps for one epoch are the same) ...
        total_val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_g_b, X_l_b, y_b in val_loader:
                X_g_b, X_l_b, y_b = X_g_b.to(st.session_state.device), X_l_b.to(st.session_state.device), y_b.to(st.session_state.device)
                outputs = model(X_g_b, X_l_b)
                loss = criterion(outputs, y_b)
                total_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += y_b.size(0)
                correct += (predicted == y_b).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        # --- **IMPROVEMENT**: Live Update with Interactive Plotly Chart ---
        progress_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch+1}/{epochs}")
        status_text.text(f"Current Val Acc: {val_accuracy:.3f} | Best Val Acc: {best_val_accuracy:.3f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history['train_loss'], mode='lines+markers', name='Train Loss'))
        fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines+markers', name='Validation Loss'))
        fig.update_layout(title="Live Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # --- **IMPROVEMENT**: Clear live elements and show final summary ---
    progress_bar.empty()
    status_text.empty()
    chart_placeholder.empty() # Clear the live chart so we can show a final one

    st.success("Training Complete!")
    st.session_state.training_history.append(history)

    with st.container(border=True):
        st.subheader("Training Summary Report")
        
        c1, c2 = st.columns(2)
        c1.metric("Best Validation Accuracy", f"{best_val_accuracy:.2%}")
        c2.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
        
        # Display the final chart
        final_fig = go.Figure()
        final_fig.add_trace(go.Scatter(y=history['train_loss'], mode='lines', name='Train Loss'))
        final_fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'))
        final_fig.update_layout(title="Final Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(final_fig, use_container_width=True)
        
        with st.expander("Show Training Hyperparameters"):
            st.json({
                "Epochs": epochs,
                "Batch Size": batch_size,
                "Learning Rate": lr,
                "Validation Split": val_split,
                "Optimizer": optimizer_choice
            })
        
        st.info("💡 **Next Step:** To assess true model performance, use the **'Test Set Evaluation'** tab with data the model has never seen.")
        model_path = "trained_model.pth"
        # Save the final state of the model (whether new or fine-tuned)
        torch.save(st.session_state.model.state_dict(), model_path)

        with open(model_path, "rb") as f:
            st.download_button(
                label="📁 Download Trained Model (.pth)",
                data=f,
                file_name=f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth",
                mime="application/octet-stream"
            )
        # Clean up the temporary file
        if os.path.exists(model_path):
            os.remove(model_path)
def render_evaluation_tab():
    st.header("Test Set Evaluation")
    st.markdown("""
    Evaluate your model's true performance on completely unseen test data. This provides scientifically rigorous metrics 
    that indicate how well your model will perform on new exoplanet candidates.
    """)
    
    if not st.session_state.model:
        st.warning("Load a model to evaluate its performance.")
        return
        
    st.info("Upload a labeled test dataset (data the model has never seen) to generate a comprehensive performance report.")
    c1, c2 = st.columns(2)
    eval_g = c1.file_uploader("Test Global CSV", type=['csv'], key='eval_g')
    eval_l = c2.file_uploader("Test Local CSV", type=['csv'], key='eval_l')

    if eval_g and eval_l:
        X_g, X_l, y = load_and_preprocess_data(eval_g, eval_l)
        if X_g is not None and y is not None:
            if st.button("Evaluate Model", type="primary", use_container_width=True):
                with st.spinner("Calculating performance metrics..."):
                    results = evaluate_model(st.session_state.model, X_g, X_l, y, st.session_state.device)
                
                st.success("Evaluation Complete!")
                
                # Metrics display
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Accuracy", f"{results['accuracy']:.3f}")
                c2.metric("Precision", f"{results['precision']:.3f}")
                c3.metric("Recall", f"{results['recall']:.3f}")
                c4.metric("F1-Score", f"{results['f1_score']:.3f}")
                st.metric("ROC AUC", f"{results['roc_auc']:.3f}")

                # Visualizations
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                cm = confusion_matrix(results['y_true'], results['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                            xticklabels=['FP', 'Planet'], yticklabels=['FP', 'Planet'])
                ax1.set_title('Confusion Matrix')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')

                fpr, tpr, _ = roc_curve(results['y_true'], results['y_probs'])
                ax2.plot(fpr, tpr, label=f'AUC = {results["roc_auc"]:.3f}')
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_title('ROC Curve')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.legend()
                ax2.grid(True)

                st.pyplot(fig)
                
                # Generate downloadable report
                report = f"""Exoplanet Model Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

===================================
PERFORMANCE METRICS
===================================
Accuracy:  {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall:    {results['recall']:.4f}
F1-Score:  {results['f1_score']:.4f}
ROC AUC:   {results['roc_auc']:.4f}

===================================
CONFUSION MATRIX
===================================
                Predicted
              FP    Planet
Actual FP     {cm[0,0]}     {cm[0,1]}
       Planet {cm[1,0]}     {cm[1,1]}

===================================
DATASET INFO
===================================
Total Samples: {len(results['y_true'])}
True Planets: {np.sum(results['y_true'])}
False Positives: {len(results['y_true']) - np.sum(results['y_true'])}

===================================
NOTES
===================================
This evaluation was performed on a held-out test set.
Metrics reflect the model's expected performance on new, unseen data.
"""
                
                st.download_button(
                    label="📄 Download Evaluation Report",
                    data=report,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
def display_data_analysis(X_g, y, title):
    """Takes data and labels and displays a full statistical analysis."""
    
    st.subheader(title)

    # --- Display Key Metrics ---
    total_samples = len(y)
    planet_count = np.sum(y)
    fp_count = total_samples - planet_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{total_samples}")
    c2.metric("Planet Candidates", f"{planet_count}")
    c3.metric("False Positives", f"{fp_count}")

    # --- Visualize Class Distribution ---
    st.subheader("Class Distribution")
    class_counts = pd.Series(y).map({1: 'Planet', 0: 'False Positive'}).value_counts()
    fig_bar = go.Figure(go.Bar(x=class_counts.index, y=class_counts.values, marker_color=['#0068C9', '#FF4B4B']))
    fig_bar.update_layout(title_text='Number of Samples per Class')
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Visualize Average Signal Shape ---
    st.subheader("Average Signal Shape (Global View)")
    avg_planet_signal = X_g[y == 1].mean(axis=0)
    avg_fp_signal = X_g[y == 0].mean(axis=0)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=avg_planet_signal, mode='lines', name='Average Planet Signal', line=dict(color='royalblue')))
    fig_line.add_trace(go.Scatter(y=avg_fp_signal, mode='lines', name='Average False Positive Signal', line=dict(color='firebrick', dash='dash')))
    fig_line.update_layout(
        title_text='Average Signal Shape: Planets vs. False Positives',
        xaxis_title='Bin Number',
        yaxis_title='Normalized Flux',
        legend_title='Signal Type'
    )
    st.plotly_chart(fig_line, use_container_width=True)
def render_data_analysis_tab():
    """
    Renders a tab to visualize dataset statistics.
    Shows user-loaded data if available, otherwise shows a default summary image.
    """
    st.header("Training Dataset Analysis")

    # Check if the user has uploaded data in the Training tab
    if 'trained_data' in st.session_state:
        st.info("Showing statistics for the dataset you uploaded in the 'Training' tab.")
        data = st.session_state.trained_data
        
        # This function displays the dynamic analysis for the user's data
        display_data_analysis(data['X_g'], data['y'], "User-Loaded Dataset Overview")

    else:
        # --- NEW: If no user data, show the default summary image ---
        st.info("No user data loaded yet. Showing a summary of the model's original training dataset.")
        
        try:
            # IMPORTANT: Change this path to match your image file
            st.image('assets/training_metrics.png', caption="Statistics from the original dataset used to train this model.")
        except FileNotFoundError:
            st.error("Default statistics image not found. Please ensure 'assets/training_stats.png' exists.")
        # -----------------------------------------------------------
def render_educational_tab():
    """Renders the educational content tab with resources."""
    st.header("How We Discover Exoplanets")
    
    st.markdown("""
    ### The Transit Method: A Cosmic Shadow Play
    The most successful method for finding planets outside our solar system is the **transit method**. Imagine watching a distant, bright light. If a small object passes in front of that light, you'll see a tiny, temporary dip in its brightness.
    
    That's exactly what telescopes like Kepler and TESS do. They stare at thousands of stars, measuring their brightness with incredible precision. When a planet's orbit takes it directly between its star and our telescope, it blocks a small fraction of the starlight, creating a "transit." By measuring these periodic dips in a star's **light curve** (a graph of its brightness over time), we can infer the presence of a planet.
    """)

    st.markdown("""
    ### The Challenge: Finding a Needle in a Haystack
    This sounds simple, but the data is incredibly noisy. The dip in brightness from an Earth-sized planet is minuscule (less than 0.01%). This tiny signal can be buried in noise from:
    - **Stellar Variability**: Stars have "starspots" (like sunspots) and pulsations that cause their brightness to fluctuate.
    - **Instrumental Noise**: The telescope's electronics aren't perfect.
    - **Astrophysical False Positives**: An eclipsing binary star system, where two stars orbit and eclipse each other, can create a signal that mimics a planet.
    
    Sifting through millions of light curves to distinguish real planet transits from these false alarms is a monumental task. This is where machine learning becomes an invaluable tool.
    
    ---
    
    ## Machine Learning Approaches to Classification
    
    Machine learning models can learn the subtle patterns that differentiate a true U-shaped planet transit from a V-shaped eclipsing binary or random noise. Here's a look at the key algorithms from recent research.
    """)
    
    with st.container(border=True):
        st.subheader("1. Deep Learning with Convolutional Neural Networks (CNNs)")
        st.markdown("**Paper:** _Identifying Exoplanets with Deep Learning_ by Shallue & Vanderburg (2018)")
        st.warning("⭐ **This is the approach implemented in this platform.**")
        
        st.markdown("""
        This pioneering paper introduced a deep learning model called **AstroNet**. The key idea is to treat the light curve data like a one-dimensional image and use a Convolutional Neural Network (CNN) to find important features automatically.

        - **Dual Views**: The model looks at the data in two ways:
            - **Global View**: The entire light curve folded on the planet's orbital period. This helps the model spot things like secondary eclipses, which are a tell-tale sign of a binary star system, not a planet.
            - **Local View**: A zoomed-in view centered on the transit itself. This allows the model to analyze the precise shape of the dip (U-shape for planets, often V-shape for stars).
        - **Automatic Feature Extraction**: Unlike older methods, the CNN learns which features are important by itself. Early layers might learn to detect simple edges, while deeper layers combine these to recognize the full transit shape.
        - **High Accuracy**: This end-to-end approach proved to be highly effective, achieving over 98% accuracy in ranking planets higher than false positives.
        
        📚 **Citation:** Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.
        """)
        
    with st.container(border=True):
        st.subheader("2. Classical Machine Learning with Feature Engineering")
        st.markdown("**Paper:** _Exoplanet detection using machine learning_ by Malik, Moster, & Obermeier (2022)")
        
        st.markdown("""
        This paper explores a different "classical" machine learning path. Instead of letting the model figure out the features from the raw data, it relies on **feature engineering**.

        - **Feature Extraction**: The light curve is first analyzed to extract a large number of statistical features (nearly 800 of them!). These features describe the light curve in mathematical terms, such as its absolute energy, the number of peaks, its standard deviation, coefficients from a Fourier transform, and more. A library called `TSFRESH` is used for this.
        - **Gradient Boosted Trees**: Once the features are extracted, a powerful but more traditional model called a **Gradient Boosted Tree (GBT)** is trained on them. A GBT model builds a series of "decision trees" sequentially, where each new tree corrects the errors of the previous one.
        - **Efficiency**: This approach is often much faster to train than a deep neural network and doesn't require specialized hardware like GPUs. The paper shows that this method can achieve results comparable to deep learning models, demonstrating the power of thoughtful feature engineering.
        """)
        
    with st.container(border=True):
        st.subheader("3. Ensemble-Based Machine Learning")
        st.markdown("**Paper:** _Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification_ by Luz, Braga, & Ribeiro (2024)")
        
        st.markdown("""
        This research focuses on **Ensemble Learning**, which is based on the idea that "many heads are better than one." Instead of relying on a single complex model, ensemble methods combine the predictions of several simpler models to make a more robust and accurate final decision.

        - **Combining Models**: The paper assesses several ensemble techniques:
            - **Random Forest**: Builds hundreds of individual decision trees on different random subsets of the data and averages their predictions.
            - **Adaboost**: Trains a sequence of simple models, giving more weight to data points that were previously misclassified.
            - **Stacking**: Trains several different types of models (e.g., a Random Forest, a Neural Network) and then trains a final "meta-model" to learn how to best combine their individual predictions.
        - **Improved Performance**: The study shows that these ensemble algorithms, particularly Stacking, perform very well on exoplanet data. By combining the strengths of different models, they can often achieve higher accuracy and reliability than any single model on its own.
        """)
    
    st.markdown("---")
    
    # Resources Section
    with st.container(border=True):
        st.subheader("📚 NASA Datasets & Resources")
        st.markdown("""
        Access the official datasets used for exoplanet research:
        
        **Kepler Mission:**
        - [NASA Exoplanet Archive - Kepler](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
        - [Kepler Confirmed Planets](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=planets)
        
        **K2 Mission:**
        - [K2 Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates)
        
        **TESS Mission:**
        - [TESS Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
        - [MAST TESS Data Archive](https://archive.sts""")

if 'model' not in st.session_state or st.session_state.model is None:
    
    # --- WELCOME SCREEN ---
    st.header("Are you ready to discover new worlds?")
    
    # Use columns to center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # This button will initialize the model and trigger a script rerun
        if st.button("🚀 Initialize Model & Get Started", type="primary", use_container_width=True):
            with st.spinner("Initializing model..."):
                st.session_state.model = ExoplanetCNN().to(st.session_state.device)
            # This forces a rerun of the script. Now st.session_state.model will exist.
            st.rerun()

else:
    # The sidebar is only rendered once the main app is active
    render_sidebar()

tab_names = ["Classification", "Training", "Dataset Analysis", "Model Evaluation", "Educational"]
tabs = st.tabs(tab_names)

with tabs[0]:
    render_classification_tab()

with tabs[1]:
    render_training_tab()

# Create a new section to call the new function
with tabs[2]:
    render_data_analysis_tab()

with tabs[3]:
    render_evaluation_tab()
    
with tabs[4]:
    render_educational_tab()