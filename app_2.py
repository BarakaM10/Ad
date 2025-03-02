import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import json
import pickle

# Set page title and configuration
st.set_page_config(page_title="Autism Detection Model", layout="wide")

# Student Model definition (keep the same as your original)
class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_rate=0.2):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.dropout(self.bn1(torch.relu(self.fc1(x))))
        out = self.fc2(out)
        return out

# Load the trained model and scaler
@st.cache_resource
def load_model(model_path="model/student_model.pth", config_path="model/model_config.json"):
    """Load the model and config with caching for better performance"""
    # Load model configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize the model with the saved configuration
    model = StudentModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, config

# Load the scaler
@st.cache_resource
def load_scaler(scaler_path="model/scaler.pkl"):
    """Load the scaler with caching for better performance"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Process multiple CSV files
def process_csv(uploaded_file, model, scaler, feature_names):
    """Process a CSV file with multiple records"""
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        
        # Check if we have all required HOG features
        hog_columns = [col for col in data.columns if col.startswith('H')]
        
        if len(hog_columns) != 756:
            st.error(f"Expected 756 HOG features, but found {len(hog_columns)} features.")
            return None
            
        # Create a DataFrame with the expected feature names filled with zeros
        all_features = pd.DataFrame(0, index=np.arange(len(data)), columns=feature_names)
        
        # Map HOG features to the corresponding Bin features
        # Assuming the first 756 features in feature_names correspond to HOG features
        hog_feature_names = feature_names[:756]
        
        # Create mapping between H0-H755 and the first 756 feature names
        feature_mapping = dict(zip([f'H{i}' for i in range(756)], hog_feature_names))
        
        # Copy HOG values to the correct feature positions
        for hog_col, bin_col in feature_mapping.items():
            if hog_col in data.columns:
                all_features[bin_col] = data[hog_col]
        
        # Scale features
        X_scaled = scaler.transform(all_features)
        
        # Convert to tensor and predict
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        results = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                output = model(X_tensor[i:i+1])
                probs = torch.softmax(output, dim=1).numpy()[0]
                prediction = torch.argmax(output, dim=1).item()
                
                # Add original data reference
                image_id = data.iloc[i].get('image_id', f'Row_{i+1}')
                actual_label = data.iloc[i].get('autism', 'Unknown')
                
                results.append({
                    "image_id": image_id,
                    "actual_label": "Autism" if actual_label == 1 else "Non-Autism" if actual_label == 0 else "Unknown",
                    "prediction": "Autism" if prediction == 1 else "Non-Autism",
                    "autism_probability": float(probs[1]),
                    "non_autism_probability": float(probs[0])
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Detailed error info:")
        st.code(str(e))
        return None

# Main application
def main():
    st.title("Autism Detection from Facial HOG Features")
    st.write("""
    Upload a CSV file containing HOG (Histogram of Oriented Gradients) features extracted from facial images.
    The file should contain 756 HOG features (H0 through H755), and optionally 'image_id' and 'autism' columns.
    """)
    
    # Load model, configuration and scaler
    try:
        model, config = load_model()
        scaler = load_scaler()
        feature_names = config.get('feature_names', [f'Bin{i+1}' for i in range(756)])
        
        if not feature_names:
            st.error("No feature names found in model config. Using default names.")
            feature_names = [f'Bin{i+1}' for i in range(756)]
            
        st.info(f"Model expects {len(feature_names)} features.")
        
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.warning("Make sure your model files are in the 'model' directory.")
        return
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(str(e))
        return
    
    # CSV Upload Section
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file with features", type="csv")
    
    if uploaded_file is not None:
        if st.button("Process CSV"):
            with st.spinner("Processing CSV file..."):
                results_df = process_csv(uploaded_file, model, scaler, feature_names)
                
            if results_df is not None:
                st.success("CSV processed successfully!")
                st.dataframe(results_df)
                
                # Option to download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
                
                # Display summary statistics
                st.subheader("Summary")
                
                # Count predictions
                autism_count = (results_df["prediction"] == "Autism").sum()
                non_autism_count = (results_df["prediction"] == "Non-Autism").sum()
                total = len(results_df)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", total)
                col2.metric("Autism Predictions", autism_count, f"{autism_count/total:.1%}")
                col3.metric("Non-Autism Predictions", non_autism_count, f"{non_autism_count/total:.1%}")
    
    # About section
    st.header("About this model")
    st.write("""
    This model detects autism based on HOG (Histogram of Oriented Gradients) features extracted from facial images.
    
    Required CSV format:
    - 756 HOG feature columns (H0 through H755)
    - Optional 'image_id' column for reference
    - Optional 'autism' column for actual labels (1 for Autism, 0 for Non-Autism)
    
    The model is a distilled neural network trained using knowledge distillation from a larger teacher model.
    """)
    
    # Required features information
    with st.expander("Required CSV Format"):
        st.write("""
        Your CSV file must contain:
        - 756 columns named H0 through H755 containing HOG features
        - Optional: 'image_id' column for image identification
        - Optional: 'autism' column with actual labels (1 for Autism, 0 for Non-Autism)
        """)
        st.write("\nExample of first few columns:")
        st.code("image_id,H0,H1,H2,...,H755,autism")

# Run the application
if __name__ == "__main__":
    main()