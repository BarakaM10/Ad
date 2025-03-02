import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import pickle
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Teacher Model
class TeacherModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout_rate=0.3):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        out = self.dropout2(self.bn2(torch.relu(self.fc2(out))))
        out = self.fc3(out)
        return out

# Student Model
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

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = correct_train / total_train
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return model

# Knowledge Distillation Training
def train_student_with_distillation(student_model, teacher_model, train_loader, val_loader,
                                   optimizer, T=2.0, alpha=0.5, num_epochs=20, patience=3):
    criterion_ce = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        student_model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            student_logits = student_model(batch_x)
            with torch.no_grad():
                teacher_logits = teacher_model(batch_x)
            
            # Calculate losses
            teacher_soft = torch.softmax(teacher_logits / T, dim=1)
            student_log_soft = torch.log_softmax(student_logits / T, dim=1)
            
            loss_ce = criterion_ce(student_logits, batch_y)
            loss_kl = nn.functional.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
            loss = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kl
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        # Validation phase
        student_model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = student_model(batch_x)
                val_loss += criterion_ce(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = correct_train / total_train
        val_accuracy = correct_val / total_val

        print(f"Student Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = student_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                student_model.load_state_dict(best_model_state)
                break
    
    return student_model

# Save model and related files for deployment
def save_model_for_deployment(student_model, scaler, feature_names, input_dim):
    # Save model
    torch.save(student_model.state_dict(), "model/student_model.pth")
    
    # Save scaler
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Save model configuration
    config = {
        "input_dim": input_dim,
        "hidden_dim": 64,
        "dropout_rate": 0.2,
        "feature_names": feature_names
    }
    
    with open("model/model_config.json", "w") as f:
        json.dump(config, f)
    
    print("Model, scaler, and configuration saved for deployment.")

# Main training function
def train_and_save_models(data_path):
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Prepare features and target
    print("Preparing features...")
    autism_columns = [col for col in data.columns if col.startswith('autism_')]
    X = data.drop(columns=["image_id"] + autism_columns, errors='ignore')
    y = data["autism"]
    
    # Store feature names for deployment
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Get input dimension
    input_dim = X_train_scaled.shape[1]
    
    # Initialize and train teacher model
    print("Training teacher model...")
    teacher_model = TeacherModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=1e-4)
    teacher_model = train_model(teacher_model, train_loader, test_loader, criterion, optimizer, num_epochs=30, patience=3)
    
    # Initialize and train student model
    print("Training student model with knowledge distillation...")
    student_model = StudentModel(input_dim)
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)
    student_model = train_student_with_distillation(
        student_model, 
        teacher_model, 
        train_loader, 
        test_loader,
        optimizer_student,
        T=2.0,
        alpha=0.5,
        num_epochs=20,
        patience=3
    )
    
    # Evaluate student model
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Student model accuracy on test set: {accuracy:.2f}%")
    
    # Save model and related files
    save_model_for_deployment(student_model, scaler, feature_names, input_dim)
    
    return student_model, scaler, feature_names

# If run directly
if __name__ == "__main__":
    train_and_save_models("merged_autism_features.csv")