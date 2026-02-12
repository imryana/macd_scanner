"""
LSTM Neural Network for MACD Signal Prediction
Learns temporal patterns from price/indicator sequences
Optimized for GPU training (CUDA)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class TradingSequenceDataset(Dataset):
    """PyTorch Dataset for LSTM training"""
    
    def __init__(self, sequences, labels):
        """
        Args:
            sequences: numpy array of shape (num_samples, sequence_length, num_features)
            labels: numpy array of shape (num_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class MACDLSTMModel(nn.Module):
    """
    LSTM network with attention mechanism for trading signal prediction
    
    Architecture:
    - LSTM layers to capture temporal dependencies
    - Attention mechanism to focus on relevant timeframes
    - Fully connected layers for classification
    """
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3):
        super(MACDLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Fully connected layers (no sigmoid - using BCEWithLogitsLoss)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
        )
        
    def apply_attention(self, lstm_output):
        """
        Apply attention mechanism to LSTM output
        Returns weighted sum of all timesteps
        """
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_size)
        return context, attention_weights
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, sequence_length, features)
        Returns:
            predictions: (batch, 1)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        context, attention_weights = self.apply_attention(lstm_out)
        
        # Classification
        output = self.fc(context)
        
        return output.squeeze()


class LSTMTrainer:
    """
    Trainer class for LSTM model
    Handles training, validation, and prediction
    """
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3, 
                 target_period=5, device=None):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            target_period: Prediction horizon (5, 10, or 20 days)
            device: 'cuda' or 'cpu'. Auto-detected if None
        """
        self.target_period = target_period
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Initialize model
        self.model = MACDLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Loss and optimizer (pos_weight computed in prepare_sequences)
        self.criterion = nn.BCEWithLogitsLoss()  # Will override with pos_weight later
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': []
        }
    
    def prepare_sequences(self, sequences_file, labels_df):
        """
        Load and prepare sequence data
        
        Args:
            sequences_file: Path to .npy file with sequences
            labels_df: DataFrame with labels
            
        Returns:
            train_loader, val_loader, test_loader
        """
        print("\n📊 Loading sequence data...")
        
        # Load sequences
        sequences = np.load(sequences_file, allow_pickle=True)
        print(f"   Loaded {len(sequences)} sequences")
        
        # Get labels
        target_col = f'profitable_{self.target_period}d'
        labels = labels_df[target_col].values
        
        # Remove samples with missing labels
        valid_mask = ~pd.isna(labels)
        sequences = sequences[valid_mask]
        labels = labels[valid_mask]
        
        print(f"   Valid samples: {len(sequences)}")
        print(f"   Positive rate: {labels.mean()*100:.1f}%")
        
        # Pad sequences to same length (30 timesteps)
        max_len = 30
        padded_sequences = []
        
        for seq in sequences:
            # Convert to numpy array if it's a list
            if isinstance(seq, list):
                seq = np.array(seq)
            
            if len(seq) < max_len:
                # Pad with zeros at the beginning
                padding = np.zeros((max_len - len(seq), seq.shape[1]))
                seq = np.vstack([padding, seq])
            elif len(seq) > max_len:
                # Take last max_len timesteps
                seq = seq[-max_len:]
            padded_sequences.append(seq)
        
        sequences = np.array(padded_sequences)
        print(f"   Sequence shape: {sequences.shape}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=0.15, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"\n📈 Data Split:")
        print(f"   Training:   {len(X_train)} samples ({y_train.mean()*100:.1f}% positive)")
        print(f"   Validation: {len(X_val)} samples ({y_val.mean()*100:.1f}% positive)")
        print(f"   Testing:    {len(X_test)} samples ({y_test.mean()*100:.1f}% positive)")
        
        # Handle class imbalance with pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"   Class balance - pos_weight: {pos_weight.item():.3f}")
        
        # Create datasets
        train_dataset = TradingSequenceDataset(X_train, y_train)
        val_dataset = TradingSequenceDataset(X_val, y_val)
        test_dataset = TradingSequenceDataset(X_test, y_test)
        
        # Create dataloaders (larger batch for stability)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for sequences, labels in train_loader:
            sequences = sequences.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions (apply sigmoid for metrics since model outputs logits now)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(np.array(all_labels), np.array(all_preds) > 0.5)
        auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, accuracy, auc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(np.array(all_labels), np.array(all_preds) > 0.5)
        auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, accuracy, auc
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        print("\n" + "="*60)
        print(f"Training LSTM Model ({self.target_period}-day prediction)")
        print("="*60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_auc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_auc = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                      f"Val AUC: {val_auc:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"   ✅ New best model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model (Val Loss: {best_val_loss:.4f})")
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                probs = torch.sigmoid(outputs)
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds > 0.5)
        auc = roc_auc_score(all_labels, all_preds)
        
        print(f"\n📊 Test Set Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC-ROC:  {auc:.3f}")
        
        print("\n📈 Classification Report:")
        print(classification_report(all_labels, all_preds > 0.5, 
                                   target_names=['Unprofitable', 'Profitable']))
        
        return {'accuracy': accuracy, 'auc': auc}
    
    def predict_single(self, sequence):
        """
        Predict for a single sequence
        
        Args:
            sequence: numpy array of shape (sequence_length, num_features)
            
        Returns:
            probability: float between 0 and 1
        """
        self.model.eval()
        
        # Pad/crop to 30 timesteps
        max_len = 30
        if len(sequence) < max_len:
            padding = np.zeros((max_len - len(sequence), sequence.shape[1]))
            sequence = np.vstack([padding, sequence])
        elif len(sequence) > max_len:
            sequence = sequence[-max_len:]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(sequence_tensor)
            prob = torch.sigmoid(output)
        
        return float(prob.cpu().numpy())
    
    def plot_training_history(self, save_path='lstm_training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # AUC
        axes[2].plot(epochs, self.history['train_auc'], label='Train AUC', linewidth=2)
        axes[2].plot(epochs, self.history['val_auc'], label='Val AUC', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].set_title('Training and Validation AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Training history plot saved to: {save_path}")
        plt.close()
    
    def save(self, filepath='lstm_model.pth'):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'target_period': self.target_period
        }, filepath)
        print(f"\n💾 LSTM model saved to: {filepath}")
    
    def load(self, filepath='lstm_model.pth'):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.target_period = checkpoint['target_period']
        print(f"✅ LSTM model loaded from: {filepath}")


if __name__ == "__main__":
    print("="*60)
    print("LSTM Model Training")
    print("="*60)
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available. Training on CPU (slower)")
    
    # Load data
    try:
        df = pd.read_csv('training_data.csv')
        print(f"\n✅ Loaded {len(df)} training samples")
    except FileNotFoundError:
        print("\n❌ training_data.csv not found!")
        print("Run: python data_preparation.py first")
        exit(1)
    
    # Train LSTM
    trainer = LSTMTrainer(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        target_period=5
    )
    
    train_loader, val_loader, test_loader = trainer.prepare_sequences(
        'training_data_sequences.npy', df
    )
    
    trainer.train(train_loader, val_loader, epochs=50, early_stopping_patience=10)
    trainer.evaluate(test_loader)
    trainer.plot_training_history()
    trainer.save('lstm_model_5d.pth')
    
    print("\n" + "="*60)
    print("✅ LSTM training complete!")
    print("="*60)
