import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import TBiLSTMModel
from data_loader import get_files_and_resample_update
from config import *  # Load configurations like paths, input/output shapes, etc.



# Load and preprocess training data
X_train, Y_train, label_train = get_files_and_resample_update(
    sampling_rate_new,
    window_size,
    locH=path_Heart_Train_Extended,
    locN=path_Lung_Train,
    mode=0  # 0: mix lung noise
)

print("Training data shape:", X_train.shape)

# Define callbacks
checkpoint = ModelCheckpoint(
    filepath=check,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Build the model
model = TBiLSTMModel(
    input_shape=input_shape,
    output_shape=output_shape,
    loss_function='mse'
).model

# Train the model
history = model.fit(
    X_train,
    Y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)

# Save final model
final_model_path = os.path.join("models", "T-BiLSTM_model.h5")
model.save(final_model_path)
print(f"Model saved to {final_model_path}")
