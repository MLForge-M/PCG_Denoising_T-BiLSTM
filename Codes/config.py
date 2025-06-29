# --------------------------
# Dataset paths
# --------------------------


# === Original Data ===
# path_Heart_Train_Original = "Data_original\PHS Data (Processed)\train"
path_Lung_Train  = r"Data_original\ICBHI Dataset (Processed)\train"

# === Extended Data ===
path_Heart_Train_Extended = r"Extended_Data\Data (Processed)\train"

# === Validation Data ===
pathheartVal     = r"Datasets\OAHS Dataset\Git_val"
pathlungval      = r"Datasets\ICBHI Dataset (Processed)\val"
pathhospitalval  = r"Datasets\Hospital Ambient Noise (HAN) Dataset"

# === Optional External Datasets ===
pathPascal = r"Datasets\PaHS Dataset\train"
pathCinC   = r"Datasets\cinc22-test-1k"

# --------------------------
# Model Configuration
# --------------------------
window_size      = 0.8
input_shape      = 800
output_shape     = 800
sampling_rate_new = 1000

# --------------------------
# Checkpoint Path
# --------------------------
check = r"Models\T-BiLSTM_model.h5"

