import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIGURATION
# =========================
data_dir = r"C:\Users\HP\Desktop\Animal footprint tracker\train"            # your original dataset folder
split_dir = r"C:\Users\HP\Desktop\Animal footprint tracker\train_split"     # new folder for auto-split data
img_size = (224, 224)
batch_size = 32
epochs_initial = 15
epochs_finetune = 30
num_unfreeze_layers = 20
USE_CLASS_WEIGHTS = False   # Set True if classes are highly imbalanced

# =========================
# AUTO SPLIT (80% train, 20% val)
# =========================
print("Preparing stratified data split...")

try:
    import splitfolders
except ImportError:
    os.system("pip install split-folders")
    import splitfolders

# Perform the split only once
if not os.path.exists(os.path.join(split_dir, "train")):
    splitfolders.ratio(
        data_dir,
        output=split_dir,
        seed=123,
        ratio=(0.8, 0.2)
    )
    print("✅ Split created at:", split_dir)
else:
    print("ℹ️ Existing split found. Skipping re-split.")

# =========================
# LOAD DATASETS
# =========================
print("\nLoading datasets...")

train_ds = image_dataset_from_directory(
    os.path.join(split_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds_metrics = image_dataset_from_directory(
    os.path.join(split_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
print("Detected Classes:", class_names)

# Extract labels for class weighting
train_labels = np.concatenate([y.numpy() for x, y in train_ds])

# Prefetch for performance
train_ds_fit = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_fit = val_ds_metrics.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================
# CLASS WEIGHTS (optional)
# =========================
class_weights = None
if USE_CLASS_WEIGHTS:
    print("\nCalculating class weights...")
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    class_weights = dict(zip(classes, weights))
    print("Computed class weights:", class_weights)

# =========================
# MODEL ARCHITECTURE
# =========================
print("\nBuilding model...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])


base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet',
    alpha=0.75
)
base_model.trainable = False  # freeze base for stage 1

inputs = layers.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# =========================
# STAGE 1: INITIAL TRAINING
# =========================
print("\n--- Stage 1: Training top layers ---\n")

initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=int(len(train_ds_fit) * 5),
    decay_rate=0.9,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callback_initial = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

history_initial = model.fit(
    train_ds_fit,
    validation_data=val_ds_fit,
    epochs=epochs_initial,
    callbacks=[callback_initial],
    class_weight=class_weights
)

# =========================
# STAGE 2: FINE-TUNING
# =========================
print(f"\n--- Stage 2: Fine-tuning last {num_unfreeze_layers} layers ---\n")

base_model.trainable = True
for layer in base_model.layers[:-num_unfreeze_layers]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callback_finetune = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

start_epoch = history_initial.epoch[-1] + 1
history_finetune = model.fit(
    train_ds_fit,
    validation_data=val_ds_fit,
    epochs=epochs_finetune,
    callbacks=[callback_finetune],
    initial_epoch=start_epoch,
    class_weight=class_weights
)

# =========================
# SAVE CLEAN INFERENCE MODEL
# =========================
inference_model = tf.keras.Model(inputs=model.input, outputs=model.output)
inference_model.save("animal_footprint_inference.keras")
print("✅ Saved inference-ready model (no data augmentation).")

# =========================
# FINAL EVALUATION
# =========================
print("\n--- Final Evaluation and Metrics ---\n")

val_metrics = model.evaluate(val_ds_metrics)
print(f"\nFinal Validation Loss: {val_metrics[0]:.4f}")
print(f"Final Validation Accuracy: {val_metrics[1]:.4f}")

# Predictions for sklearn metrics
y_true = np.concatenate([y.numpy() for x, y in val_ds_metrics])
y_pred_probs = model.predict(val_ds_metrics)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

print("\n" + "="*50)
print("Classification Report (Precision, Recall, F1-Score)")
print("="*50)
print(classification_report(
    y_true,
    y_pred_labels,
    target_names=class_names,
    zero_division=0,
    digits=3
))

print("\n" + "="*50)
print("Confusion Matrix (Rows=True, Cols=Predicted)")
print("="*50)
cm = confusion_matrix(y_true, y_pred_labels)
print(cm)
print(f"\nClass Order: {class_names}")

# Save final model
model.save("animal_footprint_optimized_v9_refined.keras")
print("\n✅ Model saved as 'animal_footprint_optimized_v9_refined.keras'")

# =========================
# PLOT TRAINING CURVES
# =========================
def combine_histories(h1, h2):
    combined = {}
    for k in h1.history.keys():
        combined[k] = h1.history.get(k, []) + h2.history.get(k, [])
    h = tf.keras.callbacks.History()
    h.history = combined
    return h

def plot_combined_history(h1, h2):
    hist = combine_histories(h1, h2)
    plt.figure(figsize=(10, 5))
    plt.plot(hist.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(hist.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.axvline(x=len(h1.history.get('accuracy', [])) - 1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.title('Training & Fine-tuning Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_combined_history(history_initial, history_finetune)
# --- Re-export clean model for compatibility ---
import keras
model.save("animal_footprint_cleaned.keras", save_format="keras_v3")
print("✅ Cleaned model exported successfully as 'animal_footprint_clean.keras'")
