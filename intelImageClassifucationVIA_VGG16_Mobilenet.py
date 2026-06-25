# -*- coding: utf-8 -*-

import os

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_067e2e37f1a41224f860af226a653ef5'

!mkdir -p ~/.kaggle
!echo "{os.environ['KAGGLE_API_TOKEN']}" > ~/.kaggle/access_token
!chmod 600 ~/.kaggle/access_token


!rm -rf /content/intel_data
!kaggle datasets download -d puneet6060/intel-image-classification --force
!unzip -q intel-image-classification.zip -d /content/intel_data

TRAIN_DIR = '/content/intel_data/seg_train/seg_train'
if os.path.exists(TRAIN_DIR):
    print(" Dataset ready!")
else:
    print(" Download failed")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
TRAIN_DIR = '/content/intel_data/seg_train/seg_train'
TEST_DIR = '/content/intel_data/seg_test/seg_test'
CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
    zoom_range=0.2, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training', shuffle=True)
val_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False)
test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Model
base = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
base.trainable = False

model = tf.keras.Sequential([
    base, layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen,
                    callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

# Evaluate
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.2%}")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Val')
ax[0].set_title('Accuracy')
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Val')
ax[1].set_title('Loss')
ax[1].legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Your actual training results from the output you shared
accuracy = [0.5841, 0.7324, 0.7650, 0.7739, 0.7883, 0.7916, 0.7980, 0.7954, 0.8031, 0.7989]
val_accuracy = [0.7593, 0.7668, 0.7949, 0.8163, 0.8110, 0.8096, 0.8135, 0.8224, 0.8153, 0.8213]
loss = [1.0649, 0.7220, 0.6429, 0.6130, 0.5890, 0.5649, 0.5533, 0.5443, 0.5303, 0.5334]
val_loss = [0.7031, 0.6110, 0.5623, 0.5246, 0.5197, 0.5200, 0.4926, 0.4861, 0.4973, 0.4784]

epochs_ran = len(accuracy)

# Create the plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History - Intel Scene Classification', fontsize=14)

# Accuracy plot
axes[0].plot(range(1, epochs_ran+1), accuracy, 'b-o', label='Training')
axes[0].plot(range(1, epochs_ran+1), val_accuracy, 'r-o', label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(range(1, epochs_ran+1), loss, 'b-o', label='Training')
axes[1].plot(range(1, epochs_ran+1), val_loss, 'r-o', label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)
print(f"Best Training Accuracy: {max(accuracy):.2%} (Epoch {np.argmax(accuracy)+1})")
print(f"Best Validation Accuracy: {max(val_accuracy):.2%} (Epoch {np.argmax(val_accuracy)+1})")
print(f"Final Test Accuracy: 84.43%")
print(f"Training completed: 10 epochs")

# CONTINUE TRAINING UNTIL ACCURACY DECREASES

from tensorflow.keras import callbacks

print("=" * 60)
print("CONTINUING TRAINING WITH EARLY STOPPING")
print("=" * 60)

# Early stopping - stops when validation accuracy stops improving
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',     # Watch validation accuracy
    mode='max',                 # We want it to go UP
    patience=5,                 # Wait 5 epochs after best before stopping
    restore_best_weights=True,  # Go back to best epoch
    verbose=1
)

# Continue training (start from epoch 10, go up to 50)
history_extended = model.fit(
    train_gen,
    epochs=50,                  # Max 50, but will stop earlier
    validation_data=val_gen,
    callbacks=[early_stop],
    verbose=1,
    initial_epoch=10            # Start from where you left off
)

print("\n" + "=" * 60)
print("TRAINING STOPPED AUTOMATICALLY")
print("=" * 60)

# ============================================
# FINAL EVALUATION ON TEST DATA
# ============================================

print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

# Evaluate on test data (3,000 unseen images)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print(f"\n{'='*60}")
print(f"🎯 FINAL RESULTS")
print(f"{'='*60}")
print(f"✅ Test Accuracy: {test_acc:.2%}")
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Best Validation Accuracy: 84.17% (Epoch 24)")
print(f"✅ Total Epochs Trained: 29 (stopped automatically)")

import matplotlib.pyplot as plt

# Your complete training history
all_acc = history.history['accuracy'] + history_extended.history['accuracy']
all_val_acc = history.history['val_accuracy'] + history_extended.history['val_accuracy']
all_loss = history.history['loss'] + history_extended.history['loss']
all_val_loss = history.history['val_loss'] + history_extended.history['val_loss']

epochs_total = len(all_acc)

# Find best epoch
best_val_acc = max(all_val_acc)
best_epoch = all_val_acc.index(best_val_acc) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Training History - {epochs_total} Epochs (Early Stopping at Epoch 29)', fontsize=14)

# Accuracy plot
axes[0].plot(range(1, epochs_total+1), all_acc, 'b-o', label='Training', markersize=4)
axes[0].plot(range(1, epochs_total+1), all_val_acc, 'r-o', label='Validation', markersize=4)
axes[0].axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best: Epoch {best_epoch} ({best_val_acc:.2%})')
axes[0].axvline(x=29, color='red', linestyle='--', linewidth=2, label='Early Stopping')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(range(1, epochs_total+1), all_loss, 'b-o', label='Training', markersize=4)
axes[1].plot(range(1, epochs_total+1), all_val_loss, 'r-o', label='Validation', markersize=4)
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📊 Summary:")
print(f"   Best Validation Accuracy: {best_val_acc:.2%} (Epoch {best_epoch})")
print(f"   Early Stopping at: Epoch 29")
print(f"   Total Epochs Trained: {epochs_total}")

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 60)
print("FINAL CONFUSION MATRIX")
print("=" * 60)

test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title('Confusion Matrix - Intel Scene Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CATEGORIES))

# Save final model
model.save('intel_final_model.h5')
print("✅ Model saved as 'intel_final_model.h5'")

# Download to your computer
from google.colab import files
files.download('intel_final_model.h5')
print("✅ Download started!")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)

# -*- coding: utf-8 -*-
"""FAST Intel Classification - MobileNetV2 (15 min training)"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ============================================
# STEP 1: DOWNLOAD DATASET (30 sec)
# ============================================

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_067e2e37f1a41224f860af226a653ef5'

!mkdir -p ~/.kaggle
!echo "{os.environ['KAGGLE_API_TOKEN']}" > ~/.kaggle/access_token
!chmod 600 ~/.kaggle/access_token

print("📥 Downloading dataset...")
!rm -rf /content/intel_data
!kaggle datasets download -d puneet6060/intel-image-classification --force
!unzip -q intel-image-classification.zip -d /content/intel_data

TRAIN_DIR = '/content/intel_data/seg_train/seg_train'
TEST_DIR = '/content/intel_data/seg_test/seg_test'
print("✅ Dataset ready!\n")

# ============================================
# STEP 2: OPTIMIZED CONFIGURATION (FAST)
# ============================================

IMG_SIZE = (160, 160)  # Smaller than 224x224 = FASTER
BATCH_SIZE = 128       # Larger batch = FASTER
EPOCHS = 8             # Fewer epochs = MUCH FASTER
LR = 0.0005            # Optimal learning rate
CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

print("=" * 50)
print("⚡ FAST TRAINING CONFIG")
print("=" * 50)
print(f"   Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LR}")
print("=" * 50)

# ============================================
# STEP 3: DATA LOADING (NO AUGMENTATION FOR SPEED)
# ============================================

# Minimal augmentation (only rescale and flip)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  # Simple flip is fast
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("\n📂 Loading data...")

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True,
    classes=CATEGORIES
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False,
    classes=CATEGORIES
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False,
    classes=CATEGORIES
)

print(f"\n✅ Training: {train_gen.samples} | Validation: {val_gen.samples} | Test: {test_gen.samples}")

# ============================================
# STEP 4: BUILD LIGHTWEIGHT MOBILENETV2
# ============================================

print("\n🏗️ Building MobileNetV2...")

# Load with smaller input
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)
base_model.trainable = False  # Keep frozen for speed

# Very simple head (fewer layers = faster)
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Better than Flatten
    layers.Dropout(0.5),              # Single dropout
    layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model ready!")
print(f"   Total params: {model.count_params():,}")
print(f"   Trainable params: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

# ============================================
# STEP 5: TRAIN (FAST - 8 EPOCHS)
# ============================================

print("\n" + "=" * 50)
print("🏋️ TRAINING STARTED (8 EPOCHS)")
print("=" * 50)

# Simple early stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop],
    verbose=1
)

print("\n✅ Training complete!")

# ============================================
# STEP 6: EVALUATION
# ============================================

print("\n" + "=" * 50)
print("📊 EVALUATION")
print("=" * 50)

test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\n🎯 TEST ACCURACY: {test_acc:.2%}")
print(f"   Test Loss: {test_loss:.4f}")

# Predictions for confusion matrix
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

# ============================================
# STEP 7: PLOTS
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], 'b-o', label='Training')
axes[0].plot(history.history['val_accuracy'], 'r-o', label='Validation')
axes[0].set_title(f'Model Accuracy (Test: {test_acc:.2%})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], 'b-o', label='Training')
axes[1].plot(history.history['val_loss'], 'r-o', label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
plt.show()

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CATEGORIES))

# ============================================
# STEP 8: COMPARISON
# ============================================

print("\n" + "=" * 60)
print("📊 BENCHMARK COMPARISON")
print("=" * 60)

benchmarks = {
    'Model': ['Custom CNN', 'MobileNetV2 (Lit)', 'VGG16', 'ResNet34', 'MobileNetV2 (Ours)'],
    'Accuracy': ['62.50%', '83.63%', '84.43%', '~94.00%', f'{test_acc:.2%}'],
    'Speed': ['Fast', 'Fast', 'Slow', 'Medium', '⚡ VERY FAST']
}

print(f"{'Model':<25} {'Accuracy':<12} {'Speed':<15}")
print("-" * 52)
for i in range(len(benchmarks['Model'])):
    print(f"{benchmarks['Model'][i]:<25} {benchmarks['Accuracy'][i]:<12} {benchmarks['Speed'][i]:<15}")
print("=" * 60)

# ============================================
# STEP 9: DOWNLOAD RESULTS
# ============================================

model.save('fast_mobilenet_model.h5')
print("\n✅ Model saved: fast_mobilenet_model.h5")

from google.colab import files
files.download('fast_mobilenet_model.h5')
files.download('training_history.png')
files.download('confusion_matrix.png')

print("\n" + "=" * 50)
print(f"🎉 COMPLETE! Test Accuracy: {test_acc:.2%}")
print("=" * 50)