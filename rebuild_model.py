"""
Rebuild and train the model in a compatible format
"""
import json
import os
import sys
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# First, check if we have the training data
plant_dir = "PlantVillage"
if not os.path.exists(plant_dir):
    print("Error: PlantVillage directory not found")
    sys.exit(1)

# Count classes
classes = [d for d in os.listdir(plant_dir) if os.path.isdir(os.path.join(plant_dir, d))]
num_classes = len(classes)

print(f"Found {num_classes} plant disease classes")

# Build the same model architecture
img_size = 224
batch_size = 32

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Model architecture created")
print(model.summary())

# Create data generators for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Load training data
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    plant_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    plant_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Train the model
print("\nTraining model...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

print("\nEvaluating model...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save with a different name to avoid conflict
try:
    # Save as SavedModel format which is more compatible
    model.save('plant_model_new', save_format='tf')
    print("\n✓ Model saved in TensorFlow SavedModel format")
except Exception as e:
    print(f"Error saving model: {e}")

# Also save class indices mapping
class_indices = {str(i): class_name for i, class_name in enumerate(sorted(train_generator.class_indices.keys()))}
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print("✓ Class indices saved")
print(f"Classes: {list(class_indices.values())}")
