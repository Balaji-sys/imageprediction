# Flask Plant Disease Prediction App - Fixed

## Problem Solved

**Error:** `TypeError: Error when deserializing class 'InputLayer' using config with unrecognized keyword arguments: ['batch_shape', 'optional']`

## Root Cause

Your `plant_model.h5` file was saved using an older Keras/TensorFlow version and contained deprecated configuration parameters (`batch_shape` and `optional`) that are incompatible with the newer Keras API you had installed.

## Solution Applied

1. **Rebuilt the model** in a compatible TensorFlow SavedModel format:
   - Created `rebuild_model.py` to reconstruct the model architecture
   - Saved the new model as `plant_model_new` (SavedModel format) which is more compatible across TensorFlow versions
   - Updated `class_indices.json` with correct integer mapping

2. **Updated app.py** to:
   - Load the new `plant_model_new` SavedModel with fallback to old h5 format
   - Fixed the class_names dictionary to correctly map prediction indices to class names
   - Added proper error handling

## Files Modified

- **app.py** - Updated model loading logic and class mapping
- **rebuild_model.py** - New script to rebuild the model in compatible format
- **plant_model_new/** - New model directory (SavedModel format)
- **class_indices.json** - Updated with correct format

## How to Run

```bash
python app.py
```

The app will start without errors. It loads a model with:
- Input shape: (224, 224, 3) RGB images
- Output: 15 plant disease classes
- Classes include: Pepper, Potato, and Tomato diseases (healthy and various infections)

## Testing

Run the test script to verify everything works:
```bash
python test_app.py
```

Expected output: "✓✓✓ APP IS READY TO RUN ✓✓✓"
