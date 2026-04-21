from tensorflow.keras.models import load_model

# load old model
model = load_model("deepfake_model.h5", compile=False)

# save in newer format
model.save("deepfake_model_fixed.keras")
print("Done")