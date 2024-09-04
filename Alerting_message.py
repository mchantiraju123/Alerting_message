import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
import openai

# Add your OpenAI API key here
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Set directories for train, validation, and test datasets
train_dir = '/Users/chantirajumylay/Documents/Gra/k/RiceDiseaseDetection/RiceDiseaseDataset/1_train/'
validation_dir = '/Users/chantirajumylay/Documents/Gra/k/RiceDiseaseDetection/RiceDiseaseDataset/2_validation/'
test_dir = '/Users/chantirajumylay/Documents/Gra/k/RiceDiseaseDetection/RiceDiseaseDataset/3_test/'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation and test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical'
)

# Steps per epoch calculation
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Model architecture
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

# Callback for average accuracy
class AverageAccuracyCallback(Callback):
    def on_train_begin(self, logs=None):
        self.accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.accuracies.append(logs.get('accuracy'))
        running_avg_accuracy = np.mean(self.accuracies)
        print(f'Running average accuracy after epoch {epoch + 1}: {running_avg_accuracy:.4f}')

    def on_train_end(self, logs=None):
        avg_accuracy = np.mean(self.accuracies)
        print(f'Average accuracy over all epochs: {avg_accuracy:.4f}')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

average_accuracy_callback = AverageAccuracyCallback()

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, average_accuracy_callback]
)

# Save the trained model
model.save('rice_disease_model.keras')

def get_openai_response_stream(disease, question):
    """
    Generate a response using OpenAI Streaming API based on the detected disease and farmer's question.
    """
    prompt = f"The detected disease is {disease}. A farmer asks: '{question}'. Provide a helpful and detailed response for a rice farmer."
    
    # Make API request with streaming enabled
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        stream=True  # Enable streaming
    )
    
    # Initialize an empty string for the response
    full_response = ""

    # Process the stream response in parts
    for chunk in response:
        if "choices" in chunk:
            text_chunk = chunk['choices'][0].get("text", "")
            print(text_chunk, end="", flush=True)  # Print chunks as they arrive
            full_response += text_chunk  # Append to the complete response

    print()  # Ensure the next output starts on a new line
    return full_response

def predict_disease_with_response_stream(img_path, question):
    """
    Predict the disease from the image and generate a response to the farmer's question using OpenAI streaming.
    """
    # Predict disease from image
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_names = ['RiceBlast', 'BrownSpot', 'BacterialLeafBlight']
    predicted_class = class_names[np.argmax(predictions)]

    # Show the predicted disease
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted disease: {predicted_class}")
    plt.axis('off')
    plt.show()

    # Get a streaming response using OpenAI API
    response = get_openai_response_stream(predicted_class, question)
    print(f"Farmer's Question: {question}")
    print(f"OpenAI's Streaming Answer: {response}")

# Example usage: Predict the disease and get a response from OpenAI
question = "What should I do to treat this disease?"
predict_disease_with_response_stream('/Users/chantirajumylay/Documents/Gra/k/RiceDiseaseDetection/RiceDiseaseDataset/3_test/2_brownSpot/orig/brownspot_orig_009.jpg', question)
