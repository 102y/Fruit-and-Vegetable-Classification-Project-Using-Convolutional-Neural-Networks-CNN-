import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define directories
train_dir_100x100 = r'C:\Users\NITRO\Desktop\AI Proj\Classification of fruits and vegetables\archive (1)\fruits-360_dataset_100x100\fruits-360\Training'
test_dir_100x100 = r'C:\Users\NITRO\Desktop\AI Proj\Classification of fruits and vegetables\archive (1)\fruits-360_dataset_100x100\fruits-360\Test'

train_dir_original = r'C:\Users\NITRO\Desktop\AI Proj\Classification of fruits and vegetables\archive (1)\fruits-360_dataset_original-size\fruits-360-original-size\Training'
test_dir_original = r'C:\Users\NITRO\Desktop\AI Proj\Classification of fruits and vegetables\archive (1)\fruits-360_dataset_original-size\fruits-360-original-size\Test'
val_dir_original = r'C:\Users\NITRO\Desktop\AI Proj\Classification of fruits and vegetables\archive (1)\fruits-360_dataset_original-size\fruits-360-original-size\Validation'

# Data augmentation and data generators
def create_data_generators(train_dir, test_dir, val_dir=None, target_size=(224, 224), batch_size=10):
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen_test = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen_train.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = datagen_test.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    if val_dir:
        val_generator = datagen_test.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        return train_generator, test_generator, val_generator
    
    return train_generator, test_generator

train_generator_100x100, test_generator_100x100 = create_data_generators(train_dir_100x100, test_dir_100x100, target_size=(100, 100))
train_generator_original, test_generator_original, val_generator_original = create_data_generators(train_dir_original, test_dir_original, val_dir_original, target_size=(224, 224))

def build_VGG16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_VGG19_model(input_shape, num_classes):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_training_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# GPU configuration
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Train and evaluate VGG16 model on 100x100 dataset
    input_shape_100x100 = (100, 100, 3)
    num_classes_100x100 = train_generator_100x100.num_classes

    model_100x100 = build_VGG16_model(input_shape_100x100, num_classes_100x100)

    steps_per_epoch_100x100 = train_generator_100x100.samples // train_generator_100x100.batch_size
    validation_steps_100x100 = test_generator_100x100.samples // test_generator_100x100.batch_size

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history_100x100 = model_100x100.fit(
        train_generator_100x100,
        steps_per_epoch=steps_per_epoch_100x100,
        validation_data=test_generator_100x100,
        validation_steps=validation_steps_100x100,
        epochs=1,
        callbacks=[early_stopping]
    )

    loss_100x100, accuracy_100x100 = model_100x100.evaluate(test_generator_100x100)
    print(f"100x100 Dataset Test Accuracy: {accuracy_100x100 * 100:.2f}%")
    plot_training_history(history_100x100, '100x100 VGG16 Training History')

    # Train and evaluate VGG16 model on original size dataset
    input_shape_original = (224, 224, 3)
    num_classes_original = train_generator_original.num_classes

    model_original = build_VGG16_model(input_shape_original, num_classes_original)

    history_original = model_original.fit(
        train_generator_original,
        validation_data=val_generator_original,
        epochs=10,
        callbacks=[early_stopping]
    )

    loss_original, accuracy_original = model_original.evaluate(test_generator_original)
    print(f"Original Size Dataset Test Accuracy: {accuracy_original * 100:.2f}%")
    plot_training_history(history_original, 'Original Size VGG16 Training History')

    # Train and evaluate VGG19 model on 100x100 dataset
    input_shape_100x100_19 = (100, 100, 3)
    num_classes_100x100_19 = train_generator_100x100.num_classes

    model_100x100_19 = build_VGG19_model(input_shape_100x100_19, num_classes_100x100_19)

    history_100x100_19 = model_100x100_19.fit(
        train_generator_100x100,
        steps_per_epoch=steps_per_epoch_100x100,
        validation_data=test_generator_100x100,
        validation_steps=validation_steps_100x100,
        epochs=10,
        callbacks=[early_stopping]
    )

    loss_100x100_19, accuracy_100x100_19 = model_100x100_19.evaluate(test_generator_100x100)
    print(f"100x100 VGG19 Dataset Test Accuracy: {accuracy_100x100_19 * 100:.2f}%")
    plot_training_history(history_100x100_19, '100x100 VGG19 Training History')

    # Train and evaluate VGG19 model on original size dataset
    model_original_19 = build_VGG19_model(input_shape_original, num_classes_original)

    history_original_19 = model_original_19.fit(
        train_generator_original,
        validation_data=val_generator_original,
        epochs=10,
        callbacks=[early_stopping]
    )

    loss_original_19, accuracy_original_19 = model_original_19.evaluate(test_generator_original)
    print(f"Original Size Dataset VGG19 Test Accuracy: {accuracy_original_19 * 100:.2f}%")
    plot_training_history(history_original_19, 'Original Size VGG19 Training History')

# Visualize images from generators
def visualize_images_from_generator(generator, num_images=9):
    images, labels = next(generator)
    
    class_labels = {v: k for k, v in generator.class_indices.items()}
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_labels[np.argmax(labels[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_images_from_generator(train_generator_100x100)
visualize_images_from_generator(train_generator_original)

# Predict image
def predict_image(model, img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_label = list(train_generator_100x100.class_indices.keys())[class_idx]
    
    return class_label

# Example usage
image_path = 'C:\\Users\\NITRO\\Desktop\\AI Proj\\Classification of fruits and vegetables\\apple.jpg'
predicted_class = predict_image(model_100x100, image_path, target_size=(100, 100))
print(f'Predicted Class: {predicted_class}')
