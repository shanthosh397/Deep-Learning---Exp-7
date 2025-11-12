# Deep-Learning-Exp-7

## **Implement an Autoencoder in TensorFlow/Keras**

## **AIM**

To develop a convolutional autoencoder for image denoising application.

## **THEORY**

An Autoencoder is an unsupervised neural network that learns to compress input data into a lower-dimensional representation and then reconstruct it back to its original form. It consists of an encoder that reduces the input dimensions and a decoder that rebuilds the input from this compressed data. The model is trained to minimize the reconstruction error between the original and reconstructed data. Autoencoders are widely used for dimensionality reduction, denoising, and anomaly detection tasks.

### **Neural Network Model**

<img width="1162" height="238" alt="image" src="https://github.com/user-attachments/assets/3c21c8db-ca54-42b9-b8e3-cbddf10bc3ff" />

## **DESIGN STEPS**

**STEP 1:** Import the necessary libraries and dataset.

**STEP 2:** Load the dataset and scale the values for easier computation.

**STEP 3**:** Add noise to the images randomly for both the train and test sets.

**STEP 4:** Build the Neural Model using
            Convolutional Layer
            Pooling Layer
            Up Sampling Layer. Make sure the input shape and output shape of the model are identical.
            
**STEP 5:** Pass test data for validating manually.

**STEP 6:** Pass test data for validating manually.

## **PROGRAM**

**Name: Shanthosh G**

**Register Number: 2305003008**

```python
# === DENOISING AUTOENCODER for MNIST ===
from tensorflow.keras import layers, models, Input, datasets
import numpy as np, matplotlib.pyplot as plt, pandas as pd

# === Load and Normalize Data ===
(x_train, _), (x_test, _) = datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.
x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)

# === Add Gaussian Noise ===
noise_factor = 0.5
x_train_noisy = np.clip(x_train + noise_factor * np.random.normal(0, 1, x_train.shape), 0, 1)
x_test_noisy  = np.clip(x_test  + noise_factor * np.random.normal(0, 1, x_test.shape),  0, 1)

# === Show Noisy Samples ===
plt.figure(figsize=(20, 2))
for i in range(10):
    ax = plt.subplot(1, 10, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle("Noisy MNIST Samples", fontsize=14)
plt.show()

# === Model Architecture (Balanced Autoencoder) ===
inp = Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inp)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same', name='Encoded_Layer')(x)

# Decoder
x = layers.Conv2DTranspose(8, (4,4), strides=(1,1), activation='relu', padding='valid')(encoded)
x = layers.Conv2DTranspose(8, (3,3), strides=(2,2), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(16, (3,3), strides=(2,2), activation='relu', padding='same')(x)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

# === Compile & Train ===
autoencoder = models.Model(inp, decoded, name='MNIST_Denoising_Autoencoder')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# === Plot Training Curves ===
metrics = pd.DataFrame(history.history)
metrics[['loss','val_loss']].plot(title='Training vs Validation Loss', figsize=(8,4))
plt.show()

# === Denoise Test Images ===
decoded_imgs = autoencoder.predict(x_test_noisy)
# === Display Original / Noisy / Denoised Images ===
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # Original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    if i == 0: ax.set_title("Original")

    # Noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    if i == 0: ax.set_title("Noisy")

    # Denoised
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    if i == 0: ax.set_title("Denoised")

plt.suptitle("MNIST Denoising Autoencoder Results", fontsize=16)
plt.show()
```

## **OUTPUT**

### **Model Summary**
<img width="829" height="622" alt="image" src="https://github.com/user-attachments/assets/520eba23-ea07-4aab-8faa-e210db50f51a" />

---

### **Training loss**
<img width="680" height="374" alt="image" src="https://github.com/user-attachments/assets/c1da3110-d3a3-4538-9ada-60ab6e05ac96" />

---

### **Original vs Noisy Vs Reconstructed Image**
<img width="1569" height="539" alt="image" src="https://github.com/user-attachments/assets/4bc195f9-fb19-4f6c-8f3f-cd321adc28e9" />


## **RESULT**


Thus the program to develop a convolutional autoencoder for image denoising application has been successfully implemented.
