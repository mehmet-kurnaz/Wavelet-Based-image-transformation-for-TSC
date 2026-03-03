import numpy as np
import pywt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------
# Example synthetic dataset (for demonstration)
# -------------------------------------------------
n_samples = 200
signal_length = 256
n_classes = 3

X = np.random.randn(n_samples, signal_length)
y = np.random.randint(0, n_classes, n_samples)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# Image Transformation Function
# -------------------------------------------------
def transform_dataset(X, image_size=64):

    gasf = GramianAngularField(image_size=image_size, method='summation')
    mtf = MarkovTransitionField(image_size=image_size, n_bins=8)
    rp = RecurrencePlot(threshold='point', percentage=20)

    images = []

    for signal in X:
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)

        coeffs = pywt.wavedec(signal, 'db4', level=2)

        channels = []

        # GAF from detail coeff
        c = coeffs[1]
        c = (c - c.min()) / (c.max() - c.min() + 1e-6)
        c_resized = np.interp(
            np.linspace(0, len(c)-1, len(signal)),
            np.arange(len(c)), c
        )
        gaf_img = gasf.fit_transform(c_resized.reshape(1, -1))[0]

        # MTF from approximation coeff
        c = coeffs[0]
        c = (c - c.min()) / (c.max() - c.min() + 1e-6)
        c_resized = np.interp(
            np.linspace(0, len(c)-1, len(signal)),
            np.arange(len(c)), c
        )
        mtf_img = mtf.fit_transform(c_resized.reshape(1, -1))[0]

        # RP from second detail coeff
        c = coeffs[2]
        c = (c - c.min()) / (c.max() - c.min() + 1e-6)
        c_resized = np.interp(
            np.linspace(0, len(c)-1, len(signal)),
            np.arange(len(c)), c
        )
        rp_img = rp.fit_transform(c_resized.reshape(1, -1))[0]
        rp_img = resize(rp_img, (image_size, image_size))

        channels.append(gaf_img)
        channels.append(rp_img)
        channels.append(mtf_img)

        images.append(np.stack(channels, axis=-1))

    return np.array(images)

# Transform data
X_train_img = transform_dataset(X_train)
X_test_img = transform_dataset(X_test)

# -------------------------------------------------
# Simple CNN Model
# -------------------------------------------------
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu',
                  input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

y_train_oh = to_categorical(y_train, n_classes)
y_test_oh = to_categorical(y_test, n_classes)

model.fit(
    X_train_img, y_train_oh,
    validation_data=(X_test_img, y_test_oh),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Evaluation
y_pred = np.argmax(model.predict(X_test_img), axis=1)
acc = accuracy_score(y_test, y_pred)

print("Test Accuracy:", acc)
