import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from skimage.transform import resize
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# =========================
# 1. CREATE DEMO DATA
# =========================
n_samples = 20
n_timestamps = 60
n_classes = 3

X = np.random.rand(n_samples, n_timestamps)
y = np.random.randint(0, n_classes, n_samples)

# =========================
# 2. RESIZE TO FIXED LENGTH
# =========================
img_size = 32

X_resized = np.array([
    np.interp(
        np.linspace(0, len(sig)-1, img_size),
        np.arange(len(sig)),
        sig
    )
    for sig in X
])

# =========================
# 3. IMAGE TRANSFORMATIONS
# =========================
gaf = GramianAngularField(image_size=img_size, method='summation')
mtf = MarkovTransitionField(image_size=img_size, n_bins=8)
rp = RecurrencePlot(threshold='point', percentage=20)

images = []

for sig in X_resized:
    g = gaf.fit_transform(sig.reshape(1, -1))[0]
    m = mtf.fit_transform(sig.reshape(1, -1))[0]
    r = rp.fit_transform(sig.reshape(1, -1))[0]
    r = resize(r, (img_size, img_size))

    img = np.stack([g, m, r], axis=-1)
    images.append(img)

X_img = np.array(images)
y_cat = to_categorical(y, num_classes=n_classes)

print("Input shape:", X_img.shape)

# =========================
# 4. SIMPLE CNN
# =========================
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =========================
# 5. TRAIN
# =========================
model.fit(X_img, y_cat, epochs=5, batch_size=4, verbose=1)

# =========================
# 6. PREDICT
# =========================
pred = model.predict(X_img)
print("Predicted classes:", np.argmax(pred, axis=1))
