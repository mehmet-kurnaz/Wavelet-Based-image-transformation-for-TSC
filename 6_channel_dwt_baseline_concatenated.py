import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from skimage.transform import resize
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ===== DEMO DATA =====
# 5 örnek, her biri 50 zaman adımı, 3 sınıf
n_samples = 5
n_timestamps = 50
n_classes = 3

X_demo = np.random.rand(n_samples, n_timestamps)
y_demo = np.random.randint(0, n_classes, size=n_samples)

# ===== RESIZE & NORMALIZE =====
tmstep = 32
X_resized = [np.interp(np.linspace(0, len(sig)-1, tmstep), np.arange(len(sig)), sig) for sig in X_demo]
X_resized = np.array(X_resized)

# ===== IMAGE TRANSFORMATIONS =====
gasf = GramianAngularField(image_size=tmstep, method='summation')
mtf = MarkovTransitionField(image_size=tmstep, n_bins=8)
rp = RecurrencePlot(threshold='point', percentage=20)

X_imgs = []
for sig in X_resized:
    gaf = gasf.fit_transform(sig.reshape(1, -1))[0]
    mtf_img = mtf.fit_transform(sig.reshape(1, -1))[0]
    rp_img = rp.fit_transform(sig.reshape(1, -1))[0]
    rp_img = resize(rp_img, (tmstep, tmstep))
    # 3 kanal birleştirme
    img = np.stack([gaf, mtf_img, rp_img], axis=-1)
    X_imgs.append(img)

X_imgs = np.array(X_imgs)
y_cat = to_categorical(y_demo, num_classes=n_classes)

print("Input shape for CNN:", X_imgs.shape)
print("Labels shape:", y_cat.shape)

# ===== SIMPLE CNN =====
model = models.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(tmstep, tmstep, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== TRAIN =====
model.fit(X_imgs, y_cat, epochs=3, batch_size=2, verbose=1)

# ===== PREDICT =====
y_pred = model.predict(X_imgs)
print("Predictions:\n", np.argmax(y_pred, axis=1))
