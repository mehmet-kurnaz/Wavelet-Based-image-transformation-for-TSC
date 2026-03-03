import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from skimage.transform import resize
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

# ======================
# 1. DEMO DATA
# ======================
n_samples = 20
n_timestamps = 64
n_classes = 3
img_size = 32

X = np.random.rand(n_samples, n_timestamps)
y = np.random.randint(0, n_classes, n_samples)

# ======================
# 2. RESIZE
# ======================
X_resized = np.array([
    np.interp(np.linspace(0, len(sig)-1, img_size),
              np.arange(len(sig)),
              sig)
    for sig in X
])

# ======================
# 3. IMAGE TRANSFORMS
# ======================
gaf = GramianAngularField(image_size=img_size, method='summation')
mtf = MarkovTransitionField(image_size=img_size, n_bins=8)
rp = RecurrencePlot(threshold='point', percentage=20)

X_img = []
X_dwt = []

for sig in X_resized:
    g = gaf.fit_transform(sig.reshape(1, -1))[0]
    m = mtf.fit_transform(sig.reshape(1, -1))[0]
    r = rp.fit_transform(sig.reshape(1, -1))[0]
    r = resize(r, (img_size, img_size))

    img = np.stack([g, m, r], axis=-1)
    X_img.append(img)

    # Fake DWT-like second branch (demo için)
    dwt_like = np.stack([g], axis=-1)  # tek kanal örnek
    X_dwt.append(dwt_like)

X_img = np.array(X_img)
X_dwt = np.array(X_dwt)
y_cat = to_categorical(y, n_classes)

print("Main input shape:", X_img.shape)
print("DWT input shape:", X_dwt.shape)

# ======================
# 4. DUAL BRANCH MODEL
# ======================

# ----- Branch 1 (GAF+MTF+RP)
input1 = Input(shape=(img_size, img_size, 3))
x1 = layers.Conv2D(16, (3,3), activation='relu')(input1)
x1 = layers.MaxPooling2D(2,2)(x1)
x1 = layers.Flatten()(x1)

# ----- Branch 2 (DWT)
input2 = Input(shape=(img_size, img_size, 1))
x2 = layers.Conv2D(8, (3,3), activation='relu')(input2)
x2 = layers.MaxPooling2D(2,2)(x2)
x2 = layers.Flatten()(x2)

# ----- Concatenate
combined = layers.concatenate([x1, x2])

# ----- Final layers
z = layers.Dense(32, activation='relu')(combined)
output = layers.Dense(n_classes, activation='softmax')(z)

model = Model(inputs=[input1, input2], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ======================
# 5. TRAIN
# ======================
model.fit([X_img, X_dwt], y_cat,
          epochs=5,
          batch_size=4,
          verbose=1)

# ======================
# 6. PREDICT
# ======================
pred = model.predict([X_img, X_dwt])
print("Predicted:", np.argmax(pred, axis=1))
