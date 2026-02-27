# Wavelet-Based-multi-frequency-image-transformation-for-time-series-classification
This repository contains the code and experiments for a **hybrid framework for time series classification (TSC)** that leverages **wavelet-based multi-frequency image representations** to enhance CNN-based classification performance.
## 📚 Context

Time series classification (TSC) is critical in domains such as **healthcare, fault diagnosis, and signal analysis**, where signals are often **non-stationary and noisy**. 

Although convolutional neural networks (CNNs) have achieved strong performance, their effectiveness is limited when operating directly on raw signals, especially in capturing **multi-frequency characteristics**.  

Image-based representations such as:

- **Gramian Angular Field (GAF)**  
- **Markov Transition Field (MTF)**  
- **Recurrence Plot (RP)**  

have enabled CNNs to exploit temporal structures more effectively. However, most existing approaches generate these images directly from raw signals, leaving the potential of **frequency-aware representations** underexplored.

---

## 🎯 Objective

The objective of this study is to develop a **hybrid TSC framework for sensor-based signals with rich frequency content**.  

Specifically, we investigate whether generating **GAF, MTF, and RP images from different Discrete Wavelet Transform (DWT) components** can improve representation quality for frequency-dependent time series.

- **Approximation coefficients** → MTF  
- **Medium-frequency components** → GAF  
- **High-frequency components** → RP  

This approach aims to capture complementary information across multiple frequency bands commonly observed in sensor data. Experiments are conducted on **benchmark datasets from the UCR archive**.

---

## ⚙️ Method

1. Decompose each time series using a **two-level DWT with db4 wavelet** → approximation and detail coefficients.  
2. Transform coefficients into **image representations** (MTF, GAF, RP) independently.  
3. Combine images into **multi-channel inputs**.  
4. Classify using a **lightweight CNN architecture**.  
5. Additional experiments using **direct image transformation** and **multi-branch/multi-channel CNN designs** assess the contribution of wavelet-based representations.

---

## 📊 Results

- DWT-based image representations **improve classification accuracy**, especially for **long and noisy time series**.  
- Wavelet-based encoding shows clear advantages for datasets with **complex temporal dynamics** and **low signal-to-noise ratios**.  
- For short and clean signals, CNNs trained on raw image representations achieve comparable or slightly better performance.  

**Key Insight:** Frequency-aware image encoding enhances robustness **without increasing model complexity**.
