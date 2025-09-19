# Anomaly Detection in Medical Imaging (Malaria Cell Classification)

## Overview
This project applies **deep learning for anomaly detection** to identify parasitized cells in microscopy images. Using a **convolutional autoencoder**, the model learns to reconstruct healthy cell images, with reconstruction error serving as the anomaly score. High errors indicate potential parasitized cells.  

The work emphasizes **computational efficiency**, balancing accuracy with limited compute resources.

---

## Dataset
- **NIH Malaria Dataset** (National Library of Medicine):  
  Contains microscopy images of cell samples, both infected (parasitized) and uninfected (healthy).  
  [Dataset Link](https://lhncbc.nlm.nih.gov/publication/pub9932)

---

## Approach

### 1. Preprocessing
- Converted images to grayscale and reduced dimensionality (32×32 / 64×64).  
- Applied thresholding and simplification to reduce noise and computational load.

### 2. Autoencoder Model
- **Encoder**: Conv2D + MaxPooling2D layers  
- **Bottleneck layer**: Small latent dimension to capture only key features of healthy cells  
- **Decoder**: UpSampling2D layers to reconstruct images  

### 3. Training
- Trained exclusively on **healthy cells** to learn baseline representations.  
- Loss function: **Mean Squared Error (MSE)**  
- Optimizer: Adam  

### 4. Evaluation
- Calculated **reconstruction error** on both healthy and parasitized images.  
- Set dynamic thresholds to classify anomalies.  
- Tuned threshold values to improve **precision, recall, and F1-score**.

### 5. Computational Optimization
- Used smaller image sizes and fewer network layers/filters.  
- Avoided full-resolution reconstructions to reduce training time.  
- Balanced trade-offs between detection accuracy and compute efficiency.  

---

## Results
- **Anomaly detection achieved by leveraging reconstruction error**.  
- Adjusting the threshold significantly improved classification metrics (accuracy, precision, F1).  
- Demonstrated that **lightweight networks with preprocessing** can still perform well under compute constraints.  

---

## Key Takeaways
- Autoencoders are effective for unsupervised anomaly detection in medical imaging.  
- Training only on healthy data allows the model to flag parasitized cells as anomalies.  
- Preprocessing and architecture simplification are critical for resource-limited environments.  

---

## Technologies Used
- **Python, TensorFlow/Keras**  
- **OpenCV** for preprocessing  
- **NumPy, Pandas, Matplotlib** for data handling and visualization  

---

## Future Work
- Extend to other medical imaging datasets.  
- Experiment with **variational autoencoders (VAEs)** and **transformer-based architectures**.  
- Deploy as a lightweight diagnostic aid for low-resource settings.  

---
