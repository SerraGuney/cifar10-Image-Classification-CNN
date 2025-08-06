# CIFAR-10 Image Classification with CNN

## Project Description  
This project uses the CIFAR-10 dataset to classify images into 10 categories such as airplane, car, bird, cat, and more. A Convolutional Neural Network (CNN) model is built with Keras to automatically learn and recognize features from images.

The data is preprocessed with normalization and one-hot encoding. Data augmentation techniques like rotation, zoom, and horizontal flipping are applied to improve model robustness. The CNN includes convolutional, pooling, dropout, and dense layers to effectively extract features and perform classification.

Model training runs for 50 epochs with validation on the test set. Performance is evaluated using accuracy, loss graphs, and a detailed classification report with precision, recall, and F1-score metrics.

---

## Key Features
- CIFAR-10 dataset: 60,000 32x32 color images in 10 classes  
- Data augmentation for better generalization  
- CNN architecture with dropout layers to reduce overfitting  
- Training and validation accuracy and loss visualization  
- Classification report for detailed model evaluation  

---

## Dataset Classes  
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

---

## How to Use  
1. Load and preprocess the CIFAR-10 dataset.  
2. Apply data augmentation on training data.  
3. Build and compile the CNN model.  
4. Train the model for 50 epochs.  
5. Evaluate model performance on the test set.  
6. Visualize training/validation metrics and classification report.  

---  

Bu proje, CIFAR-10 veri setindeki 10 farklı kategoriye (uçak, araba, kuş, kedi vb.) ait görüntüleri sınıflandırmak için Keras ile Evrişimli Sinir Ağı (CNN) kullanmaktadır.

Veriler, normalizasyon ve one-hot encoding ile ön işlenir. Modelin genelleme yeteneğini artırmak için döndürme, yakınlaştırma ve yatay çevirme gibi veri artırma teknikleri uygulanır. CNN, evrişim, havuzlama, dropout ve yoğun katmanlardan oluşarak özellik çıkarımı ve sınıflandırma yapar.

Model 50 epoch boyunca eğitilir ve test verisi üzerinde doğrulama yapılır. Performans, doğruluk ve kayıp grafiklerinin yanında precision, recall ve F1-score gibi metriklerle değerlendirilir.

---

## Veri Seti Sınıfları  
- Uçak  
- Araba  
- Kuş  
- Kedi  
- Geyik  
- Köpek  
- Kurbağa  
- At  
- Gemi  
- Kamyon  

---

## Kullanım Adımları  
1. CIFAR-10 veri setini yükle ve ön işle.  
2. Eğitim verisi üzerinde veri artırımı uygula.  
3. CNN modelini oluştur ve derle.  
4. Modeli 50 epoch boyunca eğit.  
5. Test verisi üzerinde performansı değerlendir.  
6. Eğitim ve doğrulama sonuçlarını grafiklerle ve sınıflandırma raporuyla görselleştir.  

---
