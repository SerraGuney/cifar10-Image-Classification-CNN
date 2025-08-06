# %% veri setini içeriye aktar ve preprocessing:normalizasyon,one-hot encoding
import numpy as np  # yardımcı kütüphanelerimiz
import matplotlib.pyplot as plt  # yardımcı kütüphanelerimiz


# keras kütüphanesini dahil ettik.
from tensorflow.keras.datasets import cifar10
# etiketleri kategorik hale getirebilmek için onehot encoding için
from tensorflow.keras.utils import to_categorical
# cnn modeli yaratmak için cnn de bir sıralı model
from tensorflow.keras.models import Sequential
from keras.models import load_model#modeli kaydedebilmek için
# cnn de feature extraction için gerekli olan layer ları içeri aktarıyoruz.
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# cnn de clssification için gerekli olan layer ları içeri aktardık.
from tensorflow.keras.layers import Flatten, Dense, Dropout
# ağırlıkları optimize etmemizi sağlar
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation

from sklearn.metrics import classification_report  # validation için gerekli

import warnings
# tüm uyarıları susturur.

# load cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# görselleştirme
class_label = ["Airplane", "Automobile", "Bird", "Cat",
               "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# bazı goruntuleri ve etiketleri görselleştir.
#fig, axes = plt.subplots(1, 5, figsize=(15, 10))
    # 1 satır ve 5 sütundan oluşan bir grafik alanı oluşturmak için kullanılır.
    # fig:Bu, tüm alt grafikleri, başlıkları ve eksen etiketlerini içeren ana konteynerdir. Grafiğinizin tamamını temsil eder.
    # ax:grafiklerin çizildiği alt grafik alanlarını temsil eden nesnedir.
    
fig, axes = plt.subplots(1, 5, figsize=(15, 10))


    # axes[i].imshow(x_train[i]):5 alt grafik alanından birini seçer. i 0 iken ilk alt grafik, 1 iken ikinci alt grafik seçilir.
    # veri setindeki i'inci sıradaki görüntüyü alır (x_train[0] ilk görüntü, x_train[1] ikinci görüntü vb.) ve bu görüntüyü seçilen alt grafik alanına çizer.
    # label=class_label[int(y_train[i])]: Örneğin, class_label listesinin 3. elemanı "Araba" ise, label değişkenine "Araba" değeri atanır.

for i in range(5):
    axes[i].imshow(x_train[i])
    label = class_label[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")
plt.show


# veri seti normalizasyonu:görsel veerilerde 255 e bölünerek yapılır.
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# one-hot encoding:label ları kategorik hale getirdik.
y_train = to_categorical(y_train, 10)  # 10 sınıf vardır.
y_test = to_categorical(y_test, 10)


# %% Veri Arttırımı Data Augmentation
datagen = ImageDataGenerator(
    # görüntüyü +20ile -20 arasında rastgle bir açıyla dereceyle döndürmeyi sağlar.
    rotation_range=20,
    width_shift_range=0.2,  # görüntüyü yatayda %20 kaydırma sağlar.
    height_shift_range=0.2,  # görüntüyü dikeyde %20 kaydırmayı sağlar.
    shear_range=0.2,  # +/-20 derece yamultma
    zoom_range=0.2,  # görüntüye zoom uygulama
    horizontal_flip=True,  # goruntuyu yatayda ters cevirme
    # dönüşüm sonrası oluşan boş piksellerin nasıl doldurulacağını belirler;nearest ise bu boşlukları en yakın komşu pikselin rengiyle doldurur.
    fill_mode="nearest"
)

datagen.fit(x_train)  # data augmentation nu eğitim verilerie üzerinde uygula


# %% Creat,compile and train Model (Modelin oluşturulması,Derlenmesi ve Eğitimi)

# Flatten: Çok boyutlu veriyi düzleştirip tek boyutlu hale getirir.
# Dense: Karar verme ve sınıflandırma için tam bağlı katmandır (genellikle çıktı katmanıdır).
# Dropout: Overfitting’i önlemek için rastgele nöronları geçici olarak devre dışı bırakır.


# CNN modeli oluştur (base model)
model = Sequential()

# Feature Extraction: CONV => RELU => CONV  => RELU => POOL => DROPOUT

# Convolutional (Evrişim) katmanı ekliyor. Satırı parça parça açıklayalım:Giriş görüntüsüne 32 adet 3x3’lük filtre uygular, kenarlardan taşma olmaması için padding ekler, sonucu ReLU aktivasyonuyla işler ve modelin ilk katmanıdır.
# Conv2D: 2 boyutlu evrişim katmanı (görüntüler için)
# 32: 32 adet filtre (kernel) kullanılır. Her biri farklı bir özelliği (kenar, köşe vs) öğrenir ve 32 farklı feature map üretir.
# (3, 3): Her filtrenin boyutu 3x3 pikseldir. Küçük filtreler yerel özellikleri iyi öğrenir.
# padding="same": Görüntü kenarlarında da işlem yapılabilmesi için girişin boyutu korunur. (Padding eklenir)
# activation="relu": Aktivasyon fonksiyonu olarak ReLU kullanılır.
# input_shape=x_train.shape[1:] :Bu, ilk katman olduğu için giriş şekli belirtilir. x_train.shape[1:], örneğin (64, 64, 3) gibi bir şekil olabilir (64x64 piksel, 3 kanal - RGB).

# NOT:Neden 32 tane feature map üretiyoruz? Bir tane yetmiyor mu?
# Her bir filtre (kernel), giriş görüntüsünde farklı bir özelliği (pattern) tanımaya çalışır.
# Bir filtre kenar yakalayabilir.Bir diğeri köşe.Bir başkası doku, renk geçişi, şekil, vs.
# Eğer sadece 1 filtre kullanırsan, 1 tane feature map üretirsin.Ama bu sadece tek tip özelliği çıkarır (örneğin sadece yatay kenarlar).Görüntüler karmaşık olduğu için birden fazla özelliğe aynı anda bakmamız gerekir.
# Örnekle düşünelim:Bir resim düşün: içinde masa var, sandalye var, masa kenarı var, gölgeler var.
# Filtre 1: yatay kenarları bulur
# Filtre 2: dikey kenarları
# Filtre 3: eğik çizgileri
# Filtre 4: koyu bölge geçişlerini
# Filtre 5: masa kenarını vs.
# Eğer 32 farklı filtre kullanırsak, her biri bu resimde farklı şeye "odaklanır", böylece çok daha zengin bir temsil elde ederiz.


# NOT:                       ilk katman           sonraki katmanlar
# input_shape              	ZORUNLU	             GEREKSİZ
# padding	                İstersen verirsin	 İstersen verirsin, yoksa default çalışır
# activation              	Genelde verilir	     Genelde verilir


# dropout katmanı:overfitting i engeller.
# Dropout, eğitim sırasında her iterasyonda (batch’te) bazı nöronları rastgele kapatma yöntemidir.
# Mesela dropout oranı %20 ise, o iterasyonda nöronların %20’si “kapatılır” yani sıfırlanır, katkı sağlamaz.
# Böylece model tek bir yola bağlı kalmaz, çok farklı alt modeller öğrenir.
# Her seferinde farklı kombinasyonlarda nöronlar kapanır.
# Yani aslında, büyük bir tam modelin içinden çok sayıda küçük alt model (subnetwork) rastgele seçilmiş oluyor.
# Eğitim boyunca binlerce farklı alt model eğitilmiş olur.


model.add(Conv2D(32, (3, 3), padding="same",
          activation="relu", input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # rastgele %25 nöron “kapatılacak”


# Feature Extraction: CONC => RELU => CONV  => RELU => POOL => DROPOUT
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Classification:FLATTEN, DENSE, RELU, DROPOUT, DENSE (OUTPUT LAYER)
# flatten katmanı: çok boyutlu matrisleri tek boyutlu hale getirerek düzleştiririz.
# Dense katmanı:
# Dense (tam bağlı) katman, önceki katmandaki tüm nöronlara bağlıdır.
# Yani, bu katmandaki her nöron, önceki katmandaki tüm nöronlardan giriş alır.
# Bu nedenle “fully connected” yani tam bağlantılı katman denir.

# model.add(Dense(10,activation="softmax")):satırının output layer (çıktı katmanı) olduğunu anlamanın yolları:
# 1. Nöron sayısı
# Çıkış katmanındaki nöron sayısı genellikle sınıf sayısına eşittir.
# Mesela 10 sınıf varsa, output layer’da 10 nöron olur.
# Dense(10, ...) ifadesi bunu gösteriyor.
# 2. Aktivasyon fonksiyonu
# Sınıflandırma problemlerinde, son katmanda genellikle:
# softmax (çok sınıflı sınıflandırma için)
# veya sigmoid (ikili sınıflandırma için) kullanılır.
# activation="softmax" çıkış katmanı olduğunu gösterir.


model.add(Flatten())  # vektör oluştur.
# Bir gizli katmanda 512 nöron var, her nöron gelen veriyi işleyip ReLU ile pozitif çıktı üretiyor ve sonucu sonraki katmana gönderiyor.
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))  # output layer

# sinir ağı modelinin katman katman detaylarını gösteren bir özet (summary) tablosu oluşturur.
model.summary()
# tabloddaki: 1,250,858 parametre değişrerek öğrenme işlemi gerçekleştirilecek.


# model derleme(compile): Keras ile oluşturduğun bir derin öğrenme modelini eğitime hazır hale getirmek için kullanılır.yani modelimi yapılandırdık fit ile de eğiticeğiz.
    # decay:Decay, learning rate’in eğitim ilerledikçe azalmasını sağlar. Bu da modeli daha kararlı ve dikkatli hale getirir
    #Optimizer (optimizasyon algoritması), ağırlıkların nasıl güncelleneceğini belirler.
    #loss="categorical_crossentropy":Kayıp fonksiyonu, modelin çıktısının gerçek etiketle ne kadar uyumsuz olduğunu ölçer.
    #metrics=["accurucy"]:Modelin başarısını ölçmek için bir metrik
model.compile(optimizer=RMSprop(learning_rate=0.0001, decay=1e-6),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


#model training:
# model.fit(...) satırı, derin öğrenme modelinin eğitim (training) sürecini başlatır.
#     batch: Verinin küçük gruplara bölünmesi. Örn: 64 resim bir grup diğer 64 resim bir grup.
#     flow()	Bu grupları sırayla modele vermek (akıtmak).
#     epochs=50 yazdığın için model, tüm eğitim verisini 50 kez görecek, yani 50 kez eğitim yapacak.Model her epoch'ta ağırlıklarını günceller
#     validation_data=(x_test, y_test):modelin her epoch (eğitim turu) sonunda, öğrenme sürecinde görmediği ayrı bir veri kümesi olan doğrulama verisi (validation data) üzerinde test edilmesini sağlar.

history=model.fit(datagen.flow(x_train,y_train,batch_size=64),#data augmentataion uygulanmiş veri akışı
          epochs=50,#eğitim dönem sayısı
          validation_data=(x_test,y_test)#doğrulama seti
          )

#model.save("cifar10_model.h5")#☺oluşturduğumuz modeli kaydettik bu sayde kodu her seferinde çalıştırdığımızda model tekrar eğitilmeyecek.



#Not:Biz loss değerinin giderek azalmasını, accuracy değerinin giderek artmassını isityoruz.
     #training lossun azalmasını isitiyoruz
     #val_accuracy'nin giderk artmasını   
     #val_loss'un giderk azalmasını isityoruz.
        

# %% Test Model and Evaulate performance

#kaydettiğimiz modeli yüklüyoruz.
#model = load_model("cifar10_model.h5")

#modelin test seti üzerinden tahminini yap:çıkan değerler şunu söyler örneğin 0 numaralı index'in 3 numaralı sınıfa ait olma olasılığını ifade eder.
y_pred=model.predict(x_test)

#Not:y_pred'te bulunan her bir sample'in hangi sınıfa ait olduğunu bulmanın en kolay yolu olasılığı en yüksek olanı almaktır.Ve predict olarak o sınıfı belirticez.
y_pred_class=np.argmax(y_pred,axis=1)#sample'ların her bir sınıfa ait olma olasılık değerlerinin en büyük olanın sınıf numarasını alır

#sample'ların gerçek ait oldukları sınıfların numaralarını yani index değerlerini elde ediyoruz bu sayde ne kadar başarılı olduğumuza dair karşılaştırma yapabileceğiz.
y_true=np.argmax(y_test,axis=1)

#classifacition report
#tahmin değerlerimiz (y_pred_class) ile gerçek değerleriemiz (y_true) arasında karşılaştırma yapacağımız bu sayede ne kadar başarılı olduğumuzu ölçeceğiz.precision,recall,F1-skor 
report=classification_report(y_true, y_pred_class,target_names=class_label)
print(report)

#accuracy ve loss değerlerini grafik ile inclemek.
plt.figure()# yeni boş bir çizim alanı (figure) yaratılır.

#kayip(loss) grafikleri
plt.subplot(1,2,1)# 1 satır, 2 sutun ve 1.subplot dizaynı
plt.plot(history.history["loss"],label="Train loss")
plt.plot(history.history["val_loss"],label="Validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()#Grafiklerin birbirine çakışmasını engeller, yerleşimi düzenler.
plt.show()

#accuracy grafiği
plt.subplot(1,2,2)# 1 satır, 2 sutun ve 1.subplot dizaynı
plt.plot(history.history["accuracy"],label="Train Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()#Grafiklerin birbirine çakışmasını engeller, yerleşimi düzenler.
plt.show()