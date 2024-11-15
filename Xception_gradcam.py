# Gerekli Kütüphaneleri İçe Aktarma
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import os

def main():
    # Veri Seti Yolunu Belirleme
    data_dir = 'dataset_gradcam' 
    if not os.path.isdir(data_dir):
        raise ValueError(f"Veri seti yolu bulunamadı: {data_dir}")
    
    classes = sorted(os.listdir(data_dir))
    print("Sınıflar:", classes)
    
    # Veri Setini Yükleme ve Hazırlama (Veri Artırımı Yapılmıyor)
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # %80 eğitim, %20 doğrulama
    )
    
    batch_size = 32
    target_size = (299, 299)  # Xception için giriş boyutu
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Xception Modelini Oluşturma ve Transfer Öğrenme
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Tüm katmanları dondurma
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(classes), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    
    # Modeli Derleme ve Eğitme
    learning_rate = 0.001
    epochs = 20
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early Stopping Callback
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Eğitim Süreci
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping],
        verbose=1  # Eğitim sırasında bilgi vermek için
    )
    
    # Modeli Değerlendirme ve Metrikleri Hesaplama
    validation_generator.reset()
    predictions = model.predict(validation_generator, steps=validation_generator.samples // batch_size + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    # Sınıflandırma Raporu (Her Sınıf için)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0)
    print("Sınıflandırma Raporu:\n", report)
    
    # Karışıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.title('Confusion Matrix (Karışıklık Matrisi)')
    plt.show()
    
    # Genel başarı metriklerini hesaplama
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    
    # Genel metrikleri yazdırma
    print(f"Genel Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Genel Precision: {precision:.4f}")
    print(f"Genel Recall: {recall:.4f}")
    print(f"Genel F1-Score: {f1:.4f}")
    
    # Eğitim ve Doğrulama Doğruluğunu Görselleştirme
    plt.figure(figsize=(12, 4))
    
    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.title('Doğruluk Eğrisi')
    
    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.title('Kayıp Eğrisi')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
