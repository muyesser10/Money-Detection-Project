import cv2
import numpy as np
from keras import Model
from keras.src.applications import imagenet_utils
from keras.src.applications.vgg16 import VGG16
from keras.src.utils import load_img, img_to_array
import tensorflow as tf


# Load the pre-trained CNN from disk
model = VGG16(weights="imagenet")

# Load and preprocess the input image
image_path = "resim_dataset/resimler1/1Lira/1_frame_0.jpg"
orig = cv2.imread(image_path)
resized = cv2.resize(orig, (224, 224))  # Resize the original image
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

# Make predictions and decode the ImageNet predictions
preds = model.predict(image)
i = np.argmax(preds[0])
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)

# Initialize GradCAM and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)

# Define the region of interest (ROI) as (x1, y1, x2, y2)
region_bbox = [472.3408, 267.3728, 1446.2030, 797.9960]
x1, y1, x2, y2 = map(int, region_bbox)
roi_heatmap = np.zeros_like(heatmap)
roi_heatmap[y1:y2, x1:x2] = heatmap[y1:y2, x1:x2]

# Resize the heatmap to the original image dimensions
roi_heatmap = cv2.resize(roi_heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(roi_heatmap, orig, alpha=0.5)

# Draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display the original image, heatmap, and output image
output = np.vstack([orig, heatmap, output])
output = cv2.resize(output, (output.shape[1], 700))  # Corrected resize usage
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()