# Import Kivy Dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os 
import numpy as np

# Build app and layout
class CamApp(App):
    
    def build(self):
        # Main layout components
        self.img1 = Image(size_hint=(1,.8))
        self.button = Button(text="verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))
        
        # Add items to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv1.h5' , custom_objects={'L1Dist': L1Dist})
        
        # Setup vidoe capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    def update(self, *args):
        
        # Read frame from video capture device
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        
        # flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]) , colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = img_texture
    
    def preprocess(self,file_path):
    
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0

        # Return image
        return img
        
    def verify(self, *args):
        
        # specify thershould 
        detection_threshold = 0.5
        verification_threshold = 0.5
        
        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        
        
        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
        
        # Set Verification Text
        self.verification_label.text = 'verified' if verified == True else 'unverified'
        
        # log out details
        Logger.info(f'Detection Threshold: {results}')
        Logger.info(f'Verification Threshold: {np.sum(np.array(results)> 0.2)}')
        Logger.info(f'Verification Threshold: {np.sum(np.array(results)> 0.4)}')
        Logger.info(f'Verification Threshold: {np.sum(np.array(results)> 0.5)}')
        Logger.info(f'Verification Threshold: {np.sum(np.array(results)> 0.8)}')
        
        return results, verified
    
    
if __name__ == "__main__":
    CamApp().run()
