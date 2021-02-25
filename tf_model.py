import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from flask import Flask, render_template, Response


app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

model = tf.keras.models.load_model('my_model.h5')
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Show the model architecture
model.summary()
cap=cv2.VideoCapture(0)

# load an image from file
warning=[]

def gen():
	while(True) :
		awareness=0
		ret,frame = cap.read()
		image = cv2.resize(frame,(224, 224))
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces=faceDetect.detectMultiScale(gray, 1.3,5)

		font = cv2.FONT_HERSHEY_SIMPLEX 
  
		org = (50, 50) 
		  
		# fontScale 
		fontScale = 1
		   
		# Blue color in BGR 
		color = (255, 0, 0) 
		  
		# Line thickness of 2 px 
		thickness = 2
		   
		# Using cv2.putText() method 
		

		for(x,y,w,h) in faces:
			frame = cv2.putText(frame, 'Trespassing detected', org, font,  
		                   fontScale, color, thickness, cv2.LINE_AA)
			#print('face detected')
		
		# convert the image pixels to a numpy array
		image = img_to_array(image)


		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

		
		# prepare the image for the VGG model
		image = preprocess_input(image)


		# predict the probability across all output classes
		yhat = model.predict(image)


		
		# convert the probabilities to class labels
		label = decode_predictions(yhat)
		# retrieve the most likely result, e.g. highest probability
		label = label[0][0]
		# print the classification
		#print(label[1])
		tag=label[1]
		if tag in warning:
			frame = cv2.putText(frame, 'Arms Detected', org, font,  
		                   fontScale, color, thickness, cv2.LINE_AA)

		ret, jpeg = cv2.imencode('.jpg', frame)
		image= jpeg.tobytes()
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port


    app.run(host='0.0.0.0',port='5000', debug=True)