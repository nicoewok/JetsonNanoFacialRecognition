import face_recognition
import cv2
import numpy as np
import platform
import pickle


#Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []


def save_known_faces():
	with open("known_faces.dat", "wb") as face_data_file:
		face_data = [known_face_encodings, known_face_metadata]
		pickle.dump(face_data, face_data_file)
		print("Known face backed up to disk.")


def load_known_faces():
	global known_face_encodings, known_face_metadata
	with open("known_faces.dat", "rb") as face_data_file:
		known_face_encodings, known_face_metadata = pickle.load(face_data_file)
		print("Known faces loaded from disk.")


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
	
	#OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
	
	return (
		f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
		f'width=(int){capture_width}, height=(int){capture_height}, ' +
		f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
		f'nvvidconv flip-method={flip_method} ! ' +
		f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
		'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
		)


def register_new_face(face_encoding, face_image):

	# Add the face encoding and metadata
	known_face_encodings.append(face_encoding)
	known_face_metadata.append({
		"face_image": face_image,
	})


def lookup_known_face(face_encoding):
	
	#See if this is a face we already have in our face list
	
	metadata = None


	#Calculate the "face distance" between the unknown face and every face on in our known face list
	#returns a float between 0.0 and 1.0 for each known face and puts them in a list
	#The smaller the number, the more similar that face was to the current face.
	face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

	if len(face_distances)>0: #check if we even have known_face_encodings
		best_match_index = np.argmin(face_distances)

		#the neural net was trained to have a match at max. 0.6, so we will use that reference
		if face_distances[best_match_index] < 0.6:
		    	#If we have a match, look up the metadata we've saved for it
			metadata = known_face_metadata[best_match_index]
	return metadata


def main_loop():
	# Get access to the webcam
	video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)



	while True:
		# Grab a single frame of video
		ret, frame = video_capture.read()

		# Resize frame of video to 1/4 size for faster face recognition processing
		#you could modify the scaling to achieve better quality with the price of slower recognition
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Find all the face locations and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		# Loop through each detected face and see if it is one we have seen before
		for face_location, face_encoding in zip(face_locations, face_encodings):
			# See if this face is in our list of known faces.
			metadata = lookup_known_face(face_encoding)

			if metadata is not None:
				"""
				here you can put code if it recognises
				"""

				print("We know you!")

			# If this is a brand new face, add it to our list of known faces
			else:
				print("New visitor!")

				# Grab the image of the the face from the current frame of video
				top, right, bottom, left = face_location
				face_image = small_frame[top:bottom, left:right]
				face_image = cv2.resize(face_image, (150, 150))

				# Add the new face to our known face data
				register_new_face(face_encoding, face_image)
				#and save it
				save_known_faces()


	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	load_known_faces()
	main_loop()
