# JetsonNanoFacialRecognition
Two small programs to detect and save faces on Jetson Nano using Python

This project is heavily based on [this repository.](https://gist.github.com/ageitgey/84943a12dd0d9f54e90f824b94e4c2a9)
For my project I needed a much simpler facial recognition without visual feedback and extra info that he added to his project, so I created this project based on his.


### What you need
1. A jetson nano
2. A camera module: I used the [Raspberry Pi Noir camera V2.](https://www.amazon.de/Raspberry-Pi-V2-1-1080P-Kamera-Modul/dp/B01ER2SKFS)
    Check how to get the video from your camera if you are using a different one and adjust the sizes in doorbell_new_face.py accordingly

If you are not sure how to plug everything in and set up your Jetson Nano I recommend [this link](https://medium.com/@ageitgey/build-a-face-recognition-system-for-60-with-the-new-nvidia-jetson-nano-2gb-and-python-46edbddd7264) to get everything right (that tutorial also follows the following steps on how to get the librabries in greater detail)


### Installing required libraries
Execute these commands in your terminal: `sudo apt-get update`

`sudo apt-get install python3-pip cmake libopenblas-dev liblapack-dev libjpeg-dev`

and

`sudo pip3 -v install Cython face_recognition`

You might also need to install libraries to get your camera to work on Jetson Nano



## doorbell_new_face.py
This program let's you save new faces to the known_faces.dat file.
You could also run whatever action you want to run with your facial recognition in this file, but be vary that you will save every new face and said action will be executed with every face.

You can make adjustments to the quality of the recognition in *line 84*

## doorbell_existing_faces.py
Only loads, checks and executes code from the known_faces.dat
Add the code you want to run when it recognises to *line 83*
