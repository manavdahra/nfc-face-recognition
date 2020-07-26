Face Recognition for attendance
==

### Setup
1. Install python3
2. pip3 install -r requirements.txt
3. Setup your aws s3 credentials
4. Optional: Create folder: `dataset/known_faces/`
5. Optional: To add your face into recognition system create a folder with your name under `dataset/known_faces/` (eg: `dataset/known_faces/manav`) and add an image of you there.

### Running Facial recognition system
1. python3 ./server.py (when run first time this shall fetch images of all children from aws s3 and trains the model)
2. python3 ./client.py
3. Next webcam window will show up. Then you can play around with the env
