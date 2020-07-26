Face Recognition for attendance
==

### Setup
1. Install python3
2. pip3 install -r requirements.txt
3. Setup your aws s3 credentials
4. Run `python3 ./data.py` (this shall download all profile images for children in schools)
> Optional part
5. Folder: `dataset/known_faces/{child_id}` shall be generated
6. To add your face into recognition system create a folder with your name under `dataset/known_faces/` and add an image of you there
7. Now run `rm ./children.npy ./encodings.npy` (this shall remove the pre-trained model)

### Running Facial recognition system
1. python3 ./server.py
2. python3 ./client.py
3. Next webcam window will show up. Then you can play around with the env
