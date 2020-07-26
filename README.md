Face Recognition for attendance
==

### Setup (requirements)
1. Install python3
2. `pip3 install -r requirements.txt`
3. Setup your aws s3 credentials. Boto3 library uses the credentials

### Running Facial recognition system
1. `python3 server.py` (when run first time this shall fetch images of all children from aws s3 and trains the model)
2. `python3 client.py`
3. Next webcam window will show up. Then you can play around with the env

### Adding your face into recognition system
1. Create folder `dataset/known_faces/` (if it does not exists) in project root directory
2. Add new folder with your name under that (eg: `dataset/known_faces/{your_name}`)
3. Add an image of you under that directory
4. Now run `python3 model.py {your_name} {your_name}/{image_name}`
5. Re-start the server again this time system should detect your face also
