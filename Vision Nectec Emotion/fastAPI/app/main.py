from fastapi import FastAPI, File, Form, UploadFile
import io
import cv2
import openpifpaf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageDraw
from typing import List
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse


app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<h2>Try Face Emotion Recognition api - it is just a learning app demo</h2>
<br>by Arnon Monkong"""

app = FastAPI(title='Tensorflow FastAPI Starter Pack', description=app_desc)

@app.get('/')
def get_root():
    return {'message': 'Welcome to the face API'}

model = None

class2idx = {
    "angry":0,
    "disgust":1,
    "fear":2,
    "happy":3,
    "neutral":4,
    "sad":5,
    "surprise":6
}

idx2class = {
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"neutral",
    5:"sad",
    6:"surprise"
}

image_labels_dic_idx2class = {

    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral',
}

def load_model(x):
    if x == 0:
        model = keras.models.load_model('modelANN_NoDrop_optimal.h5')
    else :
        model = keras.models.load_model('modelCNNnoGen_optimal.h5')
    return model

def predict(img):
    global model
    model = load_model(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    crop_img = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        crop_img.append([img[y:y + h, x:x + w], x, y, w, h])

    if len(crop_img) == 0:
        return {"Error": "No face Detected"}

    response = []
    for iteration,j in enumerate(crop_img):
        image = cv2.cvtColor(j[0], cv2.COLOR_BGR2RGB)
        predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-wholebody')
        predictions, gt_anns, meta = predictor.numpy_image(image)
        if predictions == [] :
            response.append({"Error": "OpenPifPaf can't detect face"})
            continue

        faceKey = predictions[0].data[24:92]
        for i in faceKey:
            i[0] = i[0] / j[0].shape[0]
            i[1] = i[1] / j[0].shape[1]
        y_pred = model.predict(faceKey.reshape(1, 204))
        y_pred_class = np.argmax(y_pred, axis=1)
        response.append({"Face_ID": iteration,"Emotion": idx2class[y_pred_class[0]], "X" : j[1].tolist(), "Y" : j[2].tolist(), "Width" : j[3].tolist(), "Height" : j[4].tolist()})
        #response.append({"face": idx2class[y_pred_class[0]]})

    return response

def predictCNN(img):
    global model
    model = load_model(1)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    crop_img = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        crop_img.append([img[y:y + h, x:x + w], x, y, w, h])

    if len(crop_img) == 0:
        return {"Error": "No face Detected"}

    X = []
    response = []

    for image in crop_img:
        resized_img = cv2.resize(image[0], (48, 48))
        X.append(resized_img)
    X = np.array(X)
    X = X / 255
    y_pred = model.predict(X)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_pred_class
    for i, j in enumerate(crop_img):
        response.append({"Emotion": image_labels_dic_idx2class[y_pred_class[i]], "X": j[1].tolist(), "Y": j[2].tolist(),
                         "Width": j[3].tolist(), "Height": j[4].tolist()})

    return response

def imageShow(img,response):
    imgInput = img
    imgPred = response['Predictions']
    for i, d in enumerate(imgPred):
        x1, y1, w, h = d['X'], d['Y'], d['Width'], d['Height']
        cv2.rectangle(imgInput, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(imgInput, d['Emotion'], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return imgInput

@app.post("/faces_recognition/")
async def faces_recognition(image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    #image = cv2.imread(data)
    image = Image.open(io.BytesIO(data))
    image = np.array(image)
    pred = predict(image)
    return pred

@app.post("/faces_recognition_MultiImage/")
async def faces_recognition(image_uploads: List[UploadFile] = File(...)):
    preds = []
    for iteration,image_upload in enumerate(image_uploads):
        data = await image_upload.read()
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        pred = predict(image)
        preds.append({"Image "+str(iteration) :pred })
    return preds

@app.post("/faces_recognition_MultiImage_OpenPifpuf_Model/")
async def faces_recognition(model_id: int,image_uploads: List[UploadFile] = File(...)):
    preds = []
    for iteration,image_upload in enumerate(image_uploads):
        data = await image_upload.read()
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        pred = predict(image)
        preds.append({"Image "+str(iteration) :image_uploads[iteration], "Predictions":pred })
    return preds

@app.post("/faces_recognition_MultiImage_OpenPifPuf_Model_Picture/")
async def faces_recognition(model_id: int,image_uploads: List[UploadFile] = File(...)):
    preds = []
    for iteration,image_upload in enumerate(image_uploads):
        data = await image_upload.read()
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        pred = predict(image)
        preds.append({"Image "+str(iteration) :image_uploads[iteration], "Predictions":pred })
        #ansImg = imageShow(image,preds[0])
        #pathImg = str(image_upload)+"_Answer.jpg"
        #cv2.imwrite(pathImg, image)
        for i, d in enumerate(pred):
            x1, y1, w, h = d['X'], d['Y'], d['Width'], d['Height']
            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(image, d['Emotion'], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res, im_png = cv2.imencode(".png", RGB_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    #return StreamingResponse(io.BytesIO(ansImg.tobytes()), media_type="image/png")
    #return FileResponse(pathImg)

@app.post("/faces_recognition_SingleImage_OpenPifPuf_Model_showAnswer/")
async def faces_recognition(model_id: int,image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    image = Image.open(io.BytesIO(data))
    image = np.array(image)
    pred = predict(image)
    for i, d in enumerate(pred):
        x1, y1, w, h = d['X'], d['Y'], d['Width'], d['Height']
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(image, d['Emotion'], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res, im_png = cv2.imencode(".png", RGB_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/faces_recognition_SingleImage_CNN_Model_showAnswer/")
async def faces_recognition(model_id: int,image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    image = Image.open(io.BytesIO(data))
    image = np.array(image)
    pred = predictCNN(image)
    for i, d in enumerate(pred):
        x1, y1, w, h = d['X'], d['Y'], d['Width'], d['Height']
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(image, d['Emotion'], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res, im_png = cv2.imencode(".png", RGB_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/faces_recognition_MultiImage_CNN_Model/")
async def faces_recognition(model_id: int,image_uploads: List[UploadFile] = File(...)):
    preds = []
    for iteration,image_upload in enumerate(image_uploads):
        data = await image_upload.read()
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        pred = predictCNN(image)
        preds.append({"Image "+str(iteration) :image_uploads[iteration], "Predictions":pred })
    return preds
