import json
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

from flask import Flask

from flask import request, Response, jsonify, send_from_directory, abort
import os

from werkzeug.utils import secure_filename
from flask import send_file


# customize your API through the following parameters
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 80                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# Initialize Flask application
app = Flask(__name__)




app.config['IMAGE_FOLDER'] = os.path.abspath('.') + '\\data/\\' +'\\images\\'


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

name_urls = []
#upload file from client to server
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        for k in request.files:
            file = request.files[k]
            print(file)
            image_urls = []
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['IMAGE_FOLDER'], filename))
                image_urls.append("data/images/%s" % filename)

                # detect
                raw_images = []
                img_raw = tf.image.decode_image(
                    open(image_urls[0], 'rb').read(), channels=3)
                raw_images.append(img_raw)

                num = 0
                # create list for final response
                response = []
                for j in range(len(raw_images)):
                    # create list of responses for current image
                    responses = []
                    raw_img = raw_images[j]
                    num += 1
                    img = tf.expand_dims(raw_img, 0)
                    img = transform_images(img, size)
                    t1 = time.time()
                    boxes, scores, classes, nums = yolo(img)
                    t2 = time.time()
                    print('time: {}'.format(t2 - t1))

                    print('detections:')
                    for i in range(nums[0]):
                        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                    np.array(scores[0][i]),
                                                    np.array(boxes[0][i])))
                        responses.append({
                            "class": class_names[int(classes[0][i])],
                            "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100))
                        })
                    response.append({
                        "image": filename,
                        "detections": responses
                    })
                    img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
                    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                    cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
                    print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))


                try:
                    return jsonify({"response": response}), 200
                except FileNotFoundError:
                    abort(404)

                # name_urls.append("data/images/%s" % filename)

        # return jsonify({"code": 1, "image_urls": image_urls})
# Let file mapping access, otherwise the default can only access files in the static folder
@app.route('/get_image_from_server', methods=['GET'])
def get_image_from_server (imgname):
    return send_from_directory(app.config['IMAGE_FOLDER'], imgname)


# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        # image_name = name_urls[0]
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
    num = 0
    # create list for final response
    response = []
    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    #remove temporary images
    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)

# API that returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()
    
    #remove temporary image
    os.remove(image_name)

    try:

        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)



#upload file from client to server
@app.route('/upload_return_image', methods=['POST'])
def upload_file_return_image():
    if request.method == 'POST':
        for k in request.files:
            file = request.files[k]
            print(file)
            image_urls = []
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['IMAGE_FOLDER'], filename))
                image_urls.append("data/images/%s" % filename)


                # detect
                raw_images = []

                img_raw = tf.image.decode_image(
                    open(image_urls[0], 'rb').read(), channels=3)
                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, size)

                t1 = time.time()
                boxes, scores, classes, nums = yolo(img)
                t2 = time.time()
                print('time: {}'.format(t2 - t1))

                print('detections:')
                for i in range(nums[0]):
                    print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(output_path + 'detection.jpg', img)
                print('output saved to: {}'.format(output_path + 'detection.jpg'))

                # prepare image for response
                _, img_encoded = cv2.imencode('.png', img)
                response = img_encoded.tostring()

                # remove temporary image
                # os.remove(image_name)

                # data = json.loads(response)
                # s = json.dumps(data, indent=4, sort_keys=True)
                try:
                    # return jsonify({"response": s}), 200
                    # return send_file(filename, mimetype='image/gif')
                    return Response(response=response, status=200, mimetype='image/png')

                except FileNotFoundError:
                    abort(404)

                # name_urls.append("data/images/%s" % filename)

        # return jsonify({"code": 1, "image_urls": image_urls})

#demo test connect client - server
@app.route('/detect', methods=['GET'])  # chi ra url cua api
def demo_detect():

    person = {"id": 1,
        "name": "Car",
        "mean": "O to",
        "path": "https://autobikes.vn/stores/news_dataimages/nguyenlien/012019/28/16/0351_D35.jpg"}

    return jsonify(person)



if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)