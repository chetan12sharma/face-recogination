import argparse
import os
import pickle
import sys

import cv2
import imutils
import numpy as np
import tensorflow as tf
from PIL import Image

import facenet
from align import detect_face

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.compat.v1.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy,
# det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, "align")

# read 20170512-110547 model file downloaded
# from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph(
).get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


def main(args):

    if args.mode == "TRAIN":
        dataset_path = args.data_dir
        dataset = load_data(dataset_path)
        save_face_embedding(dataset)

    elif args.mode == "CLASSIFY":
        testset_path = args.test_image
        name = compare2face(testset_path)
        if name is not -1:
            print(f"Name of the Person :: {name}")
        else:
            print('no data available')


def classify(img):
    testset_path = img
    name = compare2face_api(testset_path)
    if name is not -1:
        print(f"Name of the Person :: {name}")
        return "Hello "+name
    else:
        return "No Match Found"


def compare2face_api(image):
    """
    PARAMS:
        img : image to be search
    RETURN:
        name of the person which have least distance in the embedding face

    """

    dist_list = []  # it holds the euclidean distance between two images
    face2 = getFace(image)
    name = ''

    with open("embedding_file.pkl", "rb") as f:
        emb_file = pickle.load(f)
        for i in emb_file.items():
            face1 = i[1]
            if face1 and face2:
                # calculate Euclidean distance
                dist = np.sqrt(
                    np.sum(
                        np.square(
                            np.subtract(face1[0]["embedding"],
                                        face2[0]["embedding"])
                        )
                    )
                )
                dist_list.append(dist)
        if np.min(dist_list) < 1.10:
            matching_index = np.argmin(dist_list)
            name = list(emb_file.keys())[matching_index]
            return name
        else:
            return -1


def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor
    )
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]: bb[3], bb[0]: bb[2], :]
                resized = cv2.resize(
                    cropped,
                    (input_image_size, input_image_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                prewhitened = facenet.prewhiten(resized)
                faces.append(
                    {
                        "face": resized,
                        "rect": [bb[0], bb[1], bb[2], bb[3]],
                        "embedding": getEmbedding(prewhitened),
                    }
                )
    return faces


def getEmbedding(resized):
    reshaped = resized.reshape(-1, input_image_size, input_image_size, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def load_data(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [
        path
        for path in os.listdir(path_exp)
        if os.path.isdir(os.path.join(path_exp, path))
    ]
    no_of_classes = len(classes)
    for i in range(no_of_classes):
        class_name = classes[i]
        dirc = os.path.join(path_exp, class_name)
        images_path = get_image_paths(dirc)
        dataset.append(images_path)
    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def save_face_embedding(dataset):
    embedding_dict = {}
    for image_path in dataset:
        for i in image_path:
            class_name = i.split("/")[2]
            image = cv2.imread(i)
            emb_array = getFace(image)
            embedding_dict[class_name] = emb_array
    with open("embedding_file.pkl", "wb") as f:
        pickle.dump(embedding_dict, f)
    print(f'embedding file has been saved')

    # name = classify(open_cv_image)


def compare2face(img):
    """
    PARAMS:
        img : image to be search
    RETURN:
        name of the person which have least distance in the embedding face

    """

    dist_list = []  # it holds the euclidean distance between two images
    image = cv2.imread(img)
    face2 = getFace(image)
    name = ''

    with open("embedding_file.pkl", "rb") as f:
        emb_file = pickle.load(f)
        for i in emb_file.items():
            face1 = i[1]
            if face1 and face2:
                # calculate Euclidean distance
                dist = np.sqrt(
                    np.sum(
                        np.square(
                            np.subtract(face1[0]["embedding"],
                                        face2[0]["embedding"])
                        )
                    )
                )
                dist_list.append(dist)

        name = show_image(emb_file, dist_list, face2, image)
    return name


def show_image(emb_file, dist_list, face2, image):
    '''
        Will show the image if match found
    '''
    if np.min(dist_list) < 1.10:
        matching_index = np.argmin(dist_list)
        name = list(emb_file.keys())[matching_index]
        # fc = face co-ordinate
        fc = face2[0]['rect']
        cv2.rectangle(image, (fc[0], fc[1]),
                      (fc[2], fc[3]), (0, 255, 0), 2)
        cv2.putText(image, 'Hello '+name, (fc[0], fc[1]-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.imshow("faces", image)
        cv2.waitKey(0)
    else:
        print('no match found')
        return -1
    return name


def parsing_argument(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["TRAIN", "CLASSIFY"],
        help="Indicates if a new classifier should be trained or a \
            classification model should be used for classification",
        default="CLASSIFY",
    )
    parser.add_argument("--data_dir", type=str,
                        help="Path for the data to be trained")
    parser.add_argument("--test_image", type=str,
                        help="Path for the test image")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parsing_argument(sys.argv[1:]))
    # temp_function()
    # ! For training
    # dataset_path = "./data/"
    # dataset = load_data(dataset_path)
    # get_face_embedding(dataset)
    # ! For Testing
    # testset_path = "./test/anupam.jpg"
    # name = compare2face(testset_path)
    # print(f"Name of the Person :: ", name)
