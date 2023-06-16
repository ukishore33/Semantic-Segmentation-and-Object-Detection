import os
import cv2
import copy
import openai
import numpy as np
import matplotlib.pyplot as plt


def load_yolo_model():
    """
    Load the YOLO model.

    Returns:
        tuple: The loaded YOLO model and output layers.
    """
    # load yolo model
    net = cv2.dnn.readNet('yolov3_parameters\\yolov3.weights', 'yolov3_parameters\\yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    return net, output_layers


def load_image(image_path, quality=100):
    """
    Load an image from a file and save it with lower quality.

    Args:
        image_path (str): The path to the image file.
        quality (int): The quality of the image to be saved. It can be a number between 0 and 100.

    Returns:
        ndarray: The loaded image.
    """
    # load image
    img = cv2.imread(image_path)

    # encoding parameters with quality level
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # encode image
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    
    img = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    
    return img


def detect_objects(net, output_layers, img):
    """
    Detect objects in the image using the YOLO model.

    Args:
        net: The loaded YOLO model.
        output_layers (list): The output layers of the model.
        img (ndarray): The image in which to detect objects.

    Returns:
        tuple: The output from the YOLO model, and the width and height of the image.
    """
    img = copy.deepcopy(img)
    
    height, width, channels = img.shape

    # detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    return outs, width, height


def get_classes_colors():
    """
    Get the class labels and colors for bounding boxes.

    Returns:
        tuple: The class labels and the colors for bounding boxes.
    """
    # load classes from the YOLO dataset
    with open('yolov3_parameters/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # create color array for bounding boxes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return classes, colors


def process_detections(outs, width, height, classes, colors, img):
    """
    Process the output from the YOLO model.

    Args:
        outs: The output from the YOLO model.
        width (int): The width of the image.
        height (int): The height of the image.
        classes (list): The class labels.
        colors (list): The colors for bounding boxes.
        img (ndarray): The image.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected object.
    """
    img = copy.deepcopy(img)
    
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []

    # process each detection output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  # class with highest score
            confidence = scores[class_id]  # confidence of best class
            
            # process only if confidence is greater than 0.5
            if confidence > 0.5:
                # get center, width and height of each of bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # calculate the coordinates of the top left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # append information
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    # draw bounding box and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # draw bounding box
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)  # display class label

            # normalize values for center, width and height of the bounding box
            center_x_norm = (x + w / 2) / width
            center_y_norm = (y + h / 2) / height
            width_norm = w / width
            height_norm = h / height

            # make dictionary of detected object and append it to list
            detected_objects.append({
                'label': label,
                'confidence': confidences[i],
                'location': {
                    'center': (center_x_norm, center_y_norm),
                    'width': width_norm,
                    'height': height_norm
                }
            })

    return detected_objects


def plot_image_with_detections(img, detected_objects, width, height):
    """
    Plot the original image and the image with detected objects side by side.

    Args:
        img (ndarray): The original image.
        detected_objects (list): List of detected objects.
        width (int): Width of the image.
        height (int): Height of the image.
    """
    
    img = copy.deepcopy(img)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot the original img
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')

    # plot img with detected objects
    image_with_detections = img.copy()
    for obj in detected_objects:
        label = obj['label']
        confidence = obj['confidence']
        location = obj['location']
        x = int(location['center'][0] * width - location['width'] * width / 2)
        y = int(location['center'][1] * height - location['height'] * height / 2)
        w = int(location['width'] * width)
        h = int(location['height'] * height)
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_detections, f'{label}: {confidence:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    ax[1].imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Image with Detected Objects')

    # remove axis labels
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()