

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)


weights_dir = "data/retinanet"
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)


# ## vizualize images and biounding boxes1

"""
## Building inference model
"""

image = tf.keras.Input(shape=[None, None, 3], name="image")
# Out[13]: <tf.Tensor 'image_1:0' shape=(None, None, None, 3) dtype=float32>

predictions = model(image, training=False)
# Out[12]: <tf.Tensor 'RetinaNet_1/Identity:0' shape=(None, None, 84) dtype=float32>

threshold = 0.33 #0.5
detections = DecodePredictions(confidence_threshold=threshold)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""

# val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str

val_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*val*.tfrecord'))
val_dataset = tf.data.TFRecordDataset(val_filenames)
val_dataset = val_dataset.map(_parse_function)

for sample in val_dataset.take(2):
    # image = tf.squeeze(tf.cast(sample[0], dtype=tf.float32))

    image = tf.image.decode_jpeg(sample['image'], channels=3)
    image = tf.cast(image, tf.float32)

    input_image, ratio = prepare_image(image)
    #detections = inference_model.predict(tf.squeeze(input_image))
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=[1,0,0], linewidth=2
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": [0,1,0], "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()


plt.close('all')

##read image from secoora sample directory

sample_data_path = 'data/secoora/sample'

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))


plt.figure(figsize=(16,16))

for counter,f in enumerate(sample_filenames):
    image = file2tensor(f)

    image = tf.cast(image, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=[1,0,0], linewidth=2
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": [0,1,0], "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
