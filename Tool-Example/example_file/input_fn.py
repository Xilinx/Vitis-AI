import cv2
from sklearn import preprocessing

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]

def resize_shortest_edge(image, size):
  H, W = image.shape[:2]
  if H >= W:
    nW = size
    nH = int(float(H)/W * size)
  else:
    nH = size
    nW = int(float(W)/H * size)
  return cv2.resize(image,(nW,nH))

def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image

def BGR2RGB(image):
  B, G, R = cv2.split(image)
  image = cv2.merge([R, G, B])
  return image

def central_crop(image, crop_height, crop_width):
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2
  return image[offset_height:offset_height + crop_height, offset_width:
               offset_width + crop_width, :]

def normalize(image):
  image=image/256.0
  image=image-0.5
  image=image*2
  return image


#def preprocess_fn(image, crop_height, crop_width):
#    image = resize_shortest_edge(image, 256)
#    image = mean_image_subtraction(image, MEANS)
#    image = central_crop(image, crop_height, crop_width)
#    return image 

eval_batch_size = 1
def eval_input(iter, eval_image_dir, eval_image_list, class_num):
    images = []
    labels = []
    line = open(eval_image_list).readlines()
    for index in range(0, eval_batch_size):
        curline = line[iter * eval_batch_size + index]
        [image_name, label_id] = curline.split(' ')
        image = cv2.imread(eval_image_dir + image_name)
        image = central_crop(image, 224, 224)
        image = mean_image_subtraction(image, MEANS)
        images.append(image)
        labels.append(int(label_id))
    lb = preprocessing.LabelBinarizer()
    lb.fit(range(0, class_num))
    labels = lb.transform(labels)
    return {"input": images, "labels": labels}


calib_image_dir = "images/"
calib_image_list = "images/tf_calib.txt"
calib_batch_size = 10
def calib_input(iter):
    images = []
    line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        image = cv2.imread(calib_image_dir + calib_image_name)  
        image = resize_shortest_edge(image, 256)
        image = mean_image_subtraction(image, MEANS)
        image = central_crop(image, 224, 224)
        images.append(image)
    return {"input": images}

