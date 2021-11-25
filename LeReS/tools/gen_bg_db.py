from cv2 import data
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
from skimage.segmentation import slic
from skimage.util import img_as_float
from collections import Counter
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch
import h5py

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def predict_depth(image, depth_model):
    resized_image = cv2.resize(image, (448, 448))
    image_torch = scale_torch(resized_image)[None, :, :, :]
    pred_depth = depth_model.inference(image_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (image.shape[1], image.shape[0]))
    return pred_depth_ori.T

def predict_seg(image):
    segment = slic(img_as_float(image), n_segments=10)
    area = []
    label =[]
    s = segment.reshape(segment.shape[1]*segment.shape[0])
    word_count = Counter(s)
    occ = word_count.items()
    for i in occ:
        area.append(i[1])
        label.append(i[0]+1)
    return segment, area, label

def append_to_db(database, image_name, image, depth, seg):
    database.create_dataset("/image/"+image_name, data=image)
    database.create_dataset("/depth/"+image_name, data=depth)
    segment, area, label = seg
    mask_database = database.create_dataset("/seg/"+image_name, data=segment)
    mask_database.attrs['area'] = area
    mask_database.attrs['label'] = label


if __name__ == '__main__':

    args = parse_args()

    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    load_ckpt(args, depth_model, None, None)
    # depth_model.cuda()

    image_dir = os.path.dirname(os.path.dirname(__file__)) + 'test_images/'
    imgs_list = os.listdir(image_dir)
    imgs_list.sort()
    imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    image_dir_out = image_dir + '/outputs'
    os.makedirs(image_dir_out, exist_ok=True)

    database = h5py.File('bg_database.h5','w')
    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        bgr_image = cv2.imread(v)
        image = bgr_image[:, :, ::-1].copy() # BGR -> RGB
        # import ipdb; ipdb.set_trace()
        depth = predict_depth(image, depth_model)
        seg = predict_seg(image)
        
        # segment = cv2.applyColorMap(seg[0].astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imshow('test', segment)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # import ipdb; ipdb.set_trace()

        image_name = os.path.basename(v)
        append_to_db(database, image_name, image, depth, seg)
        

    database.close()