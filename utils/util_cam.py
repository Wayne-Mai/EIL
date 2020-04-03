import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import filters

from scipy.ndimage import label


# def get_cam(model, target=None, input=None, args=None):
#     '''
#     Return CAM tensor which shape is (batch, 1, h, w)
#     '''
#     with torch.no_grad():
#         if input is not None:
#             _ = model(input)

#         if args.distributed:
#             feature_map, score = model.module.get_feature_map()
#             fc_weight = model.module.fc.weight.squeeze()
#             # fc_bias = model.module.fc.bias.squeeze()
#         else:
#             feature_map, score = model.get_feature_map()
#             fc_weight = model.fc.weight.squeeze()
#             # fc_bias = model.fc.bias.squeeze()

#         # print("fc_weight",fc_weight.shape)
#         # 200 x 1024

#         batch, channel, _, _ = feature_map.size()

#         # print(feature_map.shape)

#         # get prediction in shape (batch)
#         if target is None:
#             _, target = score.topk(1, 1, True, True)
#         target = target.squeeze()

#         # print("target",target.shape)
#         # [32]


#         cam_weight = fc_weight[target]
#         # fc_weight[target].shape b * 1024


#         cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(feature_map)
#         print("cam weight",cam_weight.shape)
#         print("feature map",feature_map.shape)
#         cam = (cam_weight * feature_map)
#         # cam_weight b * 1024 * 14 * 14
#         # feature map b * 1024 * 14 *14
#         cam = cam.mean(1).unsqueeze(1)

#     return cam


def get_cam(model, target=None, input=None, args=None):
    '''
    Return CAM tensor which shape is (batch, 1, h, w)
    '''
    with torch.no_grad():
        if input is not None:
            _ = model(input)

        if args.distributed:
            # feature_map, score = model.module.get_cam()
            # fc_weight = model.module.fc.weight.squeeze()
            # fc_bias = model.module.fc.bias.squeeze()
            feature_map, score = model.module.get_fused_cam(target)

        else:
            # deprecated code
            # feature_map, score = model.get_cam()
            # fc_weight = model.fc.weight.squeeze()
            # fc_bias = model.fc.bias.squeeze()
            feature_map, score = model.get_fused_cam(target)

        # print("fc_weight",fc_weight.shape)
        # 200 x 1024

    return feature_map.unsqueeze(1)


def get_heatmap(image, mask, require_norm=False, pillow=False):
    '''
    Return heatmap and blended from image and mask in OpenCV scale
    image : OpenCV scale image with shape (h,w,3)
    mask : OpenCV scale image with shape (h,w)
    isPIL : if True, return PIL scale images with shape(3,h,w)
    '''
    if require_norm:
        mask = mask - np.min(mask)
        mask = mask / np.max(mask) * 255.
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blend = np.float32(heatmap) / 255. + np.float32(image) / 255.
    blend = blend / np.max(blend)
    if pillow:
        return heatmap.transpose(2, 0, 1), blend.transpose(2, 0, 1)
    else:
        return heatmap * 255., blend * 255.


def generate_blend_tensor(image_tensor, mask_tensor):
    '''
    Return a tensor with blended image(image+heatmap)
    image : PIL scale image tensor which shape is (batch, 3, h, w)
    mask : PIL scale image tensor which shape is (batch, 1, h', w')
    heatmap : PIL scale image tensor which shape is (batch, 3, h, w)
    For the WSOL (h,w)/(h', w') is (224, 224)/(14,14), respectively.
    For the WSSS (h,w)/(h', w') is (321, 321)/(41,41), respectively.
    '''
    batch, _, h, w = image_tensor.shape

    image = image_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    mask_tensor = F.interpolate(input=mask_tensor,
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False)
    mask = mask_tensor.cpu().numpy().transpose(0, 2, 3, 1)

    blend_tensor = torch.zeros((batch, 3, h, w))
    for i in range(batch):
        _, blend_map = get_heatmap(image[i] * 255.,
                                   mask[i] * 255.,
                                   require_norm=True,
                                   pillow=True)
        blend_tensor[i] = torch.tensor(blend_map)

    return blend_tensor


def generate_bbox(image, cam, gt_bbox, thr_val, args):
    '''
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1), data type is numpy
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)

    return estimated bounding box, blend image with boxes
    '''
    image_height, image_width, _ = image.shape
    # print("here get image shape",image_height,image_width)
    # print("but input atten shape",cam.shape)

    _gt_bbox = list()
    _gt_bbox.append(max(int(gt_bbox[0]), 0))
    _gt_bbox.append(max(int(gt_bbox[1]), 0))
    _gt_bbox.append(min(int(gt_bbox[2]), image_height-1))
    _gt_bbox.append(min(int(gt_bbox[3]), image_width))

    cam = cv2.resize(cam, (image_height, image_width),
                     interpolation=cv2.INTER_CUBIC)
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_cls), cv2.COLORMAP_JET)
        

    blend = image * 0.6 + heatmap_BGR * 0.4
    # gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    gray_heatmap = intensity_to_gray(cam, normalize=True)
    #
    # temp1=thr_val * np.max(gray_heatmap)
    # temp2=filters.threshold_local(gray_heatmap,35)

    thr_val = thr_val * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_BINARY)

    if args.bbox_mode=='classical':

        dt_gray_heatmap = thr_gray_heatmap

        # Wayne added, try a distance transform here
        # thr_gray_heatmap = cv2.distanceTransform(thr_gray_heatmap, cv2.DIST_L2, 3)
        # thr_gray_heatmap = intensity_to_gray(
        #     thr_gray_heatmap, normalize=True, _sqrt=True)
        # _, dt_gray_heatmap = cv2.threshold(thr_gray_heatmap,
        #                                    10, 255,
        #                                    cv2.THRESH_BINARY)
        try:
            _, contours, _ = cv2.findContours(dt_gray_heatmap,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours(dt_gray_heatmap,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

        _img_bbox = (image.copy()).astype('uint8')

        blend_bbox = blend.copy()
        cv2.rectangle(blend_bbox,
                    (_gt_bbox[0], _gt_bbox[1]),
                    (_gt_bbox[2], _gt_bbox[3]),
                    (0, 0, 255), 2)

        # may be here we can try another method to do
        # TODO
        # threshold all the box and then merge it
        # and then rank it,
        # finally merge it
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            estimated_bbox = [x, y, x + w, y + h]
            cv2.rectangle(blend_bbox,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0), 2)
        # estimated_bboxs = []
        # if len(contours) != 0:
        #     for c in contours:
        #         x, y, w, h = cv2.boundingRect(c)
        #         estimated_bbox = [x, y, x + w, y + h]
        #         estimated_bboxs.append(estimated_bbox)
        else:
            estimated_bbox = [0, 0, 1, 1]
            # estimated_bboxs = [estimated_bbox]

        # estimated_bboxs = sorted(estimated_bboxs, key=lambda x: (
        #     x[3]-x[1])*(x[2]-x[0]), reversed=True)

        # estimated_bbox = estimated_bboxs[0]
        # for box in estimated_bboxs:
        #     t = 0.5*max((estimated_bbox[3]-estimated_bbox[1])*(estimated_bbox[2] -
        #                                                    estimated_bbox[0]), (box[3]-box[1])*(box[2]-box[0]))
        #     x_overlap = max(
        #         0, min(estimated_bbox[2], box[2])-max(estimated_bbox[0], box[0]))
        #     y_overlap = max(
        #         0, min(estimated_bbox[3], box[3])-max(estimated_bbox[1], box[1]))
        #     if x_overlap*y_overlap>t:
        #         do merge

        return estimated_bbox, blend_bbox

    elif args.bbox_mode=='DANet': # mode is union
        def extract_bbox_from_map(boolen_map):
            assert boolen_map.ndim == 2, 'Invalid input shape'
            rows = np.any(boolen_map, axis=1)
            cols = np.any(boolen_map, axis=0)
            if rows.max() == False or cols.max() == False:
                return 0, 0, 0, 0
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            # here we modify a box to a list 
            return [xmin, ymin, xmax, ymax]
        
        # thr_gray_map is a gray map
        estimated_bbox = extract_bbox_from_map(thr_gray_heatmap)
        blend_bbox = blend.copy()
        cv2.rectangle(blend_bbox,
                    (_gt_bbox[0], _gt_bbox[1]),
                    (_gt_bbox[2], _gt_bbox[3]),
                    (0, 0, 255), 2)
        cv2.rectangle(blend_bbox,
                        (estimated_bbox[0], estimated_bbox[1]),
                        (estimated_bbox[2], estimated_bbox[3]),
                        (0, 255, 0), 2)
        return estimated_bbox, blend_bbox







def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    for i in range(len(rect)):
        area = rect[i][2] * rect[i][3]
        if large_area < area:
            large_area = area
            target = i

    x = rect[target][0]
    y = rect[target][1]
    w = rect[target][2]
    h = rect[target][3]

    return x, y, w, h


def get_bbox(image, cam, thresh, gt_box, image_name, save_dir='test', isSave=False):
    gxa = int(gt_box[0])
    gya = int(gt_box[1])
    gxb = int(gt_box[2])
    gyb = int(gt_box[3])

    image_size = 224
    adjusted_gt_bbox = []
    adjusted_gt_bbox.append(max(gxa, 0))
    adjusted_gt_bbox.append(max(gya, 0))
    adjusted_gt_bbox.append(min(gxb, image_size-1))
    adjusted_gt_bbox.append(min(gyb, image_size-1))
    '''
    image: single image, shape (224, 224, 3)
    cam: single image, shape(14, 14)
    thresh: the floating point value (0~1)
    '''
    # resize to original size
    # image = cv2.resize(image, (224, 224))
    cam = cv2.resize(cam, (image_size, image_size))

    # convert to color map
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')

    # blend the original image with estimated heatmap
    blend = image * 0.5 + heatmap * 0.5

    # initialization for boundary box
    bbox_img = image.astype('uint8').copy()
    blend = blend.astype('uint8')
    blend_box = blend.copy()
    # thresholding heatmap
    gray_heatmap = cv2.cvtColor(heatmap.copy(), cv2.COLOR_RGB2GRAY)
    th_value = np.max(gray_heatmap) * thresh

    _, thred_gray_heatmap = \
        cv2.threshold(gray_heatmap, int(th_value),
                      255, cv2.THRESH_BINARY)
    try:
        _, contours, _ = \
            cv2.findContours(thred_gray_heatmap, cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = \
            cv2.findContours(thred_gray_heatmap, cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)

    # calculate bbox coordinates

    rect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect.append([x, y, w, h])
    if len(rect) == 0:
        estimated_box = [0, 0, 1, 1]
    else:
        x, y, w, h = large_rect(rect)
        estimated_box = [x, y, x + w, y + h]

        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(blend_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.rectangle(bbox_img, (adjusted_gt_bbox[0], adjusted_gt_bbox[1]),
                  (adjusted_gt_bbox[2], adjusted_gt_bbox[3]), (0, 0, 255), 2)
    cv2.rectangle(blend_box, (adjusted_gt_bbox[0], adjusted_gt_bbox[1]),
                  (adjusted_gt_bbox[2], adjusted_gt_bbox[3]), (0, 0, 255), 2)
    concat = np.concatenate((bbox_img, heatmap, blend), axis=1)

    if isSave:
        if not os.path.isdir(os.path.join('image_path/', save_dir)):
            os.makedirs(os.path.join('image_path', save_dir))
        cv2.imwrite(os.path.join(os.path.join('image_path/',
                                              save_dir,
                                              image_name.split('/')[-1])), concat)
    blend_box = cv2.cvtColor(blend_box, cv2.COLOR_BGR2RGB).copy()

    return estimated_box, adjusted_gt_bbox, blend_box


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def intensity_to_gray(intensity, normalize=True, _sqrt=False):
    assert intensity.ndim == 2

    if _sqrt:
        intensity = np.sqrt(intensity)

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    intensity = np.uint8(255*intensity)
    return intensity


def main():
    return


if __name__ == '__main__':
    main()
