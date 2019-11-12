# -*- coding: utf-8 -*-
# @Time    : 18-7-10 上午9:41
# @Author  : zengzihua@huya.com
# @FileName: benchmark.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os
import math
import time

from scipy.ndimage.filters import gaussian_filter


def cal_coord(pred_heatmaps, pad_x1, pad_y1, pad_x2, pad_y2, images_anno):
    coords = {}
    for img_id in pred_heatmaps.keys():
        heat_h, heat_w, n_kpoints = pred_heatmaps[img_id].shape
        img_h = images_anno[img_id]['height']
        img_w = images_anno[img_id]['width']
        pad_left = pad_x1[img_id]
        pad_top = pad_y1[img_id]
        pad_right = pad_x2[img_id]
        pad_bottom = pad_y2[img_id]
        coord = []
        for p_ind in range(n_kpoints):
            heat = pred_heatmaps[img_id][:, :, p_ind]
            #heat = gaussian_filter(heat, sigma=5)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            coord_x = (ind[1] + 0.5) / heat_w * (img_w+pad_left+pad_right) - 0.5 - pad_left
            coord_y = (ind[0] + 0.5) / heat_h * (img_h+pad_top+pad_bottom) - 0.5 - pad_top
            coord.append((coord_x, coord_y, heat[ind[0],ind[1]]))
        coords[img_id] = coord
    return coords


def infer(frozen_pb_path, output_node_name, img_path, images_anno):
    with tf.gfile.GFile(frozen_pb_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name("image:0")
    output_heat = graph.get_tensor_by_name("%s:0" % output_node_name)

    res = {}
    pad_x1 = {}
    pad_y1 = {}
    pad_x2 = {}
    pad_y2 = {}
    use_times = []
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for img_id in images_anno.keys():
            ori_img = cv2.imread(os.path.join(img_path, images_anno[img_id]['file_name']))
            shape = input_image.get_shape().as_list()
            ori_im_width = ori_img.shape[1]
            ori_im_height = ori_img.shape[0]
            scale_x = float(ori_im_width) / float(shape[1])
            scale_y = float(ori_im_height) / float(shape[2])
            max_scale = max(scale_x,scale_y)
            max_side_x = int(max_scale * shape[1]+0.5)
            max_side_y = int(max_scale * shape[2]+0.5)
           
            pad_top = (max_side_y - ori_im_height)//2
            pad_bottom = max_side_y - ori_im_height - pad_top
            pad_left = (max_side_x - ori_im_width)//2
            pad_right = max_side_x - ori_im_width - pad_left
            pad_img = cv2.copyMakeBorder(ori_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            #pad_img = cv2.copyMakeBorder(ori_img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
            inp_img = cv2.resize(pad_img, (shape[1], shape[2]))
            st = time.time()
            heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            infer_time = 1000 * (time.time() - st)
            print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
            res[img_id] = np.squeeze(heat)
            pad_x1[img_id] = pad_left
            pad_y1[img_id] = pad_top
            pad_x2[img_id] = pad_right
            pad_y2[img_id] = pad_bottom
    print("Average inference time = %.2f ms" % np.mean(use_times))
    return res,pad_x1,pad_y1,pad_x2,pad_y2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCKh benchmark")
    parser.add_argument("--frozen_pb_path", type=str, default="")
    parser.add_argument("--anno_json_path", type=str, default="")
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--output_node_name", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    anno = json.load(open(args.anno_json_path))
    print("Total test example=%d" % len(anno['images']))

    images_anno = {}
    keypoint_annos = {}
    transform = list(zip(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    ))
    for img_info, anno_info in zip(anno['images'], anno['annotations']):
        images_anno[img_info['id']] = img_info

        prev_xs = anno_info['keypoints'][0::3]
        prev_ys = anno_info['keypoints'][1::3]

        new_kp = []
        for idx, idy in transform:
            new_kp.append(
                (prev_xs[idx-1], prev_ys[idy-1])
            )

        keypoint_annos[anno_info['image_id']] = new_kp

    pred_heatmap,pad_x1,pad_y1,pad_x2,pad_y2 = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)
    pred_coords = cal_coord(pred_heatmap, pad_x1,pad_y1,pad_x2,pad_y2,images_anno)

    scores = []
    scores_dot5 = []
    scores_dot2 = []
    scores_dot2_thresh_dot3 = []
    for img_id in keypoint_annos.keys():
        groundtruth_anno = keypoint_annos[img_id]

        head_gt = groundtruth_anno[0]
        neck_gt = groundtruth_anno[1]

        threshold = math.sqrt((head_gt[0] - neck_gt[0]) ** 2 + (head_gt[1] - neck_gt[1]) ** 2)

        curr_score = []
        curr_score_dot5 = []
        curr_score_dot2 = []
        curr_score_dot2_thresh_dot3 = []
        for index, coord in enumerate(pred_coords[img_id]):
            pred_x, pred_y, conf = coord
            gt_x, gt_y = groundtruth_anno[index]

            d = math.sqrt((pred_x-gt_x)**2 + (pred_y-gt_y)**2)
            if d > threshold:
                curr_score.append(0)
            else:
                curr_score.append(1)
            if d > threshold*0.5:
                curr_score_dot5.append(0)
            else:
                curr_score_dot5.append(1)
            if d > threshold*0.2:
                curr_score_dot2.append(0)
            else:
                curr_score_dot2.append(1)
            if conf < 0.3 or d > threshold*0.2:
                curr_score_dot2_thresh_dot3.append(0)
            else:
                curr_score_dot2_thresh_dot3.append(1)
            
        scores.append(np.mean(curr_score))
        scores_dot5.append(np.mean(curr_score_dot5))
        scores_dot2.append(np.mean(curr_score_dot2))
        scores_dot2_thresh_dot3.append(np.mean(curr_score_dot2_thresh_dot3))

    print("PCKh@1.0=%.2f" % (np.mean(scores) * 100))
    print("PCKh@0.5=%.2f" % (np.mean(scores_dot5) * 100))
    print("PCKh@0.2=%.2f" % (np.mean(scores_dot2) * 100))
    print("PCKh@0.2@thresh0.3=%.2f" % (np.mean(scores_dot2_thresh_dot3) * 100))

