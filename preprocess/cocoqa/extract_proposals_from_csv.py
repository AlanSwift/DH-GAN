"""
    Meta-information:
        train_proposals: "/home/shiina/data/nips/vqa2/features/proposals/train"
        val_proposals: "/home/shiina/data/nips/vqa2/features/proposals/val"
    image_id.pkl
        "features": np.array(NUM_PPL, 2048),
        "spatial_features": np.array(NUM_PPL, 6),
        "num_boxes": int,
        "confidence": np.array(NUM_PPL, 1),
        "original_spatial_features": np.array(NUM_PPL, 4)
"""
import pickle
from tqdm import tqdm
import os
import sys
import numpy as np
import csv
import base64
csv.field_size_limit(sys.maxsize)

PROPOSAL_NUM = 36

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def extract_visual(proposals_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("Start to extract visual features to {}".format(save_dir))

    with open(proposals_path, "r") as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            item['features'] = bytes(item['features'], 'utf')
            features = np.frombuffer(
                        base64.decodebytes(item['features']),
                        dtype=np.float32)

            item['boxes'] = bytes(item['boxes'], 'utf')
            bboxes = np.frombuffer(
                base64.decodebytes(item['boxes']),
                dtype=np.float32)

            item['num_boxes'] = min(int(features.shape[0] / 2048), int(bytes(item["num_boxes"], "utf")))
            item["num_boxes"] = min(item["num_boxes"], int(bboxes.shape[0]/4))



            image_id = int(item['img_id'].split('_')[-1])

            image_w = float(item['img_w'])
            image_h = float(item['img_h'])
            bboxes = np.frombuffer(
                base64.decodebytes(item['boxes']),
                dtype=np.float32).reshape((-1, 4))[:item["num_boxes"], :]
            assert bboxes.shape[0] == item["num_boxes"]

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)
            assert spatial_features.shape[0] == item["num_boxes"]

            original_bbox_infos = bboxes  # xy, zw

            num_boxes = item['num_boxes']

            obj_conf = bytes(item["objects_conf"], "utf")

            obj_conf = np.frombuffer(
                base64.decodebytes(obj_conf),
                dtype=np.float32).reshape((-1, 1))[:item["num_boxes"], :]

            features = np.frombuffer(
                        base64.decodebytes(item['features']),
                        dtype=np.float32).reshape((-1, 2048))[:item["num_boxes"], :]
            assert features.shape[0] == item["num_boxes"]

            objects_id = bytes(item["objects_id"], "utf")
            objects_id = np.frombuffer(
                base64.decodebytes(objects_id), dtype=np.int
            ).reshape((-1))[:item["num_boxes"]]


            output = {
                "features": features,
                "spatial_features": spatial_features,
                "num_boxes": num_boxes,
                "confidence": obj_conf,
                "original_spatial_features": original_bbox_infos,
                "object_id": objects_id
            }
            file_name = str(image_id) + ".pkl"
            with open(os.path.join(save_dir, file_name), "wb") as f:
                pickle.dump(output, f)

    print("Saving visual proposals to {} finish".format(save_dir))

    pass

if __name__ == "__main__":

    train_proposals_path = "/home/shiina/data/acl/train2014_d2obj36_batch.tsv"
    val_proposals_path = "/home/shiina/data/acl/val2014_d2obj36_batch.tsv"

    train_proposals_save_dir = "/home/shiina/data/nips/vqa2/features/proposals/train"
    val_proposals_save_dir = "/home/shiina/data/nips/vqa2/features/proposals/val"

    extract_visual(proposals_path=train_proposals_path, save_dir=train_proposals_save_dir)
    extract_visual(proposals_path=val_proposals_path, save_dir=val_proposals_save_dir)

    pass