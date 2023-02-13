import argparse
import io
import os
import os.path as osp
import pickle
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from zipreader import ZipReader

from main.config import cfg
from main.model import get_model
from main.preprocessing import generate_patch_image, load_img, process_bbox
from main.transforms import cam2pixel, pixel2cam
from main.vis import render_mesh, save_obj, vis_mesh
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


def images_to_video(img_folder, frame_str, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-threads",
        "16",
        "-i",
        f"{img_folder}/{frame_str}",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]

    print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


class ImageFolder(Dataset):
    def __init__(self, image_folder):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
        ]
        self.image_file_names = sorted(self.image_file_names)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        return to_tensor(img)


def load_data(args):

    split_meta_file = os.path.join(args.data_folder, f"{args.split}.pkl")
    with open(split_meta_file, "rb") as f:
        split_meta_dicts = pickle.load(f)
    # keys: keys(['seq_len', 'img_dir', 'name', 'video_file', 'label'])
    return split_meta_dicts


def read_img(frame_path):
    img_data = ZipReader.read(frame_path)
    rgb_im = Image.open(io.BytesIO(img_data)).convert("RGB")
    return rgb_im


def prepare_image_folder(args, vid_dict):
    video_name = vid_dict["name"]
    seq_len = vid_dict["seq_len"]
    img_dir = vid_dict["img_dir"]
    frames_zip_file = args.frames_zip_file

    image_folder = os.path.join("/tmp/MSASL", video_name)
    os.makedirs(image_folder, exist_ok=True)
    frame_paths = [f"{frames_zip_file}@{img_dir}{frame_id:04d}.png" for frame_id in range(seq_len)]

    frame_images = [read_img(frame_path) for frame_path in frame_paths]
    for frame_id, frame_image in enumerate(frame_images):
        frame_image.save(os.path.join(image_folder, f"{frame_id:04d}.png"))
    orig_width, orig_height = frame_images[0].size
    # return video_name, image_folder, seq_len, orig_width, orig_height
    return video_name, image_folder


def define_model(args):
    # SMPL joint set
    # fmt: off
    joint_num = 29 # original: 24. manually add nose, L/R eye, L/R ear
    joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest',
                'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
    flip_pairs = ((1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28))
    skeleton = ((0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14), (14,17),
                (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,24), (24,15),
                (24,25), (24,26), (25,27), (26,28) )
    # fmt: on

    # SMPl mesh
    vertex_num = 6890
    smpl_layer = SMPL_Layer(gender="neutral", model_root="../smplpytorch_models")
    face = smpl_layer.th_faces.numpy()
    joint_regressor = smpl_layer.th_J_regressor.numpy()
    root_joint_idx = 0

    # snapshot load
    model_path = f"./body_snapshot_{args.test_epoch}.pth.tar"
    assert osp.exists(model_path), "Cannot find model at " + model_path
    print("Load checkpoint from {}".format(model_path))
    model = get_model(vertex_num, joint_num, "test")

    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["network"], strict=False)
    model.eval()
    return model, face, joint_regressor, root_joint_idx


def process_one_video(model, image_folder, face, joint_regressor, root_joint_idx):
    # prepare input image
    transform = transforms.ToTensor()
    output_img_dir = f"{image_folder}_output"
    os.makedirs(output_img_dir, exist_ok=True)

    image_names = sorted(os.listdir(image_folder))

    # TODO: save obj per video
    for image_name in image_names:
        image_base_name = image_name.split(".")[0]
        img_path = os.path.join(image_folder, image_name)
        original_img = load_img(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        original_img_height, original_img_width = 256, 256

        bbox = [0, 0, 256, 256]  # xmin, ymin, width, height
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
        )
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]

        # forward
        inputs = {"img": img}
        targets = {}
        meta_info = {"bb2img_trans": bb2img_trans}
        with torch.no_grad():
            out = model(inputs, targets, meta_info, "test")
        img = img[0].cpu().numpy().transpose(1, 2, 0)  # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        mesh_lixel_img = out["mesh_coord_img"][0].cpu().numpy()
        mesh_param_cam = out["mesh_coord_cam"][0].cpu().numpy()

        # restore mesh_lixel_img to original image space and continuous depth space
        mesh_lixel_img[:, 0] = mesh_lixel_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        mesh_lixel_img[:, 1] = mesh_lixel_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        mesh_lixel_img[:, :2] = np.dot(
            bb2img_trans,
            np.concatenate((mesh_lixel_img[:, :2], np.ones_like(mesh_lixel_img[:, :1])), 1).transpose(1, 0),
        ).transpose(1, 0)
        mesh_lixel_img[:, 2] = (mesh_lixel_img[:, 2] / cfg.output_hm_shape[0] * 2.0 - 1) * (cfg.bbox_3d_size / 2)

        # root-relative 3D coordinates -> absolute 3D coordinates
        focal = (1500, 1500)
        princpt = (original_img_width / 2, original_img_height / 2)
        root_xy = np.dot(joint_regressor, mesh_lixel_img)[root_joint_idx, :2]
        root_depth = (
            # 11250.5732421875  # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
            6603.23  # first frame 11250.5732421875/2.5
        )
        root_depth /= 1000  # output of RootNet is millimeter. change it to meter
        root_img = np.array([root_xy[0], root_xy[1], root_depth])
        root_cam = pixel2cam(root_img[None, :], focal, princpt)
        mesh_lixel_img[:, 2] += root_depth
        mesh_lixel_cam = pixel2cam(mesh_lixel_img, focal, princpt)
        mesh_param_cam += root_cam.reshape(1, 3)

        # visualize lixel mesh in 2D space
        vis_img = original_img.copy()
        vis_lixel_img = vis_mesh(vis_img, mesh_lixel_img)
        # cv2.imwrite(f"{output_img_dir}/msasl_test_v{args.video_idx}_f{args.frame_idx}_lixel.jpg", vis_lixel_img)

        # visualize lixel mesh in 2D space
        vis_img = original_img.copy()
        mesh_param_img = cam2pixel(mesh_param_cam, focal, princpt)
        vis_param_img = vis_mesh(vis_img, mesh_param_img)
        # cv2.imwrite(f"{output_img_dir}/msasl_test_v{args.video_idx}_f{args.frame_idx}_param.jpg", vis_param_img)

        # save mesh (obj)
        # save_obj(mesh_lixel_cam, face, f"{output_img_dir}/{image_base_name}_lixel.obj")
        # save_obj(mesh_param_cam, face, f"{output_img_dir}/{image_base_name}_param.obj")

        # render mesh from lixel
        vis_img = original_img.copy()
        rendered_lixel_img = render_mesh(vis_img, mesh_lixel_cam, face, {"focal": focal, "princpt": princpt})
        # cv2.imwrite(f"{output_img_dir}/msasl_test_v{args.video_idx}_f{args.frame_idx}_mesh_lixel.jpg", rendered_lixel_img)

        # render mesh from param
        vis_img = original_img.copy()
        rendered_param_img = render_mesh(vis_img, mesh_param_cam, face, {"focal": focal, "princpt": princpt})
        # cv2.imwrite(f"{output_img_dir}/msasl_test_v{args.video_idx}_f{args.frame_idx}_mesh_param.jpg", rendered_param_img)

        merge_img = np.concatenate(
            [original_img, vis_lixel_img, rendered_lixel_img, vis_param_img, rendered_param_img], axis=1
        )
        cv2.imwrite(f"{output_img_dir}/{image_base_name}_merge.jpg", merge_img)

    return output_img_dir


def convert_to_video(video_name, image_folder, output_save_folder, output_img_folder):
    save_name = f"{video_name}.mp4"
    save_name = os.path.join(output_save_folder, "mesh_videos", save_name)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    images_to_video(img_folder=output_img_folder, frame_str="%04d_merge.jpg", output_vid_file=save_name)
    shutil.rmtree(image_folder)
    shutil.rmtree(output_img_folder)


def main(args):

    demo_output_dir = "./demo_output_msasl"
    os.makedirs(demo_output_dir, exist_ok=True)

    model, face, joint_regressor, root_joint_idx = define_model(args)

    meta_data_dicts = load_data(args)

    for vid_dict in tqdm(meta_data_dicts):
        video_name, image_folder = prepare_image_folder(args, vid_dict)
        output_img_dir = process_one_video(model, image_folder, face, joint_regressor, root_joint_idx)
        convert_to_video(video_name, image_folder, demo_output_dir, output_img_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default="0")
    parser.add_argument("--stage", type=str, dest="stage", default="param")
    parser.add_argument("--test_epoch", type=str, dest="test_epoch", default="7")

    parser.add_argument(
        "--frames_zip_file",
        type=str,
        default="/D_data/SL/data/MSASL/msasl_frames1.zip",
        help="input frames zip file path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="split of the dataset to run the demo on.",
    )
    parser.add_argument("--data_folder", type=str, default="/D_data/SL/data/MSASL/")

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"

    assert args.test_epoch, "Test epoch is required."
    return args


if __name__ == "__main__":
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.stage)
    cudnn.benchmark = True
    main(args)
