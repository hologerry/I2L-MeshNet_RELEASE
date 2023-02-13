import io
import os
import pickle

from PIL import Image

from zipreader import ZipReader


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


def main():
    zip_path = "/D_data/SL/data/MSASL/msasl_frames1.zip"
    split_file = "/D_data/SL/data/MSASL/test.pkl"

    video_ids = [0, 4]
    frame_ids = [0, 10]
    for video_idx in video_ids:
        for frame_idx in frame_ids:
            with open(split_file, "rb") as f:
                split_meta_dicts = pickle.load(f)

            vid_dict = split_meta_dicts[video_idx]
            print("vid_dict", vid_dict)
            img_dir = vid_dict["img_dir"]

            frame_path = f"{zip_path}@{img_dir}{frame_idx:04d}.png"
            img = read_img(frame_path)
            img_out_dir = "demo_output_msasl"
            os.makedirs(img_out_dir, exist_ok=True)
            img.save(f"{img_out_dir}/msasl_test_v{video_idx}_f{frame_idx}.png")


if __name__ == "__main__":
    main()
