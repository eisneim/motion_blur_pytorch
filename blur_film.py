import os, sys, math, random, time, subprocess
import re, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import cv2
# import matplotlib.pyplot as plt
import bisect
import torch
from torchvision import transforms
import moviepy.editor as mpy
import multiprocessing as mp

DEVICE = "mps" # cuda, mps
# change the path to your downloaded file
film_model_path = "/Users/teli/www/ml/frame_interpolation/frame-interpolation-pytorch-main/pretrained/film_fp16.pt"

film_model = torch.jit.load(film_model_path, map_location='cpu').to(DEVICE)
film_model.half()
film_model.eval()

def get_video_metadata(video_path):
    command = [
        'ffmpeg', '-i', video_path
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    # Extract width, height, and fps from the output
    output = result.stderr
    resolution_pattern = re.compile(r'Stream.*Video.* (\d+)x(\d+)')
    resolution_match = resolution_pattern.search(output)

    if resolution_match:
        width = int(resolution_match.group(1))
        height = int(resolution_match.group(2))
    else:
        raise ValueError("Could not extract video resolution.")

    # Regex to extract fps
    fps_pattern = re.compile(r'(\d+(\.\d+)?) fps')
    fps_match = fps_pattern.search(output)

    if fps_match:
        fps = float(fps_match.group(1))
    else:
        raise ValueError("Could not extract video FPS.")

    return width, height, fps
    

def get_frames(inp: str, w: int= None, h: int = None, start_sec: float = 0, duration: float = None, f: int = None, fps = None) -> np.ndarray:
    args = []
    if duration is not None:
        args += ["-t", f"{duration:.2f}"]
    elif f is not None:
        args += ["-frames:v", str(f)]
    if fps is not None:
        args += ["-r", str(fps)]
    if w is not None:
         args += ["-s", f"{w}x{h}"]
    
    args = ["ffmpeg", "-nostdin", "-ss", f"{start_sec:.2f}", "-i", inp, *args, 
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"]
    if w is None:
        width, height, fps = get_video_metadata(inp)
        w = width
        h = height
    
    process = subprocess.Popen(args, stderr=-1, stdout=-1)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"{inp}: ffmpeg error: {err.decode('utf-8')}")

    process.terminate()
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)

def pil2cv2(pil_img):
    np_image = np.array(pil_img)
    # Convert RGB to BGR
    # return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return np_image[:, :, ::-1].copy()

def np2pil(img):
    return Image.fromarray(np.uint8(img))


# -----------------------------------------
def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region

def load_image_np(image, align=64):
    image = image / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region

def process_frame_pair(frame_pair, inter_frames=10, half=False, device="cuda"):
    img1, img2 = frame_pair[0], frame_pair[1]
    # shape (1, height, width, 3)
    img_batch_1, crop_region_1 = load_image_np(img1)
    img_batch_2, crop_region_2 = load_image_np(img2)

    img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]

        if half:
            x0 = x0.half()
            x1 = x1.half()
        x0 = x0.to(device)
        x1 = x1.to(device)

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = film_model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
        del remains[step]

    y1, x1, y2, x2 = crop_region_1
    frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
    return frames

"""
TODO：
这里可以多线程，多GPU
每两个帧之间的插帧，可以放在一个线程上
"""
def interp_all(frames, down_scale=4, inter_frames=10, device="cuda", num_processes=5):
    interped_frames = []
    height, width = frames[0].shape[0:2]
    hh = height // down_scale
    ww = width // down_scale

    resized = [cv2.resize(ff, (ww, hh)) for ff in frames]
    frame_pairs = [(resized[i], resized[i+1]) for i in range(len(resized)-1)]
    frame_pairs_high = [(frames[i], frames[i+1]) for i in range(len(frames)-1)]

    with mp.Pool(processes=num_processes) as pool:
        # Process frame pairs in parallel, passing additional arguments
        results = pool.starmap(
            process_frame_pair,
            [(pair, inter_frames, True, device) for pair in frame_pairs]
        )

    for idx, group in enumerate(results):
        scale_back = [cv2.cvtColor(cv2.resize(ff, (width, height)), cv2.COLOR_BGR2RGB) for ff in group[1:-1]]
        # blur them
        # scale_back = [ cv2.GaussianBlur(ff, (5, 5), 0)  for ff in scale_back ]
        frame = frame_pairs_high[idx][0]
        frame_next = frame_pairs_high[idx][1]
        # interped_frames.append([frame] + scale_back + [frame_next]) 
        interped_frames.append([frame] + scale_back) 
        
    return interped_frames
# -----------------------------------------

# -----------------------------------------

def build_every_frame(frame_groups, directions, at_onece=False):
    final_frames= []
    """
    TODO：这里可以多线程
    每一个group的长曝光融合应该放在多线程上同时进行
    """
    if not at_onece:
        for ffs in tqdm(frame_groups, "blending"):
            # log_exp_frame = long_exp(ffs)
            log_exp_frame = np.stack(ffs, axis=0).mean(axis=0)
            final_frames.append(log_exp_frame)
            # final_frames.append(merged_mask)
    else:
        # memory intensive operation
        final_frames = np.stack(frame_groups).mean(axis=1)
    return final_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    prog='video trails',
    description='created by Teli, for human that is moving in the video generate motion trails')

    parser.add_argument('files', nargs='+')
    parser.add_argument('-o', '--output_dir', default="./output", type=str, help="output video file dir path")
    parser.add_argument('-d', '--interp_down_scale', default=2, type=int, help="插帧时缩小倍数 越大，速度越快")
    parser.add_argument('-n', '--interp_frames', default=6, type=int, help="两帧之间插帧多少次 ")
    parser.add_argument('-p', '--num_processes', default=8, type=int, help="用多少线程同时进行插帧，请根据显卡内存大小调整，越大越快 ")
    # parser.add_argument('-d', '--degrees', default=4.0, type=float, help="旋转多少角度")

    import psutil

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    files = args.files
    if os.path.isdir(files[0]):
        dirname = args.files[0]
        files = []
        for ii in os.listdir(dirname):
            if ii.endswith(".mp4") and ii[0] != ".":
                files.append(os.path.join(dirname, ii))
        files.reverse()
    print("found video files", len(files))

    for idx, videofile in enumerate(files):
        print(f"memory usage: { psutil.virtual_memory().percent}%")
        assert psutil.virtual_memory().percent < 70, "内存溢出，重启！"
        startTime = time.time()
        print(f"[{idx}/{len(files)}] {videofile}")
        filename = os.path.basename(videofile).rsplit(".", 1)[0] + f"_blur_f{args.interp_frames}s{args.interp_down_scale}.mp4"
        dest = os.path.join(args.output_dir, filename)
        if os.path.exists(dest):
            continue

        frames = get_frames(videofile)[0:-1]
        # frames = get_frames(videofile, w=1920, h=1080)
        # images = [Image.fromarray(img) for img in frames]

        interped_frames_group = interp_all(frames, 
          device=DEVICE, down_scale=args.interp_down_scale, 
          num_processes=args.num_processes,
          inter_frames=args.interp_frames)
        print("interped_frames_group", len(interped_frames_group), f"interp time cost: {time.time() - startTime:.1f}s")

        # directions = calc_directions(frames, masks)
        # print("directions", len(directions))
        directions = []

        final_frames = build_every_frame(interped_frames_group, directions)
        print("final_frames", len(final_frames), final_frames[0].shape)

        clip = mpy.ImageSequenceClip([ff for ff in final_frames], fps=24)
        clip.write_videofile(dest, fps=24, logger=None)
        print(f">> total time cost: {time.time() - startTime:.1f}s")


