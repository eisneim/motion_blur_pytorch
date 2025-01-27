import os, sys, math, random, time, subprocess
import re, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torch
# from torchvision import transforms
from torchvision.utils import make_grid
import multiprocessing as mp
import moviepy.editor as mpy

from amt.utils.utils import (
    read, write,
    img2tensor, tensor2img,
    check_dim_and_resize, InputPadder
    )

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
# change the path to your downloaded file
amt_model_path = "/Users/teli/www/ml/frame_interpolation/AMT/_pretrained/amt-l.pth"

from amt.AMT_L import Model

amtl = Model(corr_radius=3,
    corr_lvls=4,
    num_flows=5)
ckpt = torch.load(amt_model_path, map_location="cpu")
amtl.load_state_dict(ckpt["state_dict"])
amtl = amtl.to(DEVICE)
amtl.half()
amtl.eval()

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


# -----------------------------------------

def process_frame_pair(frame_pair, iters=4,  device="cuda"):
    img0, img1 = frame_pair[0], frame_pair[1]
    
    img0_t = img2tensor(img0).to(device)
    img1_t = img2tensor(img1).to(device)

    inputs = [img0_t, img1_t]
    
    if device == 'cuda':
        anchor_resolution = 1024 * 512
        anchor_memory = 1500 * 1024**2
        anchor_memory_bias = 2500 * 1024**2
        vram_avail = torch.cuda.get_device_properties(device).total_memory
    else:
        # Do not resize in cpu mode
        anchor_resolution = 8192*8192
        anchor_memory = 1
        anchor_memory_bias = 0
        vram_avail = 1
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).half().to(device)

    inputs = check_dim_and_resize(inputs)
    h, w = inputs[0].shape[-2:]
    scale = anchor_resolution / (h * w) * np.sqrt((vram_avail - anchor_memory_bias) / anchor_memory)
    scale = 1 if scale > 1 else scale
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    if scale < 1:
        print(f"显卡显存限制, 视频将会被缩小 {scale:.2f}倍")
    padding = int(16 / scale)
    padder = InputPadder(inputs[0].shape, padding)
    inputs = padder.pad(*inputs)

    for i in range(iters):
        # print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
        outputs = [inputs[0]]
        for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
            in_0 = in_0.to(device).half()
            in_1 = in_1.to(device).half()
            with torch.no_grad():
                imgt_pred = amtl(in_0, in_1, embt, scale_factor=scale, eval=True)['imgt_pred']
            outputs += [imgt_pred.cpu(), in_1.cpu()]
        inputs = outputs
    outputs = padder.unpad(*outputs)

    frames = []
    for i, imgt_pred in enumerate(outputs):
        imgt_pred = tensor2img(imgt_pred)
        imgt_pred = cv2.cvtColor(imgt_pred, cv2.COLOR_RGB2BGR)
        frames.append(imgt_pred)

    return frames

"""
TODO：
这里可以多线程，多GPU
每两个帧之间的插帧，可以放在一个线程上
"""
def interp_all(frames, down_scale=2, iters=4, device="cuda", num_processes=5):
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
            [(pair, iters, device) for pair in frame_pairs]
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
    parser.add_argument('-n', '--iters', default=3, type=int, help="插帧次数 (最后帧数=2的N次方)")
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
        filename = os.path.basename(videofile).rsplit(".", 1)[0] + f"_blur_n{args.iters}s{args.interp_down_scale}.mp4"
        dest = os.path.join(args.output_dir, filename)
        if os.path.exists(dest):
            continue

        dest2 = os.path.join(args.output_dir, os.path.basename(videofile).rsplit(".", 1)[0] + f"_blur_f6s2.mp4")
        if os.path.exists(dest2):
            continue

        frames = get_frames(videofile)[0:-1]
        # frames = get_frames(videofile, w=1920, h=1080)
        # images = [Image.fromarray(img) for img in frames]

        interped_frames_group = interp_all(frames, 
          device=DEVICE, down_scale=args.interp_down_scale, 
          num_processes=args.num_processes,
          iters=args.iters)
        print("interped_frames_group", len(interped_frames_group), f"interp time cost: {time.time() - startTime:.1f}s")

        # directions = calc_directions(frames, masks)
        # print("directions", len(directions))
        directions = []

        final_frames = build_every_frame(interped_frames_group, directions)
        print("final_frames", len(final_frames), final_frames[0].shape)

        clip = mpy.ImageSequenceClip([ff for ff in final_frames], fps=24)
        clip.write_videofile(dest, fps=24, logger=None)
        print(f">> total time cost: {time.time() - startTime:.1f}s")


