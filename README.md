# motion_blur_pytorch
Create smooth FPV drone footage, game recordings, or low-FPS videos using frame blending powered by PyTorch. Runs on Mac, Windows, and Linux.

## Why?:
 - repos like [f0e/blur](https://github.com/f0e/blur) and [smoothie](https://github.com/couleur-tweak-tips/smoothie) can only run on Windows
 - frame interpolation method used in those project can not interpolate large motions, and will produce artifacts

## How?
 - leveraging VFI advancements in recent years, we can improve the quality of frame interpolation by using the latest deep learning models like:
     - [FILM](https://github.com/google-research/frame-interpolation)
     - [AMT](https://nk-cs-zzl.github.io/projects/amt/index.html)
     - [EMA-VFI](https://github.com/mcg-nju/ema-vfi)
     - [PerVFI](https://mulns.github.io/pervfi-page/)
     - [MoMo](https://github.com/JHLew/MoMo)
     - and many more.

## results
| Before(24fps)             | After(24fps)      |
| ----------- | ----------- |
| https://github.com/user-attachments/assets/25977bba-957d-4791-a7c4-e8790152d9cb  | https://github.com/user-attachments/assets/7da1983e-c836-454c-a402-ca0fcabad2a5  |

## insallation
 - A fast GPU is required on **linux** and **windows**, for **Mac** you need to use M1~ M4 series chip with at least 32G RAM
 - first, install pytorch follow the offical guide: [guide](https://pytorch.org/get-started/locally/)
 - install dependencies: **pip install -r requirements.txt**

### FILM
download pretraind FILM model [google drive](https://drive.google.com/file/d/16usfzvVsa0VM2Iy32u1C-C3Rsx8Uz0Lq/view?usp=drive_link) or [百度网盘](https://pan.baidu.com/s/1GPs9ph8JbNQT87eGwUp7rQ?pwd=8pxu) password: 8pxu 

edit file `blur_film.py`
```python
DEVICE = "mps" # cuda, mps
# change the path to your downloaded file
film_model_path = "/Users/teli/www/ml/frame_interpolation/frame-interpolation-pytorch-main/pretrained/film_fp16.pt"

```
process a folder with videos inside
```
python blur_film.py -o output_folder_path input_folder_that_has_mp4_files/ 
```
process multi video files
```
python blur_film.py -o output_folder_path path/to/a.mp4 path/to/b.mp4 path/to/c.mp4
```

## TODO
 - add more models
 
