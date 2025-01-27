import subprocess

args = [
  "/Users/teli/miniforge3/envs/_learn/bin/python", 
  "blur_amt.py",
  "-o","/Volumes/红色T7/环绕拍摄/orbit_2x_blurd",
  "/Volumes/红色T7/环绕拍摄/orbit_2x"
]

for ii in range(50):
  print(">> ", ii)
  subprocess.call(args)