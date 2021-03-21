# Instalation
```
pip install git+https://github.com/anhvth/avcv.git
```
```
git clone https://github.com/anhvth/avcv.git ~/avcv && pip install -e ~/avcv
```
# Usage
Visualize a directory of seg gt
```bash
python -m avcv.run visualize_seg_gt -a ./gt21_9c_ob_extend_test ./vis
```
## Examples
Convert a folder of images to video

Take a look at the function definition by only giving it a function name, ex. images_to_video
```
    python -m avcv.run images_to_video
```
Now give it the positional arguments
```bash
    python -m avcv.run images-to-video  -a /data/fisheye_cams/20201207/timecity_T13_1/ ../output.mp4
```



