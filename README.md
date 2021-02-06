# Instalation

pip install git+https://github.com/anhvth/avcv.git

# Usage
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
