from avcv.vision import images_to_video
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', choices=["images-to-video"])
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    if args.task == "images-to-video":
        print("Converting images to videos")
        images_to_video(args.input, args.output)

