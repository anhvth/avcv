import os
from avcv.utils import read_json, mkdir, json, tqdm, shutil
from avcv.vision import plot_images, show
import matplotlib.pyplot as plt


def debug_make_mini_dataset(json_path, image_prefix, out_dir, n=1000, file_name=None):
    print("Making mini dataset", out_dir, "num images:", n)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
    j = read_json(json_path)
    img_ids = list(set([_["image_id"] for _ in j["annotations"]]))
    img_id2path = {_["id"]: _ for _ in j["images"]}
    images = []
    annotations = []
    print("make images")
    n = int(n)

    for image in tqdm(j["images"]):
        if not file_name is None:
            if file_name in image['file_name']: 
                images.append(image)
                file_name = image["file_name"]
                old_path = os.path.join(image_prefix, file_name)
                new_path = os.path.join(out_dir, "images", file_name)
                shutil.copy(old_path, new_path)    
                img_ids = [image['id']]
        else:
            for image["id"] in img_ids[:n]:
                images.append(image)
                file_name = image["file_name"]
                old_path = os.path.join(image_prefix, file_name)
                new_path = os.path.join(out_dir, "images", file_name)
                shutil.copy(old_path, new_path)
    assert len(images), len(images)
    print("make annotations")
    for annotation in tqdm(j["annotations"]):
        if annotation["image_id"] in img_ids[:n]:
            annotations.append(annotation)
    j["images"] = images
    j["annotations"] = annotations
    out_json = os.path.join(out_dir, "annotations", "mini_json.json")
    with open(out_json, "w") as f:
        json.dump(j, f)
    print(out_json)
    return os.path.abspath(out_json)


def vsl(image_or_tensor, order="bhwc", normalize=True, out_file='cache/vsl.jpg'):
    if 'Tensor' in str(type(image_or_tensor)):
        if len(image_or_tensor.shape) == 4 and (image_or_tensor.shape[1] == 1 or image_or_tensor.shape[1]==3):
            image_or_tensor = image_or_tensor.permute([0,2,3,1])
        if len(image_or_tensor.shape) == 3 and (image_or_tensor.shape[0] == 1 or image_or_tensor.shape[1]==3):
            image_or_tensor = image_or_tensor.permute([1,2,0])
            image_or_tensor = image_or_tensor[None]

        images = image_or_tensor.detach().cpu().numpy()
    elif 'ndarray' in str(type(image_or_tensor)):
        if len(image_or_tensor.shape) == 3:
            if (image_or_tensor.shape[0] == 1 or image_or_tensor.shape[1]==3):
                raise NotImplemented
            else:
                image_or_tensor = image_or_tensor[None]
        
        images = image_or_tensor
    elif isinstance(image_or_tensor, list):
        assert isinstance(image_or_tensor[0], np.ndarray)
        images = image_or_tensor
    if normalize:
        outs = []
        for i, image in enumerate(images):
            image = image-image.min()
            image = image/ image.max()
            image *= 255
            image = image.astype('uint8')
            outs += [image]
            images = outs

            
    mkdir('cache')
    if len(images) == 1:
        show(images[0], dpi=150)
        plt.savefig(out_file)
    else:
        plot_images(images, out_file=out_file)
    print(out_file)
