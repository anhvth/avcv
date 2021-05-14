from avcv import *

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
