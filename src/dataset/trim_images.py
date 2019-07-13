from argparse import ArgumentParser
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image 
from pathlib import Path

def main():
    parser = ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('annotation_dir')
    parser.add_argument('dst')

    args = parser.parse_args()

    images = list(Path(args.image_dir).glob('*.jpg'))
    category_to_dir = {str(d.stem).split('-')[0] : d for d in Path(args.annotation_dir).glob('*') if d.is_dir()}
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    for image in images:
        name = image.stem
        category, _ = name.split('_')
        xml_path = category_to_dir[category].joinpath(name)
        assert xml_path.is_file()
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        if len(objects) == 0:
            print(len(objects))
            continue
        img = Image.open(image)
        for i, o in enumerate(objects):
            category_name = o.find('name').text
            bndbox = o.find('bndbox') 
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            w = np.min((xmax - xmin, ymax - ymin))
            cropped = img.crop((xmin, ymin, xmin+w, ymin+w))
            cropped = cropped.resize((64,64), Image.ANTIALIAS)
            out_filename = image.stem + '_{}'.format(i) + image.suffix
            save_dir = Path(args.dst).joinpath(category_name)
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=False)
            cropped.save(save_dir.joinpath(out_filename))

if __name__ == '__main__':
    main()