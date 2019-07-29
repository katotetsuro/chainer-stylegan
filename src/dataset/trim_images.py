from argparse import ArgumentParser
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image 
from pathlib import Path
import cv2
from os.path import join
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def main():
    blacklist = [
        'African_hunting_dog',
        'basenji',
        'black-and-tan_coonhound',
        'borzoi',
        'Bouvier_des_Flandres'
    ]

    parser = ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('annotation_dir')
    parser.add_argument('dst')
    parser.add_argument('--save-image', action='store_true')
    parser.add_argument('--xml-dir', default='/opt/conda/lib/python3.6/site-packages/cv2/data/')

    args = parser.parse_args()

    images = list(Path(args.image_dir).glob('*.jpg'))
    category_to_dir = {str(d.stem).split('-')[0] : d for d in Path(args.annotation_dir).glob('*') if d.is_dir()}
    Path(args.dst).mkdir(parents=True, exist_ok=True)

    detectors = [
        cv2.CascadeClassifier(join(args.xml_dir, 'haarcascade_frontalcatface_extended.xml')),
#        cv2.CascadeClassifier(join(args.xml_dir, 'haarcascade_frontalcatface.xml'))
    ]

    ret = []
    categories = []

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
        # s = min(img.width, img.height)
        # org = img.crop((0, 0, s, s))
        # org = org.resize((64,64), Image.ANTIALIAS)
        # if args.save_image:
        #     org.save(Path(args.dst).joinpath(image.name))
        # ret.append(np.asanyarray(org))
        
        for i, o in enumerate(objects):
            category_name = o.find('name').text
            bndbox = o.find('bndbox') 
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            dx = xmax - xmin
            dy = ymax - ymin
            # if min(dx, dy)/max(dx, dy) < 0.6:
            #     continue
            w = np.min((dx, dy))
            cropped = img.crop((xmin, ymin, xmin+w, ymin+w))
            cropped = cropped.resize((64,64), Image.ANTIALIAS)
            out_filename = image.stem + '_{}'.format(i) + image.suffix
            if args.save_image:
                cropped.save(Path(args.dst).joinpath(out_filename))

            if category_name in categories:
                label = categories.index(category_name)
            else:
                label = len(categories)
                categories.append(category_name)
            ret.append((np.asarray(cropped).transpose(2, 0, 1).astype(np.float32), np.array(label, np.int32)))

            cropped = img.crop((xmin, ymin, xmax, ymax))
            grayimg = cv2.cvtColor(np.asarray(cropped), cv2.COLOR_RGB2GRAY)
            for d in detectors:
                pos = d.detectMultiScale(grayimg)
                if len(pos) != 0:
                    pos = pos[0]
                    left, top = pos[0], pos[1]
                    size = min(pos[2], pos[3])
                    cropped = cropped.crop((left, top, left+size, top+size))
                    cropped = cropped.resize((64,64), Image.ANTIALIAS)
                    ret.append((np.asarray(cropped).transpose(2, 0, 1).astype(np.float32), np.array(label, np.int32)))
                    break

    print(len(ret))
    np.save(Path(args.dst).joinpath('data.npy'), ret, allow_pickle=True)

def load_dataset(image_dir, annotation_dir):
    blacklist = [
        'African_hunting_dog',
        'basenji',
        'black-and-tan_coonhound',
        'borzoi',
        'Bouvier_des_Flandres'
    ]

    images = list(Path(image_dir).glob('*.jpg'))
    category_to_dir = {str(d.stem).split('-')[0] : d for d in Path(annotation_dir).glob('*') if d.is_dir()}
    
    ret = []
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
            bndbox = o.find('bndbox') 
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            w = np.min((xmax - xmin, ymax - ymin))
            cropped = img.crop((xmin, ymin, xmin+w, ymin+w))
            cropped = cropped.resize((64,64), Image.ANTIALIAS)
            ret.append(np.asarray(cropped))

    ret = np.asanyarray(ret)
    ret = ret.transpose(0, 3, 1, 2).astype(np.float32)
    return ret

if __name__ == '__main__':
    main()