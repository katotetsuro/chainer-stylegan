from pathlib import Path

def load_label_image(root):
    images = list(Path(root).glob('**/*.jpg'))
    label_set = set([f.parent.name for f in images])
    label_set = list(label_set)
    label_images = [(i, label_set.index(i.parent.name)) for i in images]
    return label_images