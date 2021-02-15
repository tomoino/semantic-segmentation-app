"""Data Utils"""

from pathlib import Path

def make_data_list(dataroot: str, annotationroot: str, annotationpath: str, img_extension='.jpg', anno_extension='.png'):
    """Make data list from dataroot and labelroot
    Parameters
    ----------
    dataroot : str
        a path to the image directory
    annotationroot : str
        a path to the annotation directory
    annotationpath : str
        a path to the text file of annotation
    Returns
    -------
    Tuple of list
        img_list: e.g. ['./data/images/car1.jpg', './data/images/dog4.jpg', ...]
        annot_list: e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
    """

    img_dir = dataroot
    annot_dir = annotationroot

    train_filenames = annotationpath

    # create train img and annotation path list
    img_list = []
    annot_list = []

    for line in open(annotationpath):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = dataroot + '/' + img_fname
        anno_fname = line + anno_extension
        annot_path = annotationroot + '/' + anno_fname
        img_list.append(str(img_path))
        annot_list.append(str(annot_path))

    return (img_list, annot_list)