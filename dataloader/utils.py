"""Data Utils"""

from pathlib import Path

def make_datapath_list(rootpath, train_data='train.txt', test_data='test.txt', img_extension='.jpg', anno_extension='.png'):
    """
    Create list of image and annotation data path
    Parameters
    ----------
    rootpath : str
        path to the data directory
    train_data : str
        text file with train filename
    test_data : str
        text file with test filename
    img_extension : str
        extension of image
    anno_extension : str
        extension of annotation
    Returns
    ----------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
    """

    img_dir = Path(rootpath) / 'JPEGImages'
    annot_dir = Path(rootpath) / 'SegmentationClass'

    train_filenames = Path(rootpath) / 'ImageSets' / 'Segmentation' / train_data
    test_filenames = Path(rootpath) / 'ImageSets' / 'Segmentation' / test_data

    # create train img and annot path list
    train_img_list = []
    train_annot_list = []

    for line in open(train_filenames):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        anno_fname = line + anno_extension
        annot_path = annot_dir / anno_fname
        train_img_list.append(str(img_path))
        train_annot_list.append(str(annot_path))

    # create test img and annot path list
    test_img_list = []
    test_annot_list = []

    for line in open(test_filenames):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        anno_fname = line + anno_extension
        annot_path = annot_dir / anno_fname
        test_img_list.append(str(img_path))
        test_annot_list.append(str(annot_path))

    return (train_img_list, train_annot_list, test_img_list, test_annot_list)