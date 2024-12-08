from dotenv import load_dotenv
from fastai.vision.all import *

HOME = os.getenv("HOME")
path = Path(HOME+"/Documents/github/PY/TBPY/labeled-images")

def label_func(fname):
    label = parent_label(fname)
    # print(fname, " -> ", label)
    return label

def get_data(path, presize=128, resize=128):
    dbl =  DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        get_y = label_func,
        splitter=GrandparentSplitter(),
        item_tfms = [Resize(presize, ResizeMethod.Squish)],
    )
    ds = dbl.datasets(path)
    print(len(ds))
    return ds.dataloaders(bs=32)


if __name__ == "__main__":
    load_dotenv()
    # fnames = get_image_files(path)
    # print(len(fnames), "files")
    #
    # dblock = DataBlock()
    # ds = dblock.datasets(fnames)
    # print(ds.train[0])
    #
    # dblock = DataBlock(get_items = get_image_files)
    # ds = dblock.datasets(path)
    # print(ds.train[0])

    dblock = DataBlock(blocks = [ImageBlock, CategoryBlock],
                   get_items = get_image_files,
                   splitter  = RandomSplitter(),
                   item_tfms = [Resize(128)],
                   get_y     = parent_label,
                )

    ds = dblock.datasets(path)
    #ds = dblock.dataloaders(path)

    print(ds.vocab)

    print(ds.train[0])
    # print(ds.valid[0])


    #dls = get_data(path)
    #dls.show_batch()



