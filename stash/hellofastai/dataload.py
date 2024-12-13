
from fastai.vision.all import *
from dotenv import load_dotenv
from pprint import pp

HOME = os.getenv("HOME")

def is_cat(x): return x[0].isupper()

def do_cats_dogs():
    print("getting data from",  URLs.PETS)
    path = untar_data(URLs.PETS)
    print("saved to",  path)

    files = get_image_files(path/"images")
    print(len(files), "files")

    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path),
        valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224))
    return

def do_bears():
    path = Path(HOME+"/data/bears")
    files = get_image_files(path)
    failed = verify_images(files)
    print("failed", len(failed))
    failed.map(Path.unlink)

    print(len(files))
    print(files[0], files[-1])

    bears = DataBlock(blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(128)])

    dls = bears.dataloaders(path)
    # dls.valid.show_batch(max_n=4, nrows=1)

    """
    """
    return


def get_data(url, presize, resize):
    path = untar_data(url)
    block = (DataBlock(blocks=[ImageBlock, CategoryBlock],
                      get_items=get_image_files,
                      splitter=GrandparentSplitter(valid_name='val'),
                      get_y=parent_label,
                      item_tfms=[Resize(presize)],
                      batch_tfms=[*aug_transforms(min_scale=0.5, size=resize),
                                  Normalize.from_stats(*imagenet_stats)]).
             dataloaders(path, bs=128))
    return block

def do_imaginette1():
    dls = get_data(URLs.IMAGENETTE_160, 160, 128)
    print(type(dls.valid))
    return

# https://docs.fast.ai/tutorial.imagenette.html
lbl_dict = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute'
)

def label_func(fname):
    return lbl_dict[parent_label(fname)]

def load_with_data_block(url):
    path = untar_data(url)
    #fnames = get_image_files(path)
    # print(parent_label(fnames[0]))
    # By itself, a DataBlock is just a blueprint on how to assemble your data. It does not do anything until you pass it a source.
    dblock = DataBlock(blocks=[ImageBlock, CategoryBlock],
                       get_items = get_image_files,
                       get_y = label_func,
                       splitter  = GrandparentSplitter(),
                       item_tfms = [RandomResizedCrop(128, min_scale=0.35)],
                       batch_tfms=Normalize.from_stats(*imagenet_stats)
                       )
    dsets = dblock.datasets(path)
    #print(dsets.vocab)
    return dsets

def do_imaginette2():
    ds  = load_with_data_block(URLs.IMAGENETTE_160)
    #print("training set", len(ds.train))
    #print("valid set", len(ds.valid))
    # By default, the data block API assumes we have an input and a target, which is why we see our filename repeated twice.
    #print(ds.train[0])
    #print(ds.valid[0])
    return

def do_imagenette3():
    # Another way to compose several functions for get_y is to put them in a Pipeline:Another way to compose several
    # functions for get_y is to put them in a Pipeline:
    url = URLs.IMAGENETTE_160
    path = untar_data(url)
    imagenette = DataBlock(blocks = [ImageBlock, CategoryBlock],
                           get_items = get_image_files,
                           get_y = Pipeline([parent_label, lbl_dict.__getitem__]),
                           splitter = GrandparentSplitter(valid_name='val'),
                           item_tfms = [RandomResizedCrop(128, min_scale=0.35)],
                           batch_tfms = Normalize.from_stats(*imagenet_stats))
    dls = imagenette.dataloaders(path)

def do_stocks():
    dir = os.path.expandvars("$HOME/Documents/github/PY/TBPY/labeled-images")
    path = Path(dir)
    labels = [p.name for p in path.ls()]
    print(labels)
    dls = get_data2(path)
    print(dls.vocab[1])
    return

def get_data2(path, presize=256, resize=224):
    dls = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        get_y = parent_label,
        splitter=RandomSplitter(),
        item_tfms = [Resize(presize, ResizeMethod.Squish)],
    ).dataloaders(path, bs=32)
    return dls


if __name__ == "__main__":
    load_dotenv()
    # do_cats_dogs()
    # do_bears()
    # do_imaginette2()
    do_stocks()

