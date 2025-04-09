import os
import glob
from PIL import Image
import shutil
import argparse
from tqdm import tqdm


if __name__=="__main__":
    parser = argparse.ArgumentParser('Undersampling')
    parser.add_argument('--ratio', type = int, default=1)
    parser.add_argument('--image_path', type = str, default=None)
    args = parser.parse_args()

    paths = glob.glob(os.path.join(args.image_path,'1x','*'))
    os.makedirs(f'{args.image_path}/{str(args.ratio)}x', exist_ok=True)
    for path in tqdm(paths):
         I = Image.open(path).resize(tuple([int(i/args.ratio) for i in Image.open(path).size]))
         I.save(os.path.join(f'{args.image_path}/{str(args.ratio)}x',os.path.basename(path)))

