import os, pandas, tqdm, argparse, openslide, re
import torch

from hipt_model_utils import get_vit256, eval_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path to directory with csv files of regions coordinates')
parser.add_argument('--output', type=str, required=True, help='path to directory save dataset')
parser.add_argument('--weights', type=str, required=True, help='path to vit small weights')
parser.add_argument('--wsi', type=str, required=True, help='path WSIs folder')
parser.add_argument('--level', type=int, required=True, help='level to extract patches')
parser.add_argument('--gpu', type=str, default='', help='GPU to use (e.g. 0)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
print('device: ', device)

model256 = get_vit256(pretrained_weights=args.weights, device=device)
model256 = model256.to(device=device)
print('model loaded')

for file_ in tqdm.tqdm(os.listdir(args.path), ncols=50):
    
    patient = re.findall('\d+', file_)[0]
    slide = os.path.splitext(file_)[0]
    
    df = pandas.read_csv(os.path.join(args.path, file_))

    wsi = openslide.OpenSlide(os.path.join(args.wsi, '.'.join((slide, 'tif'))))

    for region in df['region'].unique():
        region_patches = df[df['region'] == region]
        X, Y = [], []
        for _, patch in region_patches.iterrows():

            img = wsi.read_region((patch['x'], patch['y']), args.level, (patch['size'], patch['size'])).convert('RGB')
            label = patch['cancerous']

            X.append(img)
            Y.append(label)

        X = map(eval_transforms(), X)
        X = torch.stack(list(X), dim=0).to(device=device)
        with torch.no_grad():
            X = model256(X)

        Y = torch.tensor(Y, dtype=torch.uint8)

        path_ = os.path.join(args.output, patient, slide, str(region))
        os.makedirs(path_, exist_ok=True)
        torch.save(X.to(device='cpu'), os.path.join(path_, 'x.pt'))
        torch.save(Y, os.path.join(path_, 'y.pt'))
