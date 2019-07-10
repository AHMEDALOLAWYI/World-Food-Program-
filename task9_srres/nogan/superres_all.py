import fastai
from fastai.vision import *
from fastprogress import progress_bar

import warnings

import cv2


warnings.simplefilter("ignore")

path = Path('/home/ubuntu/WFP_Nepal_RGB-Scale3.5_PNG/')

images = ImageList.from_folder(path)
path_test= Path('test')
path_cropped_100 = Path('/home/ubuntu/NepalImages/cropped-100')

print(f'Processing {len(images)} images.')

learn = load_learner(path_test)

def get_data(sz=500):
    #Give it some folder, as long as it exists, it can be empty
    data = (ImageImageList.from_folder('data').split_by_rand_pct(0.1, seed=42)
              .label_from_func(lambda x: x)
              .transform(get_transforms(), size=sz, tfm_y=True)
              .databunch(bs=1).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data
learn.data = get_data((570,570))

def blur_and_sr(path):
    img= cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    
    #3 seems to work best
    blur = cv2.GaussianBlur(img,(3,3),0)
    blur = cv2.resize(blur,(570,570))
    t = Image(tensor(blur/225.).permute(2,0,1).float())
    p,img_hr,b = learn.predict(t)
    return img_hr
for image in progress_bar(images.items):
    dest = path.parent/'Nepal_SR'/image.name
    dest.parent.mkdir(exist_ok=True)
    hr = blur_and_sr(image.as_posix()) 
    cv2.imwrite(dest.as_posix(),cv2.cvtColor(image2np(hr.data*255), cv2.COLOR_RGB2BGR))   