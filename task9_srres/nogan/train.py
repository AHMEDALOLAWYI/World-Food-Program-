import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.vision.gan import *
from torchvision.models import vgg16_bn

#Change this to where your images are
path = Path('/home/ubuntu/NepalImages')
path_hr= path/'cropped-600'

#its fine to define a path to something that doesn't exist
#path_lr = path/'cropped_into_4_96'
path_lr = path/'cropped-100'


src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data

bs,size=1,400


name_gen = 'image_gen_600'
path_gen = path/name_gen
# shutil.rmtree(path_gen)
path_gen.mkdir(exist_ok=True)

learn=None
gc.collect()

def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data

wd = 1e-3
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)

learn_critic=None
gc.collect()
print("Training critic")
data_crit = get_crit_data([name_gen, 'cropped-600'], bs=bs, size=size)
learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)
learn_critic.fit_one_cycle(6, 1e-3)
learn_critic.save('critic-pretrained-600')
print("Critic is done")

learn_critic=None
learn_gen=None
gc.collect()

bs = 2
# data_crit = get_crit_data(['cropped_into_4_96', 'cropped_into_4'], bs=bs, size=size)
data_crit = get_crit_data(['cropped_100', 'cropped_600'], bs=bs, size=size)
learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pretrained-600')


path_cropped_100 = path/'cropped-100'
data_gen = get_data(bs,size)

##Feature Loss
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
base_loss = F.l1_loss
####


learn_gen  = load_learner(path_cropped_100)
learn_gen.data = data_gen
learn_gen.loss_func = feat_loss

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4
print('Beginning training')
learn.fit(1,lr)
learn_gen.save('gan-gen_600-1')
learn_crit.save('gan-crit_600-1')
#learn.fit(1,lr)
#learn_gen.save('gan-gen_v4-epoch2')
#learn_crit.save('gan-crit_v4-epoch2')
#print('Finished first 5 epochs')
#learn.fit(10,lr)
#learn_gen.save('gan-gen_v4-2')
#learn_crit.save('gan-crit_v4-2')
#print('Finished next 15 epochs.')
#learn.fit(10,lr)
#print('Finished last 10 epochs.')
#learn_gen.save('gan-gen_v4-3')
#learn_crit.save('gan-crit_v4-3')
print('Saved models.')
