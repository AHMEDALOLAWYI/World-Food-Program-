import models
import loaders
import vat
import metrics
import torch
from torch import nn
from torchvision import transforms
from torchvision import models as visionmodels
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import dill
from tqdm import tqdm

EXP_NO = 1
CROP_SIZE = 512
IMAGE_SIZE = 224
DEGREES_AUGMENT = 10
BRIGHTNESS_JITTER = 0.15
LR = 3e-4
BATCH_SIZE = 32
LBL_DATA_DIR = 'lbl_data_resized/'
# LBL_DATA_DIR = 'lbl_data_sample/'
UNLBL_DATA_DIR = 'unlbl_data/data/hr/'
# UNLBL_DATA_DIR = 'unlbl_data/'
SPLIT_CSV = 'train_test_split_clean.csv'
POSITIVE_CLASS = 'rice'
WORKERS = 8
DEVICE = 'cuda'
XI = 10.0
EPSILON = 1.0
IP = 1
LR_STEP = 5
LR_DECAY = 0.8
EPOCHS = 100
MODEL_NAME = 'Resnet50'
TENSORBOARD_LOGDIR = f'{EXP_NO:02d}-{MODEL_NAME}-tboard'
LOAD_CHECKPOINT = None
WEIGHTS_SAVE_PATH = f'{EXP_NO:02d}-{MODEL_NAME}-weights'


trn_tfms = transforms.Compose([
                transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(DEGREES_AUGMENT),
                transforms.ColorJitter(brightness=BRIGHTNESS_JITTER),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

val_tfms = transforms.Compose([
                transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class BookKeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['ce', 'vat', 'total', 'f1', 'accuracy']
        self.genesis()
        # Initialize tensorboard objects
        self.tboard = dict()
        if tensorboard_log_path is not None:
            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)
            for name in self.loss_names:
                self.tboard[name] = SummaryWriter(os.path.join(tensorboard_log_path, name + '_' + suffix))

    def genesis(self):
        self.losses = {key: 0 for key in self.loss_names}
        self.count = 0

    def update(self, **kwargs):
        for key in kwargs:
            self.losses[key] += kwargs[key]
        self.count += 1

    def reset(self):
        self.genesis()

    def get_avg_losses(self):
        avg_losses = dict()
        for key in self.loss_names:
            avg_losses[key] = self.losses[key] / self.count
        return avg_losses

    def update_tensorboard(self, epoch):
        avg_losses = self.get_avg_losses()
        for key in self.loss_names:
            self.tboard[key].add_scalar(key, avg_losses[key], epoch)


def pbar_desc(label, epoch, total_epochs, loss_val, acc_val, f1_val):
    return f'{label}: {epoch:04d}/{total_epochs} | loss: {loss_val:.3f} acc: {acc_val:.3f} f1:{f1_val:.3f}'


def save_checkpoint(epoch, model, best_metrics, optimizer, lr_scheduler, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_metrics': best_metrics,
             'optimizer': optimizer, 'lr_scheduler': lr_scheduler
             }
    torch.save(state, filename, pickle_module=dill)


def main():
    best_val_acc = -1.0
    start_epoch = 1

    trn_ds = loaders.SatClassificationDataset(LBL_DATA_DIR, SPLIT_CSV, POSITIVE_CLASS, False, trn_tfms)
    print('Train Samples:', len(trn_ds))
    trn_dl = DataLoader(trn_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    unlbl_ds = loaders.UnlabeledDataset(UNLBL_DATA_DIR, IMAGE_SIZE)
    print('Unlabeled:', len(unlbl_ds))
    unlbl_dl = DataLoader(unlbl_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    val_ds = loaders.SatClassificationDataset(LBL_DATA_DIR, SPLIT_CSV, POSITIVE_CLASS, True, val_tfms)
    print('Val Samples:', len(val_ds))
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    model = models.Resnet(visionmodels.resnet50, 2)
    model.to(DEVICE)

    ce_loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    vat_loss_fn = vat.VATLoss(IP, EPSILON, XI).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, LR_STEP, gamma=LR_DECAY)

    trn_metrics = BookKeeping(TENSORBOARD_LOGDIR, 'trn')
    val_metrics = BookKeeping(TENSORBOARD_LOGDIR, 'val')

    if not os.path.exists(WEIGHTS_SAVE_PATH):
        os.mkdir(WEIGHTS_SAVE_PATH)

    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module=dill)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']

        lr_sched = checkpoint['lr_scheduler']
        best_val_acc = checkpoint['best_metrics']

    for epoch in range(start_epoch, EPOCHS + 1):

        # Train
        t_pbar = tqdm(trn_dl, desc=pbar_desc('train', epoch, EPOCHS, 0.0, -1.0, -1.0))
        ul_iter = iter(unlbl_dl)

        model.train()
        for (xs, ys) in t_pbar:
            try:
                xs_ul, ys_ul = next(ul_iter)
            except StopIteration:
                # Reset the iterator in case we've used
                # up all of the images
                ul_iter = iter(unlbl_dl)
                xs_ul, ys_ul = next(ul_iter)

            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)

            y_pred1 = model(xs)
            ce_loss = ce_loss_fn(y_pred1, ys)

            xs_ul = xs_ul.to(DEVICE)
            vat_loss = vat_loss_fn(xs_ul, model, logits=True)

            total_loss = ce_loss + vat_loss

            acc = metrics.accuracy(y_pred1, ys)
            f1 = metrics.f1_score(y_pred1, ys)

            trn_metrics.update(ce=ce_loss.item(), vat=vat_loss.item(), total=total_loss.item(),
                               f1=f1.item(), accuracy=acc.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            t_pbar.set_description(pbar_desc('train', epoch, EPOCHS, total_loss.item(), acc, f1))

        # Final update to training bar
        avg_trn_metrics = trn_metrics.get_avg_losses()
        t_pbar.set_description(pbar_desc('train', epoch, EPOCHS, avg_trn_metrics['total'],
                                         avg_trn_metrics['accuracy'], avg_trn_metrics['f1']))
        trn_metrics.update_tensorboard(epoch)

        # Validate
        v_pbar = tqdm(val_dl, desc=pbar_desc('valid', epoch, EPOCHS, 0.0, -1.0, -1.0))
        model.eval()

        for xs, ys in v_pbar:

            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)

            y_pred1 = model(xs)
            ce_loss = ce_loss_fn(y_pred1, ys)

            acc = metrics.accuracy(y_pred1, ys)
            f1 = metrics.f1_score(y_pred1, ys)

            val_metrics.update(ce=ce_loss.item(), vat=0, total=ce_loss.item(),
                               f1=f1.item(), accuracy=acc.item())

            v_pbar.set_description(pbar_desc('valid', epoch, EPOCHS, ce_loss.item(), acc, f1))

        avg_val_metrics = val_metrics.get_avg_losses()
        avg_acc = avg_val_metrics['accuracy']
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(model.state_dict(),
                       f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-{MODEL_NAME}_epoch-{epoch:04d}_acc-{avg_acc:.3f}.pth')

        # Final update to validation bar
        v_pbar.set_description(pbar_desc('train', epoch, EPOCHS, avg_val_metrics['total'],
                                         avg_val_metrics['accuracy'], avg_val_metrics['f1']))
        val_metrics.update_tensorboard(epoch)

        # Update scheduler and save checkpoint
        lr_sched.step(epoch=epoch)
        save_checkpoint(epoch, model, best_val_acc, optimizer, lr_sched)


if __name__ == '__main__':
    main()
