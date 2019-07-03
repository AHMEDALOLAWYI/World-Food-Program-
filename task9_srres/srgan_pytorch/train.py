import torch
from torch import nn
from torch import optim
from torchvision import transforms
import dill
import models
import losses
import loaders
from torch.utils.data import DataLoader
import csv
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm


DEVICE = 'cuda'
EPOCHS = 5
BATCH_SIZE = 2
TRAIN_IMAGES_ROOT = 'data/train_sample'
VAL_IMAGES_ROOT = 'data/val_sample'
WORKERS = 8
HR_PATCH = (512, 512)
SCALE = 2
VGG_FEATURE_LAYER = 34
LR_DECAY = 0.5
LR_STEP = 500
LR_G = 1e-4
LR_D = 1e-4
CONTENT_LOSS_WEIGHT = 2e-6
ADVERSARIAL_LOSS_WEIGHT = 1e-3
MSE_LOSS_WEIGHT = 1.0
EXP_NO = 1
LOAD_CHECKPOINT = None
TENSORBOARD_LOGDIR = 'tboard-sampleset'
END_EPOCH_SAVE_SAMPLES_PATH = 'epoch_end_samples-sampleset'


# Too many losses to keep track of
# Put everyone in a single place
class BookKeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['content', 'mse', 'adversarial',
                           'generator', 'discriminator']
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


def save_checkpoint(epoch, generator, discriminator, best_metrics, optimizer_G, lr_scheduler_G,
                    optimizer_D, lr_scheduler_D, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch, 'G_state_dict': generator.state_dict(), 'D_state_dict': discriminator.state_dict(),
             'best_metrics': best_metrics, 'optimizer_G': optimizer_G, 'lr_scheduler_G': lr_scheduler_G,
             'optimizer_D': optimizer_D, 'lr_scheduler_D': lr_scheduler_D}
    torch.save(state, filename, pickle_module=dill)


def add_to_csv(path):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow()


def pbar_desc(label, epoch, total_epochs, loss_val):
    return f'{label}: {epoch:04d}/{total_epochs} | {loss_val:.3f}'


def save_images(path, lr_images, fake_hr, hr_images, epoch):

    images_path = os.path.join(path, f'{epoch:04d}')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    to_pil = transforms.ToPILImage()

    for i, tensor in enumerate(lr_images):
        image = to_pil(tensor)
        image.save(f'{images_path}/{i:02d}_lr.jpg', 'JPEG')

    for i, tensor in enumerate(fake_hr):
        image = to_pil(tensor)
        image.save(f'{images_path}/{i:02d}_fake.jpg', 'JPEG')

    for i, tensor in enumerate(hr_images):
        image = to_pil(tensor)
        image.save(f'{images_path}/{i:02d}_hr.jpg', 'JPEG')


def train(G, D, trn_dl, epoch, epochs, content_loss, MSE, adv_loss, opt_G, opt_D, train_losses):
    # Set the nets into training mode
    G.train()
    D.train()

    t_pbar = tqdm(trn_dl, desc=pbar_desc('train', epoch, epochs, 0.0))
    for lr_imgs, hr_imgs in t_pbar:

        # Send the images onto the appropriate device
        lr_imgs = lr_imgs.to(DEVICE)
        hr_imgs = hr_imgs.to(DEVICE)

        # Freeze discriminator, train generator
        for param in D.parameters():
            param.requires_grad = False

        fake_imgs = G(lr_imgs)
        cont_loss = content_loss(fake_imgs, hr_imgs)
        mse_loss = MSE(fake_imgs, hr_imgs)
        # Get predictions from discriminator
        d_fake_preds = D(fake_imgs)
        # Train the generator to generate fake images
        # such that the discriminator recognizes as real
        g_adv_loss = adv_loss(d_fake_preds, True)

        g_loss = CONTENT_LOSS_WEIGHT * cont_loss + MSE_LOSS_WEIGHT * mse_loss + ADVERSARIAL_LOSS_WEIGHT * g_adv_loss
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # Unfreeze discriminator, train only the discriminator
        for param in D.parameters():
            param.requires_grad = True

        d_fake_preds = D(fake_imgs.detach())  # detach to avoid backprop into G
        d_real_preds = D(hr_imgs)

        d_loss = adv_loss(d_fake_preds, False) + adv_loss(d_real_preds, True)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        t_pbar.set_description(pbar_desc('train', epoch, EPOCHS, g_loss.item()))
        train_losses.update(content=cont_loss.item(), mse=mse_loss.item(), adversarial=g_adv_loss.item(),
                            generator=g_loss.item(), discriminator=d_loss.item())


def evaluate(G, D, val_dl, epoch, epochs, content_loss, MSE, adv_loss, val_losses, best_val_loss):
    # Set the nets into evaluation mode
    G.eval()
    D.eval()

    v_pbar = tqdm(val_dl, desc=pbar_desc('valid', epoch, epochs, 0.0))
    for lr_imgs, hr_imgs in v_pbar:
        lr_imgs = lr_imgs.to(DEVICE)
        hr_imgs = hr_imgs.to(DEVICE)

        fake_imgs = G(lr_imgs)
        cont_loss = content_loss(fake_imgs, hr_imgs)
        mse_loss = MSE(fake_imgs, hr_imgs)
        d_fake_preds = D(fake_imgs)
        g_adv_loss = adv_loss(d_fake_preds, True)

        g_loss = CONTENT_LOSS_WEIGHT * cont_loss + MSE_LOSS_WEIGHT * mse_loss + ADVERSARIAL_LOSS_WEIGHT * g_adv_loss

        d_real_preds = D(hr_imgs)
        d_loss = adv_loss(d_fake_preds, False) + adv_loss(d_real_preds, True)

        val_losses.update(content=cont_loss.item(), mse=mse_loss.item(), adversarial=g_adv_loss.item(),
                          generator=g_loss.item(), discriminator=d_loss.item())
        v_pbar.set_description(pbar_desc('valid', epoch, EPOCHS, g_loss.item()))

    # Save best model weights
    avg_val_losses = val_losses.get_avg_losses()
    avg_val_loss = avg_val_losses['generator']
    avg_disval_loss = avg_val_losses['discriminator']
    if avg_val_loss < best_val_loss:
        best_val_loss = g_loss.item()
        torch.save(G.state_dict(), f'{EXP_NO:02d}-G_epoch-{epoch:04d}_total-loss-{avg_val_loss:.3f}.pth')
        torch.save(D.state_dict(), f'{EXP_NO:02d}-D_epoch-{epoch:04d}_total-loss-{avg_disval_loss:.3f}.pth')

    return best_val_loss


def main():
    trn_ds = loaders.SatelliteDataset(TRAIN_IMAGES_ROOT, HR_PATCH, scale_factor=SCALE)
    trn_dl = DataLoader(trn_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    val_ds = loaders.SatelliteValDataset(VAL_IMAGES_ROOT, HR_PATCH, scale_factor=SCALE)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    start_epoch = 1
    best_val_loss = float('inf')

    # Generator
    G = models.GeneratorBNFirst(3, 3, upscale=SCALE)
    opt_G = optim.Adam(G.parameters(), lr=LR_G)
    sched_G = optim.lr_scheduler.StepLR(opt_G, LR_STEP, gamma=LR_DECAY)

    # Discriminator
    D = models.Discriminator(64, HR_PATCH[0], sigmoid=True)
    opt_D = optim.Adam(D.parameters(), lr=LR_D)
    sched_D = optim.lr_scheduler.StepLR(opt_D, LR_STEP, gamma=LR_DECAY)

    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module=dill)
        start_epoch = checkpoint['epoch']

        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])

        opt_G = checkpoint['optimizer_G']
        opt_D = checkpoint['optimizer_D']

        sched_G = checkpoint['lr_scheduler_G']
        sched_D = checkpoint['lr_scheduler_D']

        best_val_loss = checkpoint['best_metrics']

    G.to(DEVICE)
    D.to(DEVICE)

    # Losses
    content_loss = losses.ContentLoss(VGG_FEATURE_LAYER, 'l2')
    content_loss.to(DEVICE)
    adv_loss = losses.AdversarialLoss()
    adv_loss.to(DEVICE)
    MSE = nn.MSELoss()
    MSE.to(DEVICE)

    train_losses = BookKeeping(TENSORBOARD_LOGDIR, suffix='trn')
    val_losses = BookKeeping(TENSORBOARD_LOGDIR, suffix='val')

    for epoch in range(start_epoch, EPOCHS + 1):

        # Training loop
        train(G, D, trn_dl, epoch, EPOCHS, content_loss, MSE, adv_loss, opt_G, opt_D, train_losses)

        # Validation loop
        best_val_loss = evaluate(G, D, val_dl, epoch, EPOCHS, content_loss, MSE, adv_loss, val_losses, best_val_loss)

        sched_G.step()
        sched_D.step()

        save_checkpoint(epoch, G, D, None, opt_G, sched_G, opt_D, sched_D)

        train_losses.update_tensorboard(epoch)
        val_losses.update_tensorboard(epoch)

        # Reset all loss for a new epoch
        train_losses.reset()
        val_losses.reset()

        # Save a real vs fake batch for quality inspection
        lrs, hrs = next(iter(val_dl))
        fakes = G(lrs.to(DEVICE))

        # Save samples at the end
        save_images(END_EPOCH_SAVE_SAMPLES_PATH, lrs.detach().cpu(), fakes.detach().cpu(), hrs, epoch)


if __name__ == '__main__':
    main()
