import torch
from torch import nn
from torch import optim
import dill
import models
import losses
import loaders
from torch.utils.data import DataLoader
import csv


DEVICE = 'cpu'
EPOCHS = 2000
BATCH_SIZE = 16
TRAIN_IMAGES_ROOT = 'data/val'
VAL_IMAGES_ROOT = 'data/train'
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
CSV_PATH = '01-training-log'
EXP_NO = 1
LOAD_CHECKPOINT = None


def save_checkpoint(epoch, generator, discriminator, best_metrics, optimizer_G, lr_scheduler_G,
                    optimizer_D, lr_scheduler_D, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch + 1, 'G_state_dict': generator.state_dict(), 'D_state_dict': discriminator.state_dict(),
             'best_metrics': best_metrics, 'optimizer_G': optimizer_G, 'lr_scheduler_G': lr_scheduler_G,
             'optimizer_D': optimizer_D, 'lr_scheduler_D': lr_scheduler_D}
    torch.save(state, filename, pickle_module=dill)


def add_to_csv(path):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow()


if __name__ == '__main__':
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

    for epoch in range(start_epoch, EPOCHS + 1):

        # Training loop

        # Set the nets into training mode
        G.train()
        D.train()

        tot_trn_loss = 0
        tot_distrn_loss = 0
        count = 0
        for lr_imgs, hr_imgs in trn_dl:

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
            tot_trn_loss += g_loss.item()

            # Unfreeze discriminator, train only the discriminator
            for param in D.parameters():
                param.requires_grad = False

            d_fake_preds = D(fake_imgs.detach()) # detach to avoid backprop into G
            d_real_preds = D(hr_imgs)

            d_fake_loss = adv_loss(d_fake_preds, False)
            d_real_loss = adv_loss(d_real_preds, True)

            d_loss = d_fake_loss + d_real_preds
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            tot_distrn_loss += d_loss.item()
            count += 1

        avg_trn_loss = tot_trn_loss / count
        avg_distrn_loss = tot_distrn_loss / count

        # Validation loop

        # Set the nets into evaluation mode
        G.eval()
        D.eval()

        tot_val_loss = 0
        tot_disval_loss = 0
        count = 0
        for lr_imgs, hr_imgs in val_dl:

            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            fake_imgs = G(lr_imgs)
            cont_loss = content_loss(fake_imgs, hr_imgs)
            mse_loss = MSE(fake_imgs, hr_imgs)
            d_fake_preds = D(fake_imgs)
            g_gan_loss = adv_loss(fake_imgs, True)

            g_loss = CONTENT_LOSS_WEIGHT * cont_loss + MSE_LOSS_WEIGHT * mse_loss + ADVERSARIAL_LOSS_WEIGHT * g_gan_loss
            tot_val_loss += g_loss.item()

            d_real_preds = D(hr_imgs)
            d_loss = adv_loss(fake_imgs, False) + adv_loss(hr_imgs, True)
            tot_disval_loss += d_loss.item()

            count += 1

        # Save best model weights
        avg_val_loss = tot_val_loss / count
        avg_disval_loss = tot_disval_loss / count
        if avg_val_loss < best_val_loss:
            best_val_loss = g_loss.item()
            torch.save(G.state_dict(), f'{EXP_NO:02d}-G_epoch-{epoch:04d}_total-loss-{avg_val_loss:.3f}.pth')
            torch.save(D.state_dict(), f'{EXP_NO:02d}-D_epoch-{epoch:04d}_total-loss-{avg_disval_loss:.3f}.pth')

        sched_G.step()
        sched_D.step()

        save_checkpoint(epoch+1, G, D, None, opt_G, sched_G, opt_D, sched_D)
