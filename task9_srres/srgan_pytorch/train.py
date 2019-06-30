import models
import losses
import loaders
from torch.utils.data import DataLoader


DEVICE = 'cpu'
EPOCHS = 1000
BATCH_SIZE = 32
TRAIN_IMAGES_ROOT = 'data/val'
VAL_IMAGES_ROOT = 'data/train'
WORKERS = 8
HR_PATCH = (512, 512)
SCALE = 2
VGG_FEATURE_LAYER = 34


def save_model():
    pass


if __name__ == '__main__':
    trn_ds = loaders.SatelliteDataset(TRAIN_IMAGES_ROOT, HR_PATCH, scale_factor=SCALE)
    trn_dl = DataLoader(trn_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    val_ds = loaders.SatelliteValDataset(VAL_IMAGES_ROOT, HR_PATCH, scale_factor=SCALE)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Generator
    G = models.GeneratorBNFirst(3, 3, upscale=SCALE)
    G.to(DEVICE)

    # Discriminator
    D = models.Discriminator(64, HR_PATCH[0], sigmoid=True)
    D.to(DEVICE)

    # Losses
    content_loss = losses.ContentLoss(VGG_FEATURE_LAYER, 'l2')
    content_loss.to(DEVICE)
    adv_loss = losses.AdversarialLoss()
    adv_loss.to(DEVICE)

    for epoch in range(EPOCHS):
        pass
        # Training loop

        # Validation loop
