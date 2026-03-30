def measure_dice(preds, labels):
    if labels.sum() == 0 and preds.sum() == 0:
        return 1.0

    preds = preds.view(-1)
    labels = labels.view(-1)

    intersection = (preds * labels).sum()
    smooth = 1e-2
    dice = (2. * intersection + smooth) / (preds.sum() + labels.sum() + smooth)

    return dice