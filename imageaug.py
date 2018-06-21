from torchvision import transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def transform_for_training(image_shape):
    return transforms.Compose(
      [
        transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4
        ),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)
      ]
    )


def transform_for_infer(image_shape):
    return transforms.Compose(
      [
        transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)
      ]
    )

