

import torchvision.transforms.v2 as T

# Define transforms with color/lighting augmentation
transform_train = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = T.Compose([
    T.Resize((192, 224)),  # height, width
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
