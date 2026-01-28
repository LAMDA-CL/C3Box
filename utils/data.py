import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


# class iCIFAR100(iData):
#     use_path = False
#     train_trsf = [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=63 / 255),
#         transforms.ToTensor()
#     ]
#     test_trsf = [transforms.ToTensor()]
#     common_trsf = [
#         transforms.Normalize(
#             mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
#         ),
#     ]

#     class_order = np.arange(100).tolist()

#     def download_data(self):
#         train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
#         test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
#         self.train_data, self.train_targets = train_dataset.data, np.array(
#             train_dataset.targets
#         )
#         self.test_data, self.test_targets = test_dataset.data, np.array(
#             test_dataset.targets
#         )
class iCIFAR100(iData):
    use_path = False
    # Clip preprocess transforms
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self,preprocess=None):
       # trainset = datasets.CIFAR100(root='"./data"', train=True, download=True)
       # testset = datasets.CIFAR100(root='"./data"', train=False, download=True)
        trainset = datasets.CIFAR100(root="[DATA-PATH]/train/", train=True, download=False)
        testset = datasets.CIFAR100(root="[DATA-PATH]/val/", train=False, download=False)
        self.train_data, self.train_targets = trainset.data, np.array(trainset.targets)
        self.test_data, self.test_targets = testset.data, np.array(testset.targets)


def build_transform_vit(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    return t

def build_transform(is_train, args):
    input_size = 224
    t=[  
        transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    return t

class iCIFAR224(iData):
    use_path = False

    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [ ]
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("[DATA-PATH]/train/", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("[DATA-PATH]/val/", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]


    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir ="[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir ="[DATA-PATH]/train/"
        test_dir ="[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir ="[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Caltech101(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir ="[DATA-PATH]/train/"
        test_dir ="[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Food101(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir ="[DATA-PATH]/train/"
        test_dir ="[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Flowers(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir ="[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Aircraft(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class UCF101(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir ="[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class StanfordCars(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir ="[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class TV100(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir ="[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class SUN(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [  ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir =" [DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class omnibenchmark(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
   
