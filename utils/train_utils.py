from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.sampling import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid

def get_data(args, rand_set_all=[]):
    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    trans_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
    trans_cifar_val = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = mnist_iid(dataset_train, args.num_users)
            dict_users_test = mnist_iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = mnist_noniid(dataset_train, args.num_users, num_shards=200, num_imgs=300,
                                                    train=True, rand_set_all=rand_set_all)
            dict_users_test, rand_set_all = mnist_noniid(dataset_test, args.num_users, num_shards=200, num_imgs=50,
                                                         train=False, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users_train = cifar10_iid(dataset_train, args.num_users)
            dict_users_test = cifar10_iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = cifar10_noniid(dataset_train, args.num_users, num_shards=200, num_imgs=250,
                                                      train=True, rand_set_all=rand_set_all)
            dict_users_test, rand_set_all = cifar10_noniid(dataset_train, args.num_users, num_shards=200, num_imgs=250,
                                                      train=True, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all

def get_model(args):
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob