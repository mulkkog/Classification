def choose_nets(nets_name, num_classes=100):
    nets_name = nets_name.lower()
    if nets_name == 'googlenet':
        from models.GoogLeNet import GoogLeNet
        return GoogLeNet(num_classes)
    raise NotImplementedError
