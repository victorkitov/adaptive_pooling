import torch
import torch.nn as nn
from torch import optim
import alexnet_torchvision
from VGG_modified import vgg
from alexnet_modified import alexnet
from cnn_model import CNN
import yaml
import typing as tp
import click
from tqdm import tqdm

from traininig_utils import DatasetLoader, TBLogger


def load_model(model_name: str, pooling: str, num_classes: int, adaptive: bool, concat: bool = False):
    if model_name == 'alexnet':
        model = alexnet(pooling, num_classes)
    elif model_name == 'vgg':
        print(f"concat = {concat}")
        model = vgg(pooling, num_classes, adaptive=adaptive, concat=concat)
    elif model_name == 'custom_cnn':
        model = CNN(pooling, num_classes)
    else:
        raise NotImplementedError
    return model


def train(experiment_name: str, model_name: str, pooling: str, adaptive: bool,
          dataset: tp.Dict, num_epochs: int, lr: float, concat: bool = False, device_name="cuda:0"):
    device = torch.device(device_name)
    dataset_loader = DatasetLoader(**dataset)
    loaders = dataset_loader.create_dataloaders()
    writer = TBLogger(experiment_name)

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    # loss_func = nn.MSELoss()

    model = load_model(model_name, pooling, dataset_loader.num_classes, adaptive, concat=concat)
    # path = "/home/olya/cmc/vgg16_bn-6c64b313.pth"
    # model.load_state_dict(torch.load(path, map_location=device), strict=False)
    # return
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.to(device)

    total_step = len(loaders['train'])
    loss_count = 0
    accuracy_count = 0
    step = 0
    t1 = []
    t2 = []
    t3 = []
    
    c1 = []
    c2 = []
    c3 = []

    import time
    start = time.time()
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in enumerate(loaders['train']):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            output, probas = model(images)
            # print(output.shape, probas.shape, labels.shape, probas.shape, .shape)
            loss = loss_func(output, labels)

            # ohe_labels = nn.functional.one_hot(labels).type(torch.float32)
            # print(output.dtype, ohe_labels.dtype)
            # loss = loss_func(output, ohe_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss_count = loss + loss_count / 100
            # accuracy_count = ((torch.argmax(output, 1) == labels) * 1.).mean()
            step += 1
            
            # if (i + 1) % 100 == 0:
            #     writer.add_scalar('Loss/train', loss, step)
            #     writer.add_scalar('Accuracy/train', accuracy_count, step)

        model.eval()
        accuracy_test = 0
        loss_test = 0
        with torch.no_grad():   
        #     start = time.time()
        # #
        #     for images, labels in loaders['test']:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         output, probas = model(images)
        #         accuracy_test += ((torch.argmax(output, 1) == labels) * 1.).mean()
        #         loss_test += loss_func(output, labels)
        #     print((time.time() - start) / len(loaders['test']))
            T1, T2, T3 = model.pool1.pool_coef, model.pool2.pool_coef, model.pool3.pool_coef
            # T1, T2, T3 = model.pool1.temperature, model.pool2.temperature, model.pool3.temperature
            # T1, T2, T3 = model.pool1.maxpool_coef, model.pool2.maxpool_coef, model.pool3.maxpool_coef
            # C1, C2, C3 = model.pool1.avgpool_coef, model.pool2.avgpool_coef, model.pool3.avgpool_coef
        # writer.add_scalar('Loss/test', loss_test / len(loaders['test']), step)
        # writer.add_scalar('Accuracy/test', accuracy_test / len(loaders['test']), step)
        # writer.add_scalar('Temperature/pool_1', T1.item(), step)
        # writer.add_scalar('Temperature/pool_2', T2.item(), step)
        # writer.add_scalar('Temperature/pool_3', T3.item(), step)
        t1.append(T1.item())
        t2.append(T2.item())
        t3.append(T3.item())
        
        # c1.append(C1.item())
        # c2.append(C2.item())
        # c3.append(C3.item())
    # torch.save(model.state_dict(), "./weights/cifar10_max")
    # print(time.time() - start)
        import numpy as np
        np.save("./params2/mix2_a1", np.array(c1))
        np.save("./params2/mix2_a2", np.array(c2))
        np.save("./params2/mix2_a3", np.array(c3))

        # np.save("./params2/mix1_m1", np.array(t1))
        # np.save("./params2/mix1_m2", np.array(t2))
        # np.save("./params2/mix1_m3", np.array(t3))

        # np.save("./params/new_mix1_a_1", np.array(c1))
        # np.save("./params/new_mix1_a_2", np.array(c2))
        # np.save("./params/new_mix1_a_3", np.array(c3))

def load_config(config_name: str, key: str) -> tp.Dict:
    with open(config_name, "r") as f:
        full_config = yaml.safe_load(f)
    return full_config[key]


@click.command()
@click.option("-e", "--experiment_name", type=str, help="Experiment name from config")
def main(experiment_name: str):
    config = './config.yaml'
    config = load_config(config, experiment_name)
    train(experiment_name, **config)


if __name__ == '__main__':
    main()
