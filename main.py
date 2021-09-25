from models.build import build_model
import argparse
from models.config import get_config
import torch
from k_means.k_means import k_means
import numpy as np
# def parse_option():
#     parser = argparse.ArgumentParser('Swin-Transformer as a encoder', add_help=False)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', default='./configs/swin_base_patch4_window7_224.yaml', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    #parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)


    return args, config


def main(configs):
    #data =
    # Create the dataloader
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    image_size = 64
    dataset = dset.ImageFolder(root=r'G:\crops',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, num_workers=0)

    #for epoch in range(configs.TRAIN.START_EPOCH, configs.TRAIN.EPOCHS):
    sum_loss = 0
    for i, data in enumerate(dataloader, 0):
        data_real = data[0].view(64, -1)
        print("iter:{} | data: {}".format(i, data_real.size()))
        sum_loss  = sum_loss + k_means(data_real)

    print("avg_loss loss = {}".format(sum_loss / len(dataloader)))

    sum_loss_random = 0
    for i in range(90):
        data_real = 1.5 * np.random.randn(64, 64*64*3)
        sum_loss_random = sum_loss_random + k_means(data_real)
    print("avg_loss_random loss = {}".format(sum_loss_random/len(dataloader)))



    # model = build_model(config)
    # print('model type: ', type(model))
    # #model.cuda()



if __name__ == '__main__':
    _, args = parse_option()
    #print(args)
    main(args)

