from models.build import build_model
import argparse
from models.config import get_config
import torch
from models.build_model import build_model
from k_means_module.k_means import kmeans
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


def train_one_epoch(config, model_feaExa_style, dataloader, optimizer_feaExa_style, epoch):
    for i, data in enumerate(dataloader, 0):
        real_data = data[0]
        print("epoch:{} | iter: {}".format(epoch, i))
        common_feature = model_feaExa_style(real_data)
        print("commom feature size:{}".format(common_feature.size()))
        resize = common_feature
        print("resize size = {}".format(resize.size()))
        loss_kmeans = kmeans(resize, 4, 1)
        print("loss_kmeans = {}".format(loss_kmeans))
        #loss_kmeans.backward()
        break

def main(config):
    # Create the dataloader
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    image_size = config.DATA.IMG_SIZE
    dataset = dset.ImageFolder(root='data/crops',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                             shuffle=True, num_workers=0)
    #Create the optimizer
    optimizer_feaExa_style = None


    #Create the model
    #feature extraction
    model_feaExa_style = build_model(config, "swin")
    model_feaExa_content = None
    #print("model arguments:\n",format(model_feaExa_style))

    #for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    train_one_epoch(config, model_feaExa_style, dataloader, optimizer_feaExa_style, epoch = 1)



if __name__ == '__main__':
    _, args = parse_option()
    #print(args)
    main(args)

