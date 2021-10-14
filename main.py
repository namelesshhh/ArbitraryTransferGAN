import argparse
from models.config import get_config
import torch
import torch.nn as nn
from models.build_model import build_model
from k_means_module.k_means import kmeans
from models.active_feature_fusion import active_feature_fusion


#param
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
real_label = 1
fake_label = 0


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

def param_visual(model):
    """
    :param model: the model that you want to visual it paramters
    :return: void
    """
    for name, parms in model.named_parameters():
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:', parms.requires_grad)
        print('-->grad_value:', parms.grad)
        print("==="*10)


def train_one_epoch(config, model_feaExa_style, swin_unet, discriminator,   #model
                    dataloader_style, dataloader_content,                   #dataloader
                    optimizer_feaExa_style,                                 #optimizer
                    loss_MSE, loss_BCE,                                     #loss function
                    epoch):                                                 #others
    """

    :param config: configurations for training seting
    :param model_feaExa_style: style feature extraction model
    :param swin_unet: content feature extraction and image generator
    :param discriminator:  discriminator for real and fake image
    :param dataloader_style:  a dataloader for style
    :param dataloader_content:  a dataloader for content
    :param optimizer_feaExa_style:  optimizer for cluster and style feature extraction
    :param loss_MSE: well
    :param loss_BCE: well
    :param epoch: well
    :return: void
    """
    for i_c , data_s in enumerate(dataloader_content, 0):
        data_c = data_c[0]
        for i_s, data_s in enumerate(dataloader_style, 0):
            data_s = data_s[0].to(device)

            #Style feature extract module
            common_feature = model_feaExa_style(data_s) # B L C
            common_feature_size = common_feature.size()
            print("Before flatten common feature size:{}".format(common_feature.size()))
            common_feature = torch.flatten(common_feature, start_dim=1) # B L*C
            print("After flatten common feature size:{}".format(common_feature.size()))
            label, Center = kmeans(common_feature, 6, 10)

            loss_classify = loss_MSE(common_feature, Center)
            loss_classify.backward()
            print("loss_classify = {}".format(loss_classify))

            #Style feature active fusion module
            common_feature = common_feature.reshape(*common_feature_size)
            content_feature = torch.randn([1, 49,1024] , requires_grad=True)
            fusion_feature = active_feature_fusion(content_feature, common_feature, common_feature_size[-1], 8)
            print("fusion_feature size = {}".format(fusion_feature.size()))

            #Swin-Unet
            fake_image = swin_unet(data_c, data_s)


            #Discriminator
            size_real = data_s.size(0)
            size_fake = fake_image.size(0)
            label_real = torch.full((size_real,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((size_fake,), fake_label, dtype=torch.float, device=device)

            D_fake = discriminator(fake_image) #B * 1
            D_real = discriminator(data_s)

            errD_real = loss_BCE(data_s, real_label)
            errD_fake = loss_BCE(fake_image, fake_label)

            print("epoch:{}/{} | iter_content: {}/{} | iter_style: {}/{} | D(real): {} | D(fake): {}".format(epoch, config.TRAIN.EPOCHS, i_c, len(dataloader_content)
                                                                                 , i_s, len(dataloader_style), D_real, D_fake))


            break

def main(config):
    # Create the dataloader
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    image_size = config.DATA.IMG_SIZE
    dataset_style = dset.ImageFolder(root='data/crops',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataset_content = dset.ImageFolder(root='data/WordImage',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader_style = torch.utils.data.DataLoader(dataset_style, batch_size=8,
                                             shuffle=True, num_workers=0)
    dataloader_content = torch.utils.data.DataLoader(dataset_content, batch_size=8,
                                             shuffle=True, num_workers=0)
    #Create the optimizer
    optimizer_feaExa_style = None

    #Create the Loss function
    loss_BCE = nn.BCELoss()
    loss_MSE = nn.MSELoss()


    #Create the model
    #feature extraction
    model_feaExa_style = build_model(config, "swin")
    model_feaExa_content = build_model(config, "swin_unet")
    #print("model arguments:\n",format(model_feaExa_style))

    #Discriminator
    discriminator = build_model(config, "dis")

    #for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    train_one_epoch(config,
                    model_feaExa_style, model_feaExa_content,       #model
                    dataloader_style, dataloader_content,           #dataloader
                    optimizer_feaExa_style,                         #optimizer
                    loss_MSE,  loss_BCE,                            #loss function
                    epoch = 1                                       #others
                    )

if __name__ == '__main__':
    _, args = parse_option()
    #print(args)
    main(args)

