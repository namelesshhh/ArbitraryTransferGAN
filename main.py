import argparse
from models.config import get_config
import torch
import torch.nn as nn
import torch.optim as optim
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
        #print('-->para:', parms)
        print('-->grad_requirs:', parms.requires_grad)
        print('-->grad_value:', parms.grad)
        print("==="*20)
        break

def train_one_epoch(config,
                    model_feaExa_style, swin_unet, discriminator,FeatureExtractor,                              #model
                    dataloader_style, dataloader_content,                                                       #dataloader
                    optim_feaExa_style, optim_swin_unet, optim_discriminator, optim_FeatureExtractor,       #optimizer
                    loss_MSE, loss_BCE,                                                                         #loss function
                    epoch):                                                                                     #others
    """
    :param config: configurations for training seting
    :param model_feaExa_style: style feature extraction model
    :param swin_unet: content feature extraction and image generator
    :param discriminator:  discriminator for real and fake image
    :param FeatureExtractor: feature extractor from fake image and truth image, it is aiming to chanel wise fusion
    :param dataloader_style:  a dataloader for style
    :param dataloader_content:  a dataloader for content
    :param optimizer_feaExa_style:  optimizer for cluster and style feature extraction
    :param optim_swin_unet: optimizer for swin_unet
    :param optim_discriminator: optimizer for discriminator
    :param optim_FeatureExtractor: optimizer for FeatureExtractor
    :param loss_MSE: well
    :param loss_BCE: well
    :param epoch: well
    :return: void
    """
    for i_c , data_c in enumerate(dataloader_content, 0):
        data_c = data_c[0].to(device)
        for i_s, data_s in enumerate(dataloader_style, 0):
            #reset grad
            discriminator.zero_grad()
            model_feaExa_style.zero_grad()

            data_s = data_s[0].to(device)

            #Style feature extract module
            common_feature = model_feaExa_style(data_s) # B L C
            common_feature_size = common_feature.size()
    
            common_feature = torch.flatten(common_feature, start_dim=1) # B L*C
    
            label, Center = kmeans(common_feature, 6, 10)

            loss_classify = loss_MSE(common_feature, Center)
            loss_classify.backward(retain_graph = True)



            #Style feature active fusion module
            common_feature = common_feature.reshape(*common_feature_size)

            #Swin-Unet
            fake_image = swin_unet(data_c, common_feature)

            #################################################################################
            #For Discriminator aim to min(D(fake)), that's mean truth is truth, fake is fake
            #################################################################################
            
            feature_fakeimg = FeatureExtractor(fake_image)
            feature_truthimg = FeatureExtractor(data_s)


            B_fake = []
            for i_f in range(feature_fakeimg.size()[0]):
                B_tmp = []
                B_tmp.append(feature_truthimg[i_f * 2])
                B_tmp.append(feature_fakeimg[i_f])
                B_tmp.append(feature_truthimg[i_f * 2 + 1])
                T_B_f = torch.cat(B_tmp, 0)
                B_fake.append(T_B_f)
            new_fakeimg = torch.stack(B_fake, 0)
            
            B_truth = []
            for i_t in range(feature_truthimg.size()[0] - 2):
                B_tmp = []
                B_tmp.append(feature_truthimg[i_t])
                B_tmp.append(feature_truthimg[i_t + 1])
                B_tmp.append(feature_truthimg[i_t + 2])
                T_B_t = torch.cat(B_tmp, 0)
                B_truth.append(T_B_t)
            new_truthimg = torch.stack(B_truth, 0)

            size_real = new_truthimg.size(0)
            size_fake = new_fakeimg.size(0)
            labels_real = torch.full((size_real,), real_label, dtype=torch.float, device=device)
            labels_fake = torch.full((size_fake,), fake_label, dtype=torch.float, device=device)
            
            
            D_real = discriminator(new_truthimg.detach()).view(-1)
            errD_real = loss_BCE(D_real, labels_real) #real is real
            errD_real.backward()

            D_fake = discriminator(new_fakeimg.detach()).view(-1)
            errD_fake = loss_BCE(D_fake, labels_fake) #fake is fake
            errD_fake.backward()

            optim_discriminator.step()
            ##################################################################
            #But for Generator, aim to max(D(fake)), that's mean fake is truth
            ##################################################################
            FeatureExtractor.zero_grad()
            swin_unet.zero_grad()


            G_fake = discriminator(new_fakeimg).view(-1)
            labels_fake.fill_(real_label)
            errG = loss_BCE(G_fake, labels_fake) #Generator wish fake is real
            errG.backward()

            optim_FeatureExtractor.step()
            optim_swin_unet.step()
            optim_feaExa_style.step()

            print("epoch:{}/{} | iter_content: {}/{} | iter_style: {}/{} | D(real): {} | D(fake): {}".format(epoch, config.TRAIN.EPOCHS, i_c, len(dataloader_content), i_s, len(dataloader_style), D_real.mean().item(), D_fake.mean().item()))

            print("loss_classify = {} | errD_real = {} | errD_fake = {} | errG = {}".format(loss_classify, errD_real, errD_fake, errG))


            break
        break

def main(config):
    # Create the dataloader
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    image_size = config.DATA.IMG_SIZE
    index_data = 6
    rootPath = '../TrainingData/crops' + str(index_data)
    dataset_style = dset.ImageFolder(root=rootPath,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataset_content = dset.ImageFolder(root='../TrainingData/WordImage',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader_style = torch.utils.data.DataLoader(dataset_style, batch_size=16,
                                             shuffle=True, num_workers=0)
    dataloader_content = torch.utils.data.DataLoader(dataset_content, batch_size=8,
                                             shuffle=True, num_workers=0)

    #Create the Loss function
    loss_BCE = nn.BCELoss()
    loss_MSE = nn.MSELoss()


    #Create the model
    #feature extraction
    model_feaExa_style = build_model(config, "swin").to(device)
    swin_unet = build_model(config, "swin_unet").to(device)
    #print("model arguments:\n",format(model_feaExa_style))

    #Discriminator
    discriminator = build_model(config, "discriminator").to(device)

    #FeatureExtractor
    FeatureExtractor = build_model(config, "FeatureExtractor").to(device)

    #Load Model param
    model_feaExa_style.load_state_dict(torch.load('./param/model_feaExa_style.pkl'))
    swin_unet.load_state_dict(torch.load('./param/swin_unet.pkl'))
    discriminator.load_state_dict(torch.load('./param/discriminator.pkl'))
    FeatureExtractor.load_state_dict(torch.load('./param/FeatureExtractor.pkl'))

    #Create the optimizer
    #Learning rate for optimizers
    lr = 0.0002
    optim_feaExa_style = optim.Adam(model_feaExa_style.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_swin_unet = optim.Adam(swin_unet.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_FeatureExtractor = optim.Adam(FeatureExtractor.parameters(), lr=lr, betas=(0.5, 0.999))


    #for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    train_one_epoch(config,
                    model_feaExa_style, swin_unet, discriminator, FeatureExtractor,                           #model
                    dataloader_style, dataloader_content,                                                     #dataloader
                    optim_feaExa_style,optim_swin_unet,optim_discriminator,optim_FeatureExtractor,            #optimizer
                    loss_MSE,  loss_BCE,                                                                      #loss function
                    epoch = 1                                                                                 #others
                    )

    #Save Model
    torch.save(model_feaExa_style.state_dict(), './param/model_feaExa_style.pkl')
    torch.save(swin_unet.state_dict(), './param/swin_unet.pkl')
    torch.save(discriminator.state_dict(), './param/discriminator.pkl')
    torch.save(FeatureExtractor.state_dict(), './param/FeatureExtractor.pkl')

if __name__ == '__main__':
    _, args = parse_option()
    #print(args)
    main(args)

