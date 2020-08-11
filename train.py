#training

import model
import param
from data import *
import utils
import engine

import time
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')

import gc

  
def map_fn(index=None, flags=None):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(1234)
  
    train_data = data.DATA(param.TRAIN_DIR) 

    
    train_sampler = torch.utils.data.RandomSampler(train_data)

    train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=flags['batch_size'] if param.MULTI_CORE else param.BATCH_SIZE,
      sampler=train_sampler,
      num_workers=flags['num_workers'] if param.MULTI_CORE else 4,
      drop_last=True,
      pin_memory=True)

    
    DEVICE = param.DEVICE


    netG = model.colorization_model().double()
    netD = model.discriminator_model().double()

    VGG_modelF = torchvision.models.vgg16(pretrained=True).double()
    VGG_modelF.requires_grad_(False)

    netG = netG.to(DEVICE)
    netD = netD.to(DEVICE)
  
    VGG_modelF = VGG_modelF.to(DEVICE)

    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
  ## Trains
    train_start = time.time()
    losses = {
      'G_losses' : [],
      'D_losses' : [],
      'EPOCH_G_losses' : [],
      'EPOCH_D_losses' : [],
      'G_losses_eval' : []
    }

    netG, optG, netD, optD, epoch_checkpoint = utils.load_checkpoint(param.CHECKPOINT_DIR, netG, optG, netD, optD, DEVICE)
    netmodel = model.GAN(netG, netD)
    for epoch in range(epoch_checkpoint,flags['num_epochs']+1 if param.MULTI_CORE else param.NUM_EPOCHS+1):
        print('\n')
        print('#'*8,f'EPOCH-{epoch}','#'*8)
        losses['EPOCH_G_losses'] = []
        losses['EPOCH_D_losses'] = []
    
    engine.train(train_loader, netG, netD, VGG_modelF, optG, optD, device=DEVICE, losses=losses)
    #########################CHECKPOINTING#################################
    utils.create_checkpoint(epoch, netG, optG, netD, optD, max_checkpoint=param.KEEP_CKPT, save_path = param.CHECKPOINT_DIR)
    ########################################################################
    utils.plot_some(train_data, netG, DEVICE, epoch)
    gc.collect()
# configures training (and evaluation) parameters

def run():
    map_fn()
    # print(flags)
    if __name__=='__main__':
        run()