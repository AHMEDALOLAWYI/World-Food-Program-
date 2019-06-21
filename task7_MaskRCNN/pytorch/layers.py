from fastai.vision import *
from fastai.vision.models.unet import _get_sfs_idxs, model_sizes, hook_outputs

class LateralUpsampleMerge(nn.Module):
    "Merge the features coming from the downsample path (in `hook`) with the upsample path."
    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.Px_conv1 = conv2d(ch_lat, ch, ks=1, bias=True)
            
    def forward(self, x):
        #Run a 1x1conv on the features from the downsampling path, upsample the output from P(x-1)
        res = self.Px_conv1(self.hook.stored) + F.upsample(x, scale_factor=2)
        return res

class FPN(nn.Module):
    """
    Creates upsampling path by hooking activations from the downsampling path. Tested on ResNet50.
    
    encoder: Default ResNet50
    chs: Number of intermediate channels to use between convolutions
    """
    def __init__(self, encoder:nn.Module, chs:int):
    
        super().__init__()
        
        #This runs dummy data through the encoder to get the right channel numbers after each layer C1 through C5
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        
        #Attaching hooks to the relevant layers C2 to C5 so we can get their activations during the
        #upsampling path
        self.encoder = encoder
        
        #The link between C5 and P5
        #TODO: will a stride 2 conv be better?
        self.c5_p5 = nn.Sequential(
            conv2d(sfs_szs[-1][1], chs, ks=1, bias=True),
        )
        
        
        self.p5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            conv2d(chs,chs,ks=3,padding=0,bias=True)
        )
        
        #Link between P5 and P6
        self.p5_p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        
        #These are the idxs of C4, C3, and C2 respectively
        idx  = list(reversed(sfs_idxs[-2:-5:-1]))
        self.sfs = hook_outputs([encoder[i] for i in idx])
        
        #This handles the mapping from P5 -> P4 -> P3 -> P2
        self.merges = nn.ModuleList([LateralUpsampleMerge(chs, sfs_szs[idx][1], hook) 
                                     for idx,hook in zip(idx, self.sfs)])
        
        #One final conv to smoothen things out after the merge
        self.final_convs = [
            nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
           conv2d(out_channels, out_channels, ks=3, stride=1, bias=True, padding=0),
            ) for _ in idx+[1]
        ]
           
    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5_p5(c5.clone())]
        #Mapping P5 through P2 one by one
        for merge in self.merges: p_states = [merge(p_states[0])] + p_states
         
        #Extra convs after the lateral upsampling
        for i, conv in enumerate(self.final_convs):
            p_states[i] = conv(p_states[i])
            
        #Doing P6 at the end
        p6 = self.p5_p6(p_states[-1])
        p_states += [p6]
        return p_states
    
    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()