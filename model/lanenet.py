import torch
import torch.nn as nn
from model import decoder_fcn
from model.resnet import resnet18, resnet34

class LaneNet_FCN_Res_1E1D(nn.Module):
    """ 
    A LaneNet model made up of FCN, whose backbone is ResNet.
    1E1D means one encoder and one decoder, that is the embedding and the binary segmentation 
    share the same encoder and decoder, except the last layer of the decoder.
    """

    def __init__(self):
        super().__init__()

        # comment or uncomment to choose from different encoders and decoders
        self.encoder = resnet18(pretrained=True)
        # self.encoder = resnet34(pretrained=True)
 
        self.decoder = decoder_fcn.Decoder_LaneNet_TConv()  # Decoder with Transposed Conv
        # self.decoder = decoder_fcn.Decoder_LaneNet_Interplt()  # Decoder with Interpolation

    def forward(self, input):
        x = self.encoder.forward(input)
        
        # store feature maps of the encoder for later fusion in the decoder
        input_tensor_list = [self.encoder.c1, self.encoder.c2, self.encoder.c3, self.encoder.c4, x]
        embedding, logit = self.decoder.forward(input_tensor_list)

        return embedding, logit


class LaneNet_FCN_Res_1E2D(nn.Module):
    ''' 
    A LaneNet model made up of FCN, whose backbone is ResNet.
    1E2D means one encoder and two decoder, that is the embedding and the binary segmentation 
    share the same encoder, and each owns a independant decoder.
    '''
    def __init__(self):
        super().__init__()

        self.encoder = resnet18(pretrained=True)
        # self.encoder = resnet34(pretrained=True)
        self.decoder_embed = decoder_fcn.Decoder_LaneNet_TConv_Embed()
        self.decoder_logit = decoder_fcn.Decoder_LaneNet_TConv_Logit()

    def forward(self, input):
        x = self.encoder.forward(input)

        input_tensor_list = [self.encoder.c1, self.encoder.c2, self.encoder.c3, self.encoder.c4, x]
        embedding= self.decoder_embed.forward(input_tensor_list)
        logit = self.decoder_logit.forward(input_tensor_list)

        return embedding, logit



if __name__ == '__main__':
    img = torch.randn(1, 3, 512, 512)
    model = LaneNet_FCN_Res_1E2D()
    embedding, logit = model(img)

    print(embedding.shape)
