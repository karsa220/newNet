from seed.set_seed import set_seed
import torch
import torch.nn as nn
from .segformer import *
from typing import Tuple
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x.clone())

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x


class SegU_decoder(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

        

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class BridgeLayer_4(nn.Module):
    """4-scale bridge layer. Token ratio n1:n2:n3:n4 = 16:8:5:2 (MiT backbone, C=64)."""

    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)

    @staticmethod
    def _parse_sizes_from_total(N):
        """MiT backbone token ratio 16:8:5:2, sum=31."""
        u = N // 31
        n1 = 16 * u
        n2 = 8 * u
        n3 = 5 * u
        n4 = 2 * u
        H1 = W1 = int(n1 ** 0.5)      # = H/4
        H2 = W2 = H1 // 2             # = H/8
        H3 = W3 = H2 // 2             # = H/16
        H4 = W4 = H3 // 2             # = H/32
        return (n1, n2, n3, n4), (H1, W1, H2, W2, H3, W3, H4, W4)

    def forward(self, inputs):
        C = 64
        if type(inputs) == list:
            c1, c2, c3, c4 = inputs
            B, C, H1, W1 = c1.shape
            _, _, H2, W2 = c2.shape
            _, _, H3, W3 = c3.shape
            _, _, H4, W4 = c4.shape

            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)

            n1, n2, n3, n4 = c1f.shape[1], c2f.shape[1], c3f.shape[1], c4f.shape[1]
            inputs = torch.cat([c1f, c2f, c3f, c4f], dim=1)
        else:
            B, N, C = inputs.shape
            (n1, n2, n3, n4), (H1, W1, H2, W2, H3, W3, H4, W4) = self._parse_sizes_from_total(N)

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :n1, :].reshape(B, -1, C)
        tem2 = tx[:, n1:n1 + n2, :].reshape(B, -1, C * 2)
        tem3 = tx[:, n1 + n2:n1 + n2 + n3, :].reshape(B, -1, C * 5)
        tem4 = tx[:, n1 + n2 + n3:n1 + n2 + n3 + n4, :].reshape(B, -1, C * 8)

        m1f = self.mixffn1(tem1, H1, W1).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, H2, W2).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, H3, W3).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, H4, W4).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], dim=1)
        tx2 = tx1 + t1

        return tx2


class BridgeLayer_3(nn.Module):
    """3-scale bridge layer. Token ratio n2:n3:n4 = 8:5:2 (MiT backbone, C=64)."""

    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)

    @staticmethod
    def _parse_sizes_from_total(N):
        """MiT backbone token ratio 8:5:2, sum=15."""
        u = N // 15
        n2 = 8 * u
        n3 = 5 * u
        n4 = 2 * u
        # n2 = (H/8)^2 * (C2/C) = (H/8)^2 * 2  →  H/8 = sqrt(n2/2)
        H2 = W2 = int((n2 // 2) ** 0.5)
        H3 = W3 = H2 // 2                # = H/16
        H4 = W4 = H3 // 2                # = H/32
        return (n2, n3, n4), (H2, W2, H3, W3, H4, W4)

    def forward(self, inputs):
        C = 64
        if type(inputs) == list:
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            _, _, H2, W2 = c2.shape
            _, _, H3, W3 = c3.shape
            _, _, H4, W4 = c4.shape

            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)

            n2, n3, n4 = c2f.shape[1], c3f.shape[1], c4f.shape[1]
            inputs = torch.cat([c2f, c3f, c4f], dim=1)
        else:
            B, N, C = inputs.shape
            (n2, n3, n4), (H2, W2, H3, W3, H4, W4) = self._parse_sizes_from_total(N)

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem2 = tx[:, :n2, :].reshape(B, -1, C * 2)
        tem3 = tx[:, n2:n2 + n3, :].reshape(B, -1, C * 5)
        tem4 = tx[:, n2 + n3:n2 + n3 + n4, :].reshape(B, -1, C * 8)

        m2f = self.mixffn2(tem2, H2, W2).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, H3, W3).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, H4, W4).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f], dim=1)
        tx2 = tx1 + t1

        return tx2



class BridegeBlock_4(nn.Module):
    """4-scale bridge block. Output token ratio 16:8:5:2."""

    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x):
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, N, C = bridge4.shape
        u = N // 31                     # ratio 16:8:5:2, sum=31
        n1 = 16 * u
        n2 = 8 * u
        n3 = 5 * u
        n4 = 2 * u
        H1 = W1 = int(n1 ** 0.5)
        H2 = W2 = H1 // 2
        H3 = W3 = H2 // 2
        H4 = W4 = H3 // 2

        outs = []
        sk1 = bridge4[:, :n1, :].reshape(B, H1, W1, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, n1:n1 + n2, :].reshape(B, H2, W2, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, n1 + n2:n1 + n2 + n3, :].reshape(B, H3, W3, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, n1 + n2 + n3:n1 + n2 + n3 + n4, :].reshape(B, H4, W4, C * 8).permute(0, 3, 1, 2)
        outs.extend([sk1, sk2, sk3, sk4])

        return outs


class BridegeBlock_3(nn.Module):
    """3-scale bridge block. Output token ratio 8:5:2."""

    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_3(dims, head, reduction_ratios)

    def forward(self, x):
        outs = []
        if type(x) == list:
            outs.append(x[0])

        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, N, C = bridge4.shape
        u = N // 15                     # ratio 8:5:2, sum=15
        n2 = 8 * u
        n3 = 5 * u
        n4 = 2 * u
        H2 = W2 = int((n2 // 2) ** 0.5)   # n2 = (H/8)^2 * 2  →  H/8 = sqrt(n2/2)
        H3 = W3 = H2 // 2
        H4 = W4 = H3 // 2

        sk2 = bridge4[:, :n2, :].reshape(B, H2, W2, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, n2:n2 + n3, :].reshape(B, H3, W3, C * 5).permute(0, 3, 1, 2)
        sk4 = bridge4[:, n2 + n3:n2 + n3 + n4, :].reshape(B, H4, W4, C * 8).permute(0, 3, 1, 2)
        outs.extend([sk2, sk3, sk4])

        return outs


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios,token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

class MISSFormer(nn.Module):
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip", encoder_pretrained=True, img_size=224):
        super().__init__()

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        self.img_size = img_size
        d_base_feat_size = img_size // 32
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]

        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        self.backbone = MiT(img_size, dims, layers, token_mlp_mode)

        self.reduction_ratios = [1, 2, 4, 8]
        self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3], reduction_ratios[3], token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], heads[2], reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1], reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0], reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)

    def _update_decoder_sizes(self, H4, W4):
        """根据实际 bridge 输出尺寸更新 decoder 的 input_resolution，适配任意分辨率。"""
        self.decoder_3.layer_up.input_resolution = (H4, W4)
        self.decoder_2.layer_up.input_resolution = (H4 * 2, W4 * 2)
        self.decoder_1.layer_up.input_resolution = (H4 * 4, W4 * 4)
        self.decoder_0.layer_up.input_resolution = (H4 * 8, W4 * 8)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder = self.backbone(x)
        bridge = self.bridge(encoder)  # list

        b, c, H4, W4 = bridge[3].shape
        self._update_decoder_sizes(H4, W4)

        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(bridge[3].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 1))

        return tmp_0

         
