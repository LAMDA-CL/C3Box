import copy
import logging
import torch
from sympy import false
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear,SimpleContinualLinear,EaseCosineLinear, TunaLinear
import timm
import torch.nn.functional as F
from backbone.projections import Proj_Pure_MLP, MultiHeadAttention

from utils.toolkit import get_attribute
import difflib
from PIL import Image
import random
random.seed(1993)
import types

def get_convnet(args, pretrained=False):

    backbone_name = args["backbone_type"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


# simplecil
class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        # for RanPAC
        self.W_rand = None
        self.RP_dim = None

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                nn.init.constant_(fc.sigma, 1.0)
            # fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)

    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, x):
        x = self.convnet.encode_image(x)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = self.fc(x)
        return out

#rapf/zs-clip
class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name = 'SimpleClipNet'
        self.args = args
        self._device = args["device"][0]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # fc.sigma.data = self.fc.sigma.data
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                nn.init.constant_(fc.sigma, 1.0)
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)

    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):
        image_features, text_features, logit_scale = self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)

#proof
class Proof_Net(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.args = args
        self._device = args["device"][0]
        self.projtype = get_attribute(self.args, 'projection_type', 'mlp')
        self.context_prompt_length_per_task = get_attribute(self.args, 'context_prompt_length_per_task', 3)

        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        self.img_prototypes = None

        self.context_prompts = nn.ParameterList()

    def update_prototype(self, nb_classes):
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat([copy.deepcopy(self.img_prototypes).to(self._device),
                                             torch.zeros(nb_classes - nb_output, self.feature_dim).to(
                                                 self._device)]).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(self._device)
        print('update prototype, now we have {} prototypes'.format(self.img_prototypes.shape[0]))

    def update_context_prompt(self):
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(
            nn.Parameter(torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(self._device)))
        print('update context prompt, now we have {} context prompts'.format(
            len(self.context_prompts) * self.context_prompt_length_per_task))
        self.context_prompts.to(self._device)

    def get_context_prompts(self):
        return torch.cat([item for item in self.context_prompts], dim=0)

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        img_features = [proj(basic_img_features) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)  # [bs,num_proj,dim]
        image_feas = torch.sum(img_features, dim=1)  # [bs,dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        text_features = [proj(basic_text_features) for proj in self.projs_text]
        text_features = torch.stack(text_features, dim=1)
        text_feas = torch.sum(text_features, dim=1)  # [bs,dim]
        return F.normalize(text_feas, dim=-1) if normalize else text_feas

    def encode_prototpyes(self, normalize: bool = False):
        self.img_prototypes = self.img_prototypes.to(self._device)
        img_features = [proj(self.img_prototypes) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)  # [nb_class,num_proj,dim]
        image_feas = torch.sum(img_features, dim=1)  # [nb_class,dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

    def extend_task(self):
        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())

    def extend_item(self):
        if self.projtype == 'pure_mlp':
            return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self._device)
        else:
            raise NotImplementedError

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)  # bs,dim
        text_features = self.encode_text(text, normalize=True)  # bs,dim

        prototype_features = self.encode_prototpyes(normalize=True)  # nb_class,dim
        context_prompts = self.get_context_prompts()  # num_prompt, dim

        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        # restack the features and pass them through the attention layer
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)  # bs,1,dim
        text_features = text_features.view(text_features.shape[0], self.feature_dim)  # num_text,dim
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)  # len_proto,dim
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)  # len_con,dim
        # expand text features to be the same dim as image features
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0],
                                             self.feature_dim)  # bs,num_text,dim
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0],
                                                       self.feature_dim)  # bs,len_proto,dim
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0],
                                                 self.feature_dim)  # bs,len_con,dim
        # concat them together
        # features = torch.cat([image_features, text_features, prototype_features], dim=1) # bsize * (1+num_texts+num_protos) * dim
        features = torch.cat([image_features, text_features, prototype_features, context_prompts],
                             dim=1)  # bsize * (1+num_texts+num_protos+num_context) * dim
        features = self.sel_attn(features, features, features)
        # split them back, image features are the first half, text features are the second half
        # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
        image_features = features[:, 0, :]  # bsize * dim
        text_features = features[:, 1:len_texts + 1, :]  # bsize * num_texts * dim
        prototype_features = features[:, len_texts + 1:len_texts + 1 + len_protos, :]  # bsize * num_protos * dim
        context_prompts = features[
            :, len_texts + 1 + len_protos:len_texts + 1 + len_protos + len_context_prompts, :]  # bsize * num_context * dim
        text_features = torch.mean(text_features, dim=0)  # num_texts * dim
        prototype_features = torch.mean(prototype_features, dim=0)  # num_protos * dim
        # squeeze
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features

    def forward_transformer(self, image_features, text_features, transformer=False):
        prototype_features = self.encode_prototpyes(normalize=True)
        if transformer:
            context_prompts = self.get_context_prompts()
            len_texts = text_features.shape[0]
            len_protos = prototype_features.shape[0]
            len_context_prompts = context_prompts.shape[0]
            # restack the features and pass them through the attention layer
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)  # [bs, 1, dim]
            text_features = text_features.view(text_features.shape[0], self.feature_dim)  # [total_classes, dim]
            prototype_features = prototype_features.view(prototype_features.shape[0],
                                                         self.feature_dim)  # [len_pro, dim]
            context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)  # [len_con_pro, dim]
            # expand text features to be the same dim as image features
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0],
                                                 self.feature_dim)  # [bs, total_classes, dim]
            prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0],
                                                           self.feature_dim)  # [bs, len_pro, dim]
            context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0],
                                                     self.feature_dim)  # [bs, len_con_pro, dim]
            # concat them together
            # features = torch.cat([image_features, text_features, prototype_features], dim=1) # bsize * (1+num_texts+num_protos) * dim
            features = torch.cat([image_features, text_features, prototype_features, context_prompts],
                                 dim=1)  # bsize * (1+num_texts+num_protos+num_context) * dim
            # pass through the attention layer
            features = self.sel_attn(features, features, features)
            # split them back, image features are the first half, text features are the second half
            # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
            image_features = features[:, 0, :]  # bsize * dim
            text_features = features[:, 1:len_texts + 1, :]  # bsize * num_texts * dim
            prototype_features = features[:, len_texts + 1:len_texts + 1 + len_protos, :]  # bsize * num_protos * dim
            context_prompts = features[
                :, len_texts + 1 + len_protos:len_texts + 1 + len_protos + len_context_prompts, :]  # bsize * num_context * dim
            # remove the 0-th dimension of text features to be num_texts * dim
            text_features = torch.mean(text_features, dim=0)  # num_texts * dim
            prototype_features = torch.mean(prototype_features, dim=0)  # num_protos * dim
            # squeeze
            image_features = image_features.view(image_features.shape[0], -1)
            text_features = text_features.view(text_features.shape[0], -1)
            prototype_features = prototype_features.view(prototype_features.shape[0], -1)
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
        else:
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features

    def freeze_projection_weight_new(self):
        if len(self.projs_img) > 1:
            for i in range(len(self.projs_img)):
                for param in self.projs_img[i].parameters():
                    param.requires_grad = False
                for param in self.projs_text[i].parameters():
                    param.requires_grad = True
            for param in self.projs_img[-1].parameters():
                param.requires_grad = True
        for param in self.sel_attn.parameters():
            param.requires_grad = True


#engine
class Engine(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)
        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
        self.args = args
        self.freeze(self.model)
        self.Image_Adapter = nn.ModuleList()
        self.Text_Adapter = nn.ModuleList()
        self.beta = 1
        self.decay = 1

        self.class_mean_list = []
        self.class_cov_list = []
        self.class_edge_distance = []

    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            # vecs_512 = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features = self.visual_forward_(inputs)
                # image_features_512 = image_features @ self.visual_proj
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                vecs.append(image_features)
                # vecs_512.append(image_features_512)
                labels.append(targets)

            vecs = torch.cat(vecs)
            # vecs_512 = torch.cat(vecs_512)
            labels = torch.cat(labels)

            mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)],
                           dim=0)
            center_vecs = torch.cat(
                [vecs[labels == i] - mu[i - known_classes] for i in range(known_classes, total_classes)], dim=0)
            cov_inv = center_vecs.T @ center_vecs / (center_vecs.shape[0] - 1)
            cov_inv = center_vecs.shape[1] * torch.linalg.pinv(
                (center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(
                    center_vecs.shape[1]).cuda())
            if not hasattr(self, 'mu'):
                self.mu = mu
                self.cov_inv = cov_inv
            else:
                self.cov_inv = (known_classes / total_classes) * self.cov_inv + (
                            total_classes - known_classes) / total_classes * cov_inv + (
                                           (known_classes / total_classes) * (
                                               total_classes - known_classes) / total_classes ** 2) * (
                                           self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)) @ (
                                           self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)).T
                self.mu = torch.cat([self.mu, mu])
            ps = torch.ones(self.mu.shape[0]).cuda() * 1. / self.mu.shape[0]
            self.W = torch.einsum('nd, dc -> cn', self.mu, self.cov_inv)
            self.b = ps.log() - torch.einsum('nd, dc, nc -> n', self.mu, self.cov_inv, self.mu) / 2

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)

        if self.visual.attn_pool is not None:
            if self.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.visual.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.visual.attn_pool(x)
                if self.visual.attn_pool_type == 'parallel':
                    pooled = self.visual.attn_pool_contrastive(x)
                else:
                    assert self.visual.attn_pool_type == 'cascade'
                    pooled = self.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.visual.attn_pool(x)
                x = self.visual.ln_post(x)
                pooled, tokens = self.visual._global_pool(x)
        elif self.visual.final_ln_after_pool:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)
        else:
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        return pooled

    def update_task(self):
        from backbone.linears import Adapter, MLP_Adapter
        if len(self.Image_Adapter) > 0:
            self.freeze(self.Image_Adapter[-1])
            self.freeze(self.Text_Adapter[-1])
        self.Image_Adapter.append(MLP_Adapter(512, 512))
        self.Text_Adapter.append(MLP_Adapter(512, 512))

    def Image_encode(self, image_features):
        image_res = []
        for i in range(len(self.Image_Adapter)):
            image_res.append(self.Image_Adapter[i](image_features))
        image_res = torch.sum(torch.stack(image_res), dim=0)
        return image_res

    def Text_encode(self, text_features):
        text_res = []
        for i in range(len(self.Text_Adapter)):
            text_res.append(self.Text_Adapter[i](text_features))
        text_res = torch.sum(torch.stack(text_res), dim=0)
        return text_res

    @property
    def feature_dim(self):
        return self.model.out_dim

    def extract_vector(self, x):
        return self.model.encode_image(x)

    def encode_image(self, x):
        imag_features = self.model.encode_image(x)
        imag_res = self.Image_encode(imag_features)
        return imag_res

    def encode_text(self, x):
        text_features = self.model.encode_text(x)
        text_res = self.Text_encode(text_features)
        return text_res

    def forward(self, img, text):
        image_features, text_features, logit_scale = self.model(img, text)
        return image_features, text_features, logit_scale

    def rerank(self, des_dict, outputs, image_features_raw, class_to_label, device, topk=5):
        with torch.no_grad():
            top5_predict = outputs.topk(topk, 1, True, True)[1]
            top5_predict_labels = [[class_to_label[int(label)] for label in pred] for pred in top5_predict]
            logi = 0
            for _ in range(3):
                texts = []
                for batch in range(image_features_raw.shape[0]):
                    for main_label in top5_predict_labels[batch]:
                        for second_label in top5_predict_labels[batch]:
                            if main_label == second_label:
                                continue
                            texts.append(
                                main_label + ' with ' + random.choice(des_dict[main_label][second_label]).lower())
                texts = self.tokenizer(texts).to(device)
                texts = self.model.encode_text(texts)
                texts = texts.reshape(image_features_raw.shape[0], topk, topk - 1, -1)
                texts = torch.mean(texts, dim=2)
                texts = texts / texts.norm(dim=-1, keepdim=True)
                logits = [image_features_raw[i] @ texts[i].T for i in range(image_features_raw.shape[0])]
                logits = torch.stack(logits)
                logi += logits
            logits = logi / 3
            new_logits = torch.zeros_like(outputs)
            for i in range(image_features_raw.shape[0]):
                new_logits[i, top5_predict[i]] = logits[i]
            return new_logits

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def activate_old_adapter(self):
        for item in self.Image_Adapter:
            for param in item.parameters():
                param.requires_grad = True

        for item in self.Text_Adapter:
            for param in item.parameters():
                param.requires_grad = True

#coda-prompt
class CodaPromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        import open_clip
        basic_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
        basic_model.load_state_dict(torch.load('./c.pth'))
        state_dict = basic_model.state_dict()
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict["text_projection"].shape[1]

        from backbone.codaprompt_model import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        s = basic_model.visual.state_dict()
        d = dict()
        for k, v in s.items():
            if k.endswith("in_proj_weight"):
                c = k.replace("in_proj_weight", "in_proj.weight")
                d[c] = v
            elif k.endswith("in_proj_bias"):
                c = k.replace("in_proj_bias", "in_proj.bias")
                d[c] = v
            else:
                d[k] = v

        self.backbone.load_state_dict(d, strict=False)
        #  self.backbone = get_backbone(args, pretrained)
        self.fc = nn.Linear(512, args["class_num"])
        from backbone.prompt import CodaPrompt
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:, 0, :]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.backbone(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

    @property
    def feature_dim(self):
        return 512

    def extract_vector(self, x):
        q, _ = self.backbone(x)
        q = q[:, 0, :]
        return q

#l2p
class PromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        import open_clip
        basic_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')

      #  basic_model.load_state_dict(torch.load('./c.pth'))
        state_dict = basic_model.state_dict()
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict["text_projection"].shape[1]

        from backbone.l2p_model import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            num_classes=args['nb_classes']
        )
        s = basic_model.visual.state_dict()
        d = dict()
        for k, v in s.items():
            if k.endswith("in_proj_weight"):
                c = k.replace("in_proj_weight", "in_proj.weight")
                d[c] = v
            elif k.endswith("in_proj_bias"):
                c = k.replace("in_proj_bias", "in_proj.bias")
                d[c] = v
            else:
                d[k] = v
        self.backbone.load_state_dict(d, strict=False)
        self.backbone.resize_pos_embed(self.backbone.positional_embedding.unsqueeze(0),self.backbone.new_positional_embedding, 1, (14, 14))
        self.original_backbone = basic_model.visual

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x

    @property
    def feature_dim(self):
        return 512

    def extract_vector(self, x):
        x = self.original_backbone(x)
        return x

#dual
class DualpromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        import open_clip
        basic_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')

        basic_model.load_state_dict(torch.load('./c.pth'))
        state_dict = basic_model.state_dict()
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict["text_projection"].shape[1]

        from backbone.dualprompt_model import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            num_classes=args['nb_classes']
        )
        s = basic_model.visual.state_dict()
        d = dict()
        for k, v in s.items():
            if k.endswith("in_proj_weight"):
                c = k.replace("in_proj_weight", "in_proj.weight")
                d[c] = v
            elif k.endswith("in_proj_bias"):
                c = k.replace("in_proj_bias", "in_proj.bias")
                d[c] = v
            else:
                d[k] = v
        self.backbone.load_state_dict(d, strict=False)

        self.original_backbone = basic_model.visual

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x

    @property
    def feature_dim(self):
        return 512

    def extract_vector(self, x):
        x = self.original_backbone(x)
        return x


def getbackbone():
    import open_clip
    basic_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
    basic_model.load_state_dict(torch.load('./c.pth'))
    state_dict = basic_model.state_dict()
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]

    from backbone.memo_vit import Generalized_Vit, Specialized_Vit
    TaskAgnosticExtractor = Generalized_Vit(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_width // 64,
        output_dim=embed_dim
    )
    TaskAgnosticExtractor.train()
    AdaptiveExtractors = Specialized_Vit(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_width // 64,
        output_dim=embed_dim
    )
    s = basic_model.visual.state_dict()

    TaskAgnosticExtractor.load_state_dict(s, strict=False)

    e = dict()
    for k, v in s.items():
        if k.startswith('transformer.resblocks.11'):
            c = k.replace('transformer.resblocks.11','transformer.resblocks.0')
            e[c] = v
    e['proj']=s['proj']
    e['ln_post.weight']=s['ln_post.weight']
    e['ln_post.bias']=s['ln_post.bias']
    AdaptiveExtractors.load_state_dict(e, strict=False)
    return TaskAgnosticExtractor, AdaptiveExtractors

# memo
class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.TaskAgnosticExtractor, _ = getbackbone()
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out = self.fc(features) #{logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        out.update({"base_features": base_feature_map})
        return out

    def update_fc(self,nb_classes):
        _, _new_extractor = getbackbone()
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())
        if self.out_dim is None:
            self.out_dim=self.AdaptiveExtractors[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma


#foster
class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = 512
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args
        self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):

        features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim:])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        convnet, self.preprocess, self.tokenizer = get_convnet(self.args, self.pretrained)
        self.backbones.append(convnet.visual)
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

#finetune
class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.fea_dim = self.model.out_dim


    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.fea_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.model(x)
        out = self.fc(x[0])
       # out = self.fc(x["features"])
        out["features"] = x[0]
        # out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

#clg_cbm
class PrototypicalNet(BaseNet):
    def __init__(self, args, pretrained, device):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

        # self.backbone.out_dim = 768
        self.feat_dim = 512

        self.freeze_backbone()
        self.device = device
        self.unity = nn.ModuleList()
        self.scale = self.convnet.logit_scale.exp()

        self.explainer = None
        self.relu = torch.nn.ReLU()

    def freeze_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for name, param in self.convnet.named_parameters():
            param.requires_grad = False

    def freeze_module(self, module):
        for name, param in module.named_parameters():
            param.requires_grad = False

    def forward(self, x, bottleneck, pool, sg=None):
        if len(x.shape) > 2:  # for raw image
            x = self.extract_vector(x).float()

        if sg is not None: x = torch.cat((x, sg), dim=0).float()
        x /= x.norm(dim=-1, keepdim=True)
        csv = x @ bottleneck.T

        results = None
        for i, fc in enumerate(self.unity):
            splited_csv = csv[:, i * pool:(i + 1) * pool]

            results = fc(splited_csv) if results is None else torch.concat((results, fc(splited_csv)), dim=1)
        return results, csv

    def forward_clip(self, x, t, sg=None):
        if len(x.shape) > 2: x = self.extract_vector(x).float()
        if sg is not None:
            x = torch.cat((x, sg), dim=0).float()
        t = self.convnet.encode_text(t).float()

        # normalized features
        x = x / x.norm(dim=1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.convnet.logit_scale.exp()
        logits_clip = logit_scale * x @ t.t()

        return logits_clip

    def generate_fc(self, in_dim, out_dim, bias=True):
        fc = nn.Linear(in_dim, out_dim, bias=bias).to(self.device)
        return fc

    def generate_explainer(self, cpt_num, out_dim, bias=True):
        return nn.Linear(cpt_num, out_dim, bias=bias)

    def update_explainer(self, cpt_num, new_cls, bias=True):

        if self.explainer is None:
            self.explainer = self.generate_explainer(self.feat_dim, cpt_num, bias=bias).to(self.device)
            self.unity.append(self.generate_fc(cpt_num, new_cls, bias=bias).to(self.device))

        else:
            total_cpt_num = self.explainer.out_features + cpt_num
            weight = copy.deepcopy(self.explainer.weight.data)
            new_explainer = self.generate_explainer(self.feat_dim, total_cpt_num, bias=bias).to(self.device)
            new_explainer.weight.data[:weight.shape[0]] = weight

            for id, fc in enumerate(self.unity):
                weight = copy.deepcopy(fc.weight.data)
                new_fc = self.generate_fc(total_cpt_num, fc.out_features, bias=bias).to(self.device)
                new_fc.weight.data[:, :weight.shape[1]] = weight
                del fc
                self.unity[id] = new_fc
            self.unity.append(self.generate_fc(total_cpt_num, new_cls, bias=bias).to(self.device))

            del self.explainer
            self.explainer = new_explainer

    def forward_explainer(self, x, sg=None):
        if len(x.shape) > 2: x = self.extract_vector(x).float()
        if sg is not None: x = torch.cat((x, sg), dim=0).float()

        x = x / x.norm(dim=1, keepdim=True)
        csv = self.explainer(x)
        mean = torch.mean(csv, dim=0, keepdim=True)
        std = torch.std(csv, dim=0, keepdim=True)

        norm_csv = csv - mean
        norm_csv /= std

        logits = None
        for fc in self.unity:
            logits = fc(csv) if logits is None else torch.concat((logits, fc(csv)), dim=1)
        return logits, csv

    def forward_fc(self, x):
        if len(x.shape) > 2: x = self.extract_vector(x).float()
        x = x / x.norm(dim=1, keepdim=True)

        logits = None
        for fc in self.unity:
            logits = fc(x) if logits is None else torch.concat((logits, fc(x)), dim=1)
        return logits

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def extract_pre_vector(self, x):
        with torch.no_grad():
            return self.convnet.encode_image_pre_proj(x).float()

    def add_heads(self, fc):
        # add_fc = deepcopy(fc)
        # for name, p in add_fc.named_parameters(): p.requires_grad = False
        self.unity.append(fc)

    def update_fc(self, nb_classes):
        if len(self.unity) == 0:
            self.unity.append(self.generate_fc(self.feat_dim, nb_classes, bias=False).to(self.device))
        else:
            for id, fc in enumerate(self.unity):
                weight = copy.deepcopy(fc.weight.data)
                new_fc = self.generate_fc(self.feat_dim, fc.out_features, bias=False).to(self.device)
                new_fc.weight.data[:, :weight.shape[1]] = weight
                del fc
                self.unity[id] = new_fc
            self.unity.append(self.generate_fc(self.feat_dim, nb_classes, bias=False).to(self.device))

    def forward_text(self, x):
        out = self.convnet.encode_text(x)
        return out


class Gateway(BaseNet):
    # input_dim = attribute_embeddings.shape[-1]
    # output_dim = _total_classes
    def __init__(self, args, device, pretrained=False):
        super().__init__(args, pretrained)
        self.args = args
        self.device = device
        self.convnet = None
        self.gate = None
        # self.heads = None
        self.heads = nn.ModuleList()

    def forward(self, x):
        results = self.gate(x)
        return results

    def update_gateway(self, type, output_dim, input_dim=None, num_attributes=None):
        if self.gate is None:
            self.gate = self.generate_gate(type, input_dim, output_dim, num_attributes)
            self.gate = self.gate.to(self.device)
        else:
            self.gate = self.expand(self.gate, output_dim).to(self.device)

    def expand(self, last, out_dim):
        nb_output = last.out_features
        nb_input = last.in_features

        new = nn.Linear(nb_input, out_dim, bias=True if last.bias is not None else False)
        new.weight.data[:nb_output] = copy.deepcopy(last.weight.data)
        if last.bias is not None:
            new.bias.data[:nb_output] = copy.deepcopy(last.bias.data)
        return new

    def addi(self, out_dim, attributes_embeddings=None):
        model = self.gate
        if self.heads == None:
            self.heads = model.to(self.device)
        else:
            nb_output = self.heads.out_features
            # adding trained_fc
            self.heads = self.expand(self.heads, out_dim)
            # reinit way
            self.heads.weight.data[nb_output:] = copy.deepcopy(model.weight.data)
            if model.bias is not None:
                self.heads.bias.data[nb_output:] = copy.deepcopy(model.bias.data)

            self.heads = self.heads.to(self.device)

    def addi_heads(self):
        self.heads.append(self.gate)

    def generate_gate(self, mode, input_dim, output_dim, num_attributes=None):

        if mode == ['linear', 'bn', 'linear']:
            fc = nn.Sequential(
                nn.Linear(input_dim, num_attributes, bias=False),
                nn.BatchNorm1d(num_attributes),
                nn.Linear(num_attributes, output_dim)
            )  #
        elif mode == ['bn', 'linear']:

            fc = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim, bias=False)
            )
            # if self.mode == "multi": self.heads.append(fc)
        elif mode == ['linear', 'linear']:
            fc = nn.Sequential(
                nn.Linear(input_dim, num_attributes, bias=False),
                nn.Linear(num_attributes, output_dim)
            )
        elif mode == ['linear']:
            fc = nn.Sequential(nn.Linear(num_attributes, output_dim, bias=False))
        else:
            raise NotImplementedError
        return fc


class CLASS_CONCEPT_MATRIX(nn.Module):
    def __init__(self, args, device):
        super(CLASS_CONCEPT_MATRIX, self).__init__()
        self.args = args
        self.gate = None
        self.cls_cpt_matrix = None
        self.device = device
        self.concepts = None
        self.final_matrix = None
        self.heads = nn.ModuleList()

    def update_matrix(self, pool, cls):
        if self.args["mode"] == "multi":
            self.cls_cpt_matrix = self.generate_matrix(pool, cls).to(self.device)
        else:
            if self.cls_cpt_matrix is None:
                self.cls_cpt_matrix = self.generate_matrix(pool, cls).to(self.device)
            else:
                new = self.generate_matrix(pool, cls)
                new.weight.data[:self.cls_cpt_matrix.out_features] = self.cls_cpt_matrix.weight.data
                self.cls_cpt_matrix = new.to(self.device)
                del new

    def update_concept(self, concepts):
        self.concepts = concepts.to(self.device)

    def expandition(self, cls):
        if self.final_matrix is None:
            self.final_matrix = self.cls_cpt_matrix
        else:
            new = nn.Linear(in_features=self.final_matrix.in_features, out_features=cls, bias=False)
            new.weight.data[:self.final_matrix.out_features] = self.final_matrix.weight.data
            new.weight.data[self.final_matrix.out_features:] = self.cls_cpt_matrix.weight.data
            self.final_matrix = new.to(self.device)
            del new
        self.heads.append(self.cls_cpt_matrix)

    def generate_matrix(self, in_dim, out_dim):
        mat = nn.Linear(in_dim, out_dim, bias=False)
        return mat

    def forward(self, x):
        cls_feat = self.cls_cpt_matrix(self.concepts.T)  # not raw concepts but features
        score = x @ cls_feat
        return score

#mg_clip
def forward_clip(self, image, text, return_feature=False):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    if return_feature:
        return logits_per_image, logits_per_text, image_features, text_features

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


class MgclipNet(nn.Module):
    def __init__(self, args, jit=False):
        super().__init__()
        self.cfg = args
        self.device = args["device"][0]
        self.classes_names = None
        self.feature_dim = 512

        # lora_clip
        from backbone.loraclip import lora_clip
        self.model, self.transforms = lora_clip.load(
            args["model_names"],
            device=self.device,
            jit=jit,
            r=args["lora_rank"],
            lora_mode=args["lora_mode"]
        )
        # self.model, self.transforms = lora_clip.load(args.model_names, device = self.device, jit=jit, r=args.lora_rank, lora_mode=args.lora_mode)
        self.model.forward = types.MethodType(forward_clip, self.model)
        ori_state = self.model.state_dict()
        self.text_tokens = None

    def cur_text_features(self):
        f = self.model.encode_text(self.text_tokens)
        f = f / f.norm(dim=1, keepdim=True)
        return f

    def extract_vector(self, x):
        return self.model.encode_image(x)

    def inference(self, image, text_tokens):
        text_features = self.model.encode_text(text_tokens)
        image_features = self.model.visual(image.type(self.model.dtype), all_tokens=False, adapt=self.attention_adapter)
        # pdb.set_trace()

        # image_features = self.attention_adapter(image_features.type(torch.floatPrototypicalNet2))[:, 0, :]

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def forward(self, image, test=False, all_test=False, return_feature=False, replay=None):
        if test:
            # pdb.set_trace()
            with torch.no_grad():
                if all_test:
                    if return_feature:
                        logits_per_image, _, image_features, __ = self.model(image, self.all_text_tokens,
                                                                             return_feature=return_feature)
                    else:
                        logits_per_image, _ = self.model(image, self.all_text_tokens)
                    # logits_per_image = self.inference(image, self.all_text_tokens)
                else:
                    if return_feature:
                        logits_per_image, _, image_features, __ = self.model(image, self.text_tokens,
                                                                             return_feature=return_feature)
                    else:
                        logits_per_image, _ = self.model(image, self.text_tokens)
                # pdb.set_trace()
                probs = logits_per_image.softmax(dim=-1)
        else:

            if return_feature:
                __, _, image_features, text_features = self.model(image, self.text_tokens,
                                                                  return_feature=return_feature)
                return image_features, text_features
            if replay is not None:
                logits_per_image, _ = self.model(image, self.text_tokens)
                # text_features_for_replay = self.model.encode_text(self.text_tokens[:-self.cfg.increment])
                text_features_for_replay = self.model.encode_text(self.text_tokens)
                text_features_for_replay = text_features_for_replay / text_features_for_replay.norm(dim=1, keepdim=True)
                replay_features = replay / replay.norm(dim=1, keepdim=True)
                replay_logits = replay_features @ text_features_for_replay.t() * 100
            else:
                logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image

        if return_feature:
            text_features = self.model.encode_text(self.all_text_tokens)
            return probs, image_features, text_features

        if replay is not None:
            return probs, replay_logits
        return probs

#bofa
class BofaAdapter(BaseNet):
    def __init__(self, args, pretrained=None):
        super(BaseNet, self).__init__()

        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
        self.args = args
        self.freeze(self.model)

        self.task_id = 0
        self.label2task = {}
        self.mu = None
        self.mu_norm = None
        self.cov_inv = None
        self.cov_list = []
        self.update_cov = None
        self.current_W = None
        self.current_b = None

        self.original_visual_proj = self.visual_proj
        self.original_visual_proj.requires_grad = False
        W0 = self.original_visual_proj.data
        in_dim, out_dim = W0.shape[0], W0.shape[1]
        self.hidden_dim_t = args["Kt"]

        from backbone.linears import OLF as OLF

        self.olf_layer = OLF(in_features=in_dim, out_features=out_dim, W0_torch=W0.T, rank=self.hidden_dim_t)

        self.use_up_cov = args["use_up_cov"]
        self.classifier_list = nn.ModuleList()

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)

        x = self.visual.ln_post(x)
        pooled, _ = self.visual._global_pool(x)
        return pooled

    @property
    def feature_dim(self):
        return self.model.out_dim

    def extract_vector(self, x):
        return self.model.encode_image(x)

    def update_task(self, cls_num):
        new_classifier = nn.Linear(768, cls_num)
        new_classifier.weight.data = self.current_W.T
        new_classifier.bias.data = self.current_b
        new_classifier.weight.requires_grad = True
        new_classifier.bias.requires_grad = True
        self.classifier_list.append(new_classifier)

    def start_train(self, cls_num):
        self.update_task(cls_num=cls_num)
        self.olf_layer.prepare_for_new_task()

    def prepare_stage2(self):
        self.olf_layer.prepare_for_stage2()

    def end_train(self):
        self.olf_layer.end_task()

    def encode_image(self, x, stage2=False, return_origin=False):
        input_features = self.visual_forward_(x)
        norm_input_features = input_features / input_features.norm(dim=-1, keepdim=True)

        cls_results = []
        for cls in self.classifier_list:
            cls_results.append(cls(norm_input_features))

        aligned_features = self.olf_layer(input_features, stage2=stage2)  # (batch, 512)

        if return_origin:
            origin_feature = input_features @ self.original_visual_proj
            return aligned_features, cls_results, origin_feature
        else:
            return aligned_features, cls_results

    def encode_image_eval(self, x):
        input_features = self.visual_forward_(x)
        norm_input_features = input_features / input_features.norm(dim=-1, keepdim=True)

        cls_results = []
        for cls in self.classifier_list:
            cls_results.append(cls(norm_input_features))

        aligned_features = self.olf_layer.eval_forward(input_features)  # (batch, 512)

        return aligned_features, cls_results

    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            vecs_norm = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features = self.visual_forward_(inputs)
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features)
                vecs_norm.append(image_features_norm)
                labels.append(targets)

        for i in range(known_classes, total_classes):
            self.label2task[i] = self.task_id

        vecs = torch.cat(vecs)
        self.olf_layer.update_old_features(vecs)
        vecs_norm = torch.cat(vecs_norm)
        labels = torch.cat(labels)

        mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)],
                       dim=0)
        center_vecs = torch.cat(
            [vecs[labels == i] - mu[i - known_classes] for i in range(known_classes, total_classes)], dim=0)
        cov = torch.cov(center_vecs.t()) + 1e-4 * torch.eye(center_vecs.shape[-1]).to(device)

        mu_norm = torch.cat(
            [vecs_norm[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
        center_vecs_norm = torch.cat([vecs_norm[labels == i] - mu_norm[i - known_classes]
                                      for i in range(known_classes, total_classes)], dim=0)
        cov_inv = center_vecs_norm.shape[1] * torch.linalg.pinv(
            (center_vecs_norm.shape[0] - 1) * center_vecs_norm.T.cov() + center_vecs_norm.T.cov().trace() * torch.eye(
                center_vecs_norm.shape[1]).cuda())
        current_ps = torch.ones(mu_norm.shape[0]).cuda() * 1. / mu_norm.shape[0]
        self.current_W = torch.einsum('nd, dc -> cn', mu_norm, cov_inv)
        self.current_b = current_ps.log() - torch.einsum('nd, dc, nc -> n', mu_norm, cov_inv, mu_norm) / 2

        if self.mu is None:
            self.mu = mu
            self.mu_norm = mu_norm
            self.cov_inv = cov_inv
            self.cov_list = [cov]
            self.update_cov = cov
        else:
            self.cov_inv = (known_classes / total_classes) * self.cov_inv + (
                        total_classes - known_classes) / total_classes * cov_inv + (
                                   (known_classes / total_classes) * (
                                       total_classes - known_classes) / total_classes ** 2) * (
                                   self.mu_norm.T.mean(dim=1).unsqueeze(1) - mu_norm.T.mean(dim=1).unsqueeze(1)) @ (
                                   self.mu_norm.T.mean(dim=1).unsqueeze(1) - mu_norm.T.mean(dim=1).unsqueeze(1)).T
            self.update_cov = (known_classes / total_classes) * self.update_cov + (
                        total_classes - known_classes) / total_classes * cov + (
                                      (known_classes / total_classes) * (
                                          total_classes - known_classes) / total_classes ** 2) * (
                                      self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)) @ (
                                      self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)).T
            self.mu = torch.cat([self.mu, mu])
            self.mu_norm = torch.cat([self.mu_norm, mu_norm])
            self.cov_list.append(cov)

        ps = torch.ones(self.mu_norm.shape[0]).cuda() * 1. / self.mu_norm.shape[0]
        self.W = torch.einsum('nd, dc -> cn', self.mu_norm, self.cov_inv)
        self.b = ps.log() - torch.einsum('nd, dc, nc -> n', self.mu_norm, self.cov_inv, self.mu_norm) / 2
        self.task_id += 1

    def sample_augmented_cls(self, classes: list, n: int):
        aug_features = []
        aug_labels = []

        for c in classes:
            if c not in self.label2task:
                raise ValueError(f"Class {c} not found in stored tasks")
            task_id = self.label2task[c]

            if self.use_up_cov:
                cov = self.update_cov
            else:
                cov = self.cov_list[task_id]

            mean = self.mu[c]
            vec = torch.randn(n, mean.shape[-1]).to(mean.device)
            sqrt_cov = torch.linalg.cholesky(cov)
            aug_c = vec @ sqrt_cov + mean

            aug_features.append(aug_c)
            aug_labels.extend([c] * n)

        X_aug = torch.cat(aug_features, dim=0)
        y_aug = torch.tensor(aug_labels, dtype=torch.long)
        return X_aug, y_aug

    def get_cls_center(self):
        return self.mu @ self.visual_proj

    def get_cls_center_last(self):
        with torch.no_grad():
            return self.olf_layer(self.mu)

    def get_cls_center_lora(self):
        with torch.no_grad():
            training_state = self.olf_layer.training
            self.olf_layer.eval()
            new_center = self.olf_layer(self.mu)
            self.olf_layer.train(training_state)
        return new_center

    def get_param_group(self):
        param_groups = []
        param_groups.append({'params': self.olf_layer.get_trainable_parameters()})
        param_groups.append({'params': self.olf_layer.get_stage2_parameters(), 'lr': 0.001, 'weight_decay': 0.001})

        if len(self.classifier_list) > 0:
            param_groups.append({'params': self.classifier_list[-1].parameters(),
                                 'lr': 0.001, 'weight_decay': 0.001})

        return param_groups

    def train_state(self):
        self.olf_layer.train()

    def eval_state(self):
        self.olf_layer.eval()

    def encode_text(self, x):
        return self.model.encode_text(x)

#ease
class EaseNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(EaseNet, self).__init__()
        self.args = args
        self._device = args["device"][0]
        import open_clip
        basic_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained='laion400m_e32')
        basic_model.load_state_dict(torch.load("./c.pth", map_location="cpu"))
        sd = basic_model.state_dict()

        vision_width = sd["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in sd.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch = sd["visual.conv1.weight"].shape[-1]
        grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch * grid
        embed_dim = sd["text_projection"].shape[1]  # usually 512

        from easydict import EasyDict
        self.tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=args["ffn_num"],
            d_model=vision_width,
            vpt_on=False,
            vpt_num=0,
            _device=self._device,
        )

        from backbone.ease_model import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            tuning_config=self.tuning_config,
        ).to(self._device)

        visual_sd = basic_model.visual.state_dict()
        self.backbone.load_state_dict(visual_sd, strict=False)
        self.backbone.eval()

        # 5) EASE incremental params
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1

        self.out_dim = embed_dim
        self.fc = None
        self.proxy_fc = None

        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

    @property
    def feature_dim(self):
        return self.out_dim * (self._cur_task + 2) if self.use_init_ptm else self.out_dim * (self._cur_task + 1)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def generate_fc(self, in_dim, out_dim):
        return EaseCosineLinear(in_dim, out_dim)

    def update_fc(self, nb_classes):
        self._cur_task += 1

        if self._cur_task > 0:
            self.backbone.add_adapter_to_list()

        # proxy_fc
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        # fc
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if hasattr(fc, "reset_parameters_to_zero"):
            fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb = self.fc.out_features
            if hasattr(self.fc, "sigma") and hasattr(fc, "sigma"):
                fc.sigma.data = self.fc.sigma.data
            fc.weight.data[:old_nb, :-self.out_dim] = copy.deepcopy(self.fc.weight.data)

        self.fc = fc

        # only train current adapter + proxy_fc
        self.freeze()
        for p in self.backbone.cur_adapter.parameters():
            p.requires_grad = True
        for p in self.proxy_fc.parameters():
            p.requires_grad = True

    def forward(self, x, test=False):
        x = x.to(self._device)

        if not test:
            feat = self.backbone.forward_train(x)  # [B, out_dim]
            out = self.proxy_fc(feat)
            out.update({"features": feat})
            return out

        feat_cat = self.backbone.forward_test(x, use_init_ptm=self.use_init_ptm)
      #  feat_cat = torch.cat(feats, dim=1) if isinstance(feats, (list, tuple)) else feats

        if self.args.get("moni_adam") or (not self.args.get("use_reweight")):
            out = self.fc(feat_cat)
        else:
         #   print("fyl")
            out = self.fc.forward_reweight(
                feat_cat,
                cur_task=self._cur_task,
                alpha=self.alpha,
                init_cls=self.init_cls,
                inc=self.inc,
                use_init_ptm=self.use_init_ptm,
                out_dim=self.out_dim,
                beta=self.beta,
            )
        out.update({"features": feat_cat})
      #  print(out["logits"].size())
       # print(out.shape)
        return out

    def extract_vector(self, x):
        x = x.to(self._device)
        return self.backbone.forward_train(x)

#tuna
class TUNANet(nn.Module):
    def __init__(self, args, pretrained=True):
        super().__init__()
        self.args = args
        self._device = args["device"][0]

        import open_clip
        basic_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained='laion400m_e32')
        basic_model.load_state_dict(torch.load(".p/c.pth", map_location="cpu"))
        sd = basic_model.state_dict()

        vision_width = sd["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in sd.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch = sd["visual.conv1.weight"].shape[-1]
        grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch * grid
        embed_dim = sd["text_projection"].shape[1]

        from easydict import EasyDict
        self.tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",                 # "parallel" or "sequential"
            ffn_adapter_layernorm_option="none",   # "none"/"in"/"out"
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=args.get("ffn_num", 16),       # bottleneck
            d_model=vision_width,
            _device=self._device,
            vpt_on=False,
            vpt_num=0,
        )

        from backbone.tuna_model import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            tuning_config=self.tuning_config,
        ).to(self._device)

        visual_sd = basic_model.visual.state_dict()
        self.backbone.load_state_dict(visual_sd, strict=False)
        self.backbone.eval()

        self._cur_task = -1
        self.out_dim = embed_dim

        self.fc = None

    @property
    def feature_dim(self):
        return self.out_dim

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def generate_fc(self, in_dim, out_dim):
        fc = TunaLinear(in_dim, out_dim).to(self._device)
        return fc

    def update_fc(self, nb_classes, nextperiod_initialization=None, reset_new_adapter: bool = False):
        self._cur_task += 1


        if self._cur_task > 0:
            self.backbone.adapter_update(reset_new=reset_new_adapter)

        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            if hasattr(self.fc, "update"):
                self.fc.update(nb_classes, freeze_old=False)
            else:
                new_fc = self.generate_fc(self.feature_dim, nb_classes)
                old_nb = self.fc.out_features
                new_fc.weight.data[:old_nb] = copy.deepcopy(self.fc.weight.data)
                if hasattr(self.fc, "sigma") and hasattr(new_fc, "sigma"):
                    new_fc.sigma.data = self.fc.sigma.data
                self.fc = new_fc

        self.freeze()
        for p in self.backbone.cur_adapter.parameters():
            p.requires_grad = True
        for p in self.fc.parameters():
            p.requires_grad = True

    def merge_adapters(self):

        self.backbone.merge()

    def forward(self, x, adapter_id: int = None, train: bool = False, fc_only: bool = False):
        x = x.to(self._device)


        if fc_only:
            out = self.fc(x)

            if isinstance(out, dict):
                return out
            return {"logits": out}

        if adapter_id is None:
            adapter_id = len(self.backbone.adapter_list)  # current

        res = self.backbone(x, adapter_id=adapter_id, train=train)  # {"features": feat}
        feat = res["features"]

        logits_out = self.fc(feat)
        if isinstance(logits_out, dict):
            res.update(logits_out)
        else:
            res["logits"] = logits_out

        return res

    def extract_vector(self, x):
        x = x.to(self._device)
        adapter_id = len(self.backbone.adapter_list)
        return self.backbone(x, adapter_id=adapter_id, train=True)["features"]


#ranpac/adapter
class AdapterVitNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)

        self.W_rand = None
        self.RP_dim = None

        self.args = args
        self._device = args["device"][0]
        import open_clip
        basic_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained='laion400m_e32')

        ckpt_path = args.get("clip_ckpt_path", "./c.pth")
        basic_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        sd = basic_model.state_dict()

        vision_width = sd["visual.conv1.weight"].shape[0]  # 768
        vision_layers = len([k for k in sd.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch = sd["visual.conv1.weight"].shape[-1]  # 16
        grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch * grid              # 224
        embed_dim = sd["text_projection"].shape[1]

        from easydict import EasyDict
        ffn_num = args["ffn_num"]

        self.tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=vision_width,
            # VPT
            vpt_on=False,
            vpt_num=0,
            _device=self._device,
        )

        from backbone.aper_adapter import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            tuning_config=self.tuning_config,
        ).to(self._device)

        visual_sd = basic_model.visual.state_dict()
        msg = self.backbone.load_state_dict(visual_sd, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False

        missing = set(msg.missing_keys)
        for n, p in self.backbone.named_parameters():
            if n in missing:
                p.requires_grad = True
     #   self.backbone.load_state_dict(visual_sd, strict=False)

        self.out_dim = embed_dim

        self.fc = None

        self.backbone.eval()

    @property
    def feature_dim(self):
        return self.out_dim

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.RP_dim if self.RP_dim is not None else self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                nn.init.constant_(fc.sigma, 1.0)
          #  fc.sigma.data = self.fc.sigma.data

            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([
                    weight,
                    torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)
                ])
            fc.weight = nn.Parameter(weight)

        del self.fc
        self.fc = fc

    def extract_vector(self, x):
        x = x.to(self._device)
        y = self.backbone(x)
        if isinstance(y, dict):
            return y["features"]
        return y

    def forward(self, x):
        feat = self.extract_vector(x)

        if self.W_rand is not None:
            feat = torch.nn.functional.relu(feat @ self.W_rand)

        out = self.fc(feat)
        out.update({"features": feat})
        return out

#ssf
class SSFVitNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)

        self.args = args
        self._device = args["device"][0]


        import open_clip
        basic_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained='laion400m_e32')

        ckpt_path = args.get("clip_ckpt_path", "./c.pth")
        basic_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        sd = basic_model.state_dict()


        vision_width = sd["visual.conv1.weight"].shape[0]  # 768
        vision_layers = len([k for k in sd.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch = sd["visual.conv1.weight"].shape[-1]  # 16
        grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch * grid              # 224
        embed_dim = sd["text_projection"].shape[1]


        from backbone.aper_ssf import VisionTransformer
        self.backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            tuning_mode= "ssf"
        ).to(self._device)

        visual_sd = basic_model.visual.state_dict()
        msg = self.backbone.load_state_dict(visual_sd, strict=False)
        self.out_dim = embed_dim

        # classifier
        self.fc = None

        self.backbone.eval()

    @property
    def feature_dim(self):
        return self.out_dim

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                nn.init.constant_(fc.sigma, 1.0)
          #  fc.sigma.data = self.fc.sigma.data

            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([
                    weight,
                    torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)
                ])
            fc.weight = nn.Parameter(weight)

        del self.fc
        self.fc = fc

    def extract_vector(self, x):
        x = x.to(self._device)
        y = self.backbone(x)
        if isinstance(y, dict):
            return y["features"]
        return y

    def forward(self, x):
        feat = self.extract_vector(x)

        out = self.fc(feat)
        out.update({"features": feat})
        return out

#vpt
class VPTVitNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)

        self.args = args
        self._device = args["device"][0]
        import open_clip
        basic_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained='laion400m_e32')

        ckpt_path = args.get("clip_ckpt_path", "./c.pth")
        basic_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        sd = basic_model.state_dict()

        vision_width = sd["visual.conv1.weight"].shape[0]  # 768
        vision_layers = len([k for k in sd.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch = sd["visual.conv1.weight"].shape[-1]  # 16
        grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch * grid              # 224
        embed_dim = sd["text_projection"].shape[1]
        from easydict import EasyDict

        tuning_config = EasyDict(
            ffn_adapt=args.get("ffn_adapt", False),
            ffn_option=args.get("ffn_option", "parallel"),
            ffn_adapter_layernorm_option=args.get("ffn_adapter_layernorm_option", "none"),
            ffn_adapter_init_option=args.get("ffn_adapter_init_option", "lora"),
            ffn_adapter_scalar=args.get("ffn_adapter_scalar", "0.1"),
            ffn_num=args.get("ffn_num", 0),
            d_model=vision_width,
            # VPT
            vpt_on=True,
            vpt_num=args.get("vpt_num", 10),  # Prompt_Token_num
            vpt_type=args.get("vpt_type", "deep"),
            _device=self._device,
        )
        print(vision_layers)
        from backbone.vpt import VPT_VisionTransformer
        self.backbone = VPT_VisionTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch,
        width=vision_width,
        layers=vision_layers,
        heads=vision_width // 64,
        output_dim=embed_dim,
        tuning_config=tuning_config,
        ).to(self._device)

        visual_sd = basic_model.visual.state_dict()
        self.backbone.load_state_dict(visual_sd, strict=False)
        self.backbone.Freeze()

        self.out_dim = embed_dim

        # classifier
        self.fc = None

        self.backbone.eval()

    @property
    def feature_dim(self):
        return self.out_dim

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                nn.init.constant_(fc.sigma, 1.0)
          #  fc.sigma.data = self.fc.sigma.data

            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([
                    weight,
                    torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)
                ])
            fc.weight = nn.Parameter(weight)

        del self.fc
        self.fc = fc

    def extract_vector(self, x):
        x = x.to(self._device)
        y = self.backbone(x)
        if isinstance(y, dict):
            return y["features"]
        return y

    def forward(self, x):
        feat = self.extract_vector(x)

        out = self.fc(feat)
        out.update({"features": feat})
        return out


class VitNet(BaseNet):

    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)

        self.args = args
        self._device = args["device"][0]
        self.backbone = get_vitbackbone(args =args)
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.out_dim = self.backbone.output_dim

        # classifier
        self.fc = None

        self.backbone.eval()

    @property
    def feature_dim(self):
        return self.out_dim

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                nn.init.constant_(fc.sigma, 1.0)
          #  fc.sigma.data = self.fc.sigma.data

            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([
                    weight,
                    torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)
                ])
            fc.weight = nn.Parameter(weight)

        del self.fc
        self.fc = fc

    def extract_vector(self, x):
        x = x.to(self._device)
        y = self.backbone(x)
        if isinstance(y, dict):
            return y["features"]
        return y

    def forward(self, x):
        feat = self.extract_vector(x)

        out = self.fc(feat)
        out.update({"features": feat})
        return out
#aper
def get_vitbackbone(args, pretrained=False):
    _device = args["device"][0]
    import open_clip
    basic_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained='laion400m_e32')

    ckpt_path = args.get("clip_ckpt_path", "./c.pth")
    basic_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    sd = basic_model.state_dict()

    vision_width = sd["visual.conv1.weight"].shape[0]  # 768
    vision_layers = len([k for k in sd.keys()
                         if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch = sd["visual.conv1.weight"].shape[-1]  # 16
    grid = round((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch * grid  # 224
    embed_dim = sd["text_projection"].shape[1]

    from easydict import EasyDict
  #  ffn_num = args["ffn_num"]

    tuning_config = EasyDict(
        ffn_adapt=False,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.get("ffn_num", 0),
        d_model=vision_width,
        # VPT
        vpt_on=False,
        vpt_num=0,
        _device=_device,
    )

    from backbone.aper_adapter import VisionTransformer
    backbone = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            tuning_config=tuning_config,
        ).to(_device)

    visual_sd = basic_model.visual.state_dict()
    msg = backbone.load_state_dict(visual_sd, strict=False)
    for p in backbone.parameters():
            p.requires_grad = False

    return backbone.eval()

#aper
class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        # no need the backbone.

        print(
            'Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone = torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args = args
        self.model_type = 'vit'
        self._device = args["device"][0]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if hasattr(self.fc, 'sigma') and self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
            elif hasattr(fc, 'sigma') and fc.sigma is not None:
                # 如果旧层没 sigma 但新层有，初始化为 1.0 或保持默认
                nn.init.constant_(fc.sigma, 1.0)
            # fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        feats = []
        for backbone in self.backbones:
            y = backbone(x)
            if isinstance(y, dict):
                y = y["features"]
            feats.append(y)

        features = torch.cat(feats, dim=1)
        out = self.fc(features)
        out.update({"features": features})
        return out

    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['model_name']:
            newargs = copy.deepcopy(self.args)
          #  newargs['backbone_type'] = newargs['backbone_type'].replace('_ssf', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_vitbackbone(args = newargs))  # pretrained model without scale
        elif 'vpt' in self.args['backbone_type']:
            newargs = copy.deepcopy(self.args)
           # newargs['backbone_type'] = newargs['backbone_type'].replace('_vpt', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_vitbackbone(args = newargs))  # pretrained model without vpt
        elif 'adapter' in self.args['model_name']:
            newargs = copy.deepcopy(self.args)
           # newargs['backbone_type'] = newargs['backbone_type'].replace('_adapter', '')
            print(newargs['backbone_type'])
            self.backbones.append(get_vitbackbone(args = newargs))  # pretrained model without adapter
        else:
            self.backbones.append(get_vitbackbone(args = self.args))  # the pretrained model itself

        self.backbones.append(tuned_model.backbone)  # adappted tuned model

        self._feature_dim = self.backbones[0].output_dim * len(self.backbones)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])

