import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPZeroShot(nn.Module):
    def __init__(self, clip_model):
        super(CLIPZeroShot, self).__init__()
        self.clip_model = clip_model
        for param in self.parameters():
            param.requires_grad = False

    def process_text(self, text):
        text = self.clip_model.tokenizer(text).to(device)
        text_embeds = self.clip_model.encode_text(text)
        return text_embeds

    def forward(self, samples):
        images = samples.get("image")
        text_embeds = samples.get("text_input")
        image_embeds = self.clip_model.encode_image(images)
        image_features = F.normalize(image_embeds, dim=-1)
        text_features = F.normalize(text_embeds, dim=-1)
        similarity = (100.0 * image_features @ text_features.T)
        return similarity.softmax(dim=1)


class BlipZeroShot(nn.Module):
    def __init__(self, blip_model):
        super(BlipZeroShot, self).__init__()
        self.blip_model = blip_model
        for param in self.parameters():
            param.requires_grad = False

    def process_text(self, text):
        text = self.blip_model.tokenizer(text, return_tensors="pt", padding=True).to(device)
        text_output = self.blip_model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        return text_embeds

    def forward(self, samples):
        image = samples.get("image")
        text_embeds = samples.get("text_input")

        image_embeds = self.blip_model.visual_encoder.forward_features(image)
        image_features = self.blip_model.vision_proj(image_embeds)
        image_features = F.normalize(image_features, dim=-1)
        image_features = image_features[:, 0]
        text_features = self.blip_model.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features[:, 0]

        similarity = (100.0 * image_features @ text_features.T)
        return similarity.softmax(dim=1)


class Blip2ZeroShot(nn.Module):
    def __init__(self, blip2_model):
        super(Blip2ZeroShot, self).__init__()
        self.blip2_model = blip2_model
        for param in self.parameters():
            param.requires_grad = False

    def process_text(self, text):
        text = self.blip2_model.tokenizer(text, return_tensors="pt", padding=True).to(
            self.blip2_model.device
        )

        text_output = self.blip2_model.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        return text_embeds

    def forward(self, samples):
        image = samples.get("image")
        text_embeds = samples.get("text_input")
        with self.blip2_model.maybe_autocast():
            image_embeds_frozen = self.blip2_model.ln_vision(self.blip2_model.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.blip2_model.device)
        query_tokens = self.blip2_model.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.blip2_model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.blip2_model.vision_proj(image_embeds), dim=-1)
        image_features = image_features[:, 0]

        text_features = self.blip2_model.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features[:, 0]
        similarity = (100.0 * image_features @ text_features.T)
        return similarity.softmax(dim=1)


class ClipLinearProbe(nn.Module):
    def __init__(self, clip_model, num_categories):
        super(ClipLinearProbe, self).__init__()
        self.clip_model = clip_model
        self.feature_size = self.clip_model.visual.output_dim
        self.linear = nn.Linear(self.feature_size, num_categories)

        # Set requires_grad for the linear layer and freeze other layers
        for param in self.parameters():
            param.requires_grad = False
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

    def forward(self, images):
        image_embeds = self.clip_model.encode_image(images)
        logits = self.linear(image_embeds)
        return torch.softmax(logits, dim=1)


class BlipLinearProbe(nn.Module):
    def __init__(self, blip_model, num_categories):
        super(BlipLinearProbe, self).__init__()
        self.blip_model = blip_model
        self.feature_size = self.blip_model.visual_encoder.embed_dim
        self.linear = nn.Linear(self.feature_size, num_categories)

        # Set requires_grad for the linear layer and freeze other layers
        for param in self.parameters():
            param.requires_grad = False
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

    def forward(self, images):
        image_embeds = self.blip_model.visual_encoder.forward_features(images)
        image_embeds = image_embeds[:, 0]
        logits = self.linear(image_embeds)
        return torch.softmax(logits, dim=1)


class Blip2LinearProbe(nn.Module):
    def __init__(self, blip2_model, num_categories):
        super(Blip2LinearProbe, self).__init__()
        self.blip2_model = blip2_model
        self.feature_size = self.blip2_model.visual_encoder.embed_dim
        self.linear = nn.Linear(self.feature_size, num_categories)

        # Set requires_grad for the linear layer and freeze other layers
        for param in self.parameters():
            param.requires_grad = False
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

    def forward(self, images):
        with self.blip2_model.maybe_autocast():
            image_embeds = self.blip2_model.visual_encoder(images).float()
        image_embeds = image_embeds[:, 0]
        logits = self.linear(image_embeds)
        return torch.softmax(logits, dim=1)

    
class ClipLoRA(nn.Module):
    def __init__(self, clip_model, num_categories):
        super(ClipLoRA, self).__init__()

        # Initialize LoRA
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['c_fc'],
            lora_dropout=0.1,
            bias="none"
        )
        self.clip_model = get_peft_model(clip_model.visual, config)
        print(self.clip_model)
        self.feature_size = self.clip_model.output_dim
        self.linear = nn.Linear(self.feature_size, num_categories)

    def forward(self, images):
        image_embeds = self.clip_model(images)
        logits = self.linear(image_embeds)
        return torch.softmax(logits, dim=1)


class BlipLoRA(nn.Module):
    def __init__(self, blip_model, num_categories):
        super(BlipLoRA, self).__init__()

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['qkv'],
            lora_dropout=0.1,
            bias="none"
        )
        self.blip_model = get_peft_model(blip_model.visual_encoder, config)
        print(self.blip_model)
        self.feature_size = self.blip_model.embed_dim
        self.linear = nn.Linear(self.feature_size, num_categories)

    def forward(self, images):
        image_embeds = self.blip_model.forward_features(images)
        image_embeds = image_embeds[:, 0]
        logits = self.linear(image_embeds)
        return torch.softmax(logits, dim=1)


class Blip2LoRA(nn.Module):
    def __init__(self, blip2_model, num_categories):
        super(Blip2LoRA, self).__init__()
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['qkv'],
            lora_dropout=0.1,
            bias="none"
        )
        self.maybe_autocast = blip2_model.maybe_autocast
        self.blip2_model = get_peft_model(blip2_model.visual_encoder, config)
        print(self.blip2_model)
        self.feature_size = self.blip2_model.embed_dim
        self.linear = nn.Linear(self.feature_size, num_categories)

    def forward(self, images):
        with self.maybe_autocast():
            image_embeds = self.blip2_model(images).float()
        image_embeds = image_embeds[:, 0]
        logits = self.linear(image_embeds)
        return torch.softmax(logits, dim=1)

