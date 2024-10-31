import torch
import torch.nn as nn
import clip

class CLIPEncoder(nn.Module):
    def __init__(self, output_channels=480, target_width=150, fusion_method='concat'):
        super(CLIPEncoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP ViT-B/32 model
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        assert fusion_method in ['concat', 'sum'], "Fusion method must be either 'concat' or 'sum'"
        self.fusion_method = fusion_method
        
        # Depending on the fusion method, the input size to the fc layer changes
        if self.fusion_method == 'concat':
            fusion_size = 512 * 2  # Concatenate image and text features
        else:
            fusion_size = 512  # Element-wise sum keeps the size as 512

        # Project fused features to (output_channels * target_width)
        self.fc = nn.Linear(fusion_size, output_channels * target_width)
        
        # Store target width for reshaping
        self.target_width = target_width
        self.output_channels = output_channels
        
    def forward(self, image, text=None):
        # Extract image features from CLIP (shape: [B, 512])
        image_features = self.clip_model.encode_image(image)
        if text is not None: 
            text_features = self.clip_model.encode_text(text.long())
            # Fuse image and text features
            if self.fusion_method == 'concat':
                # Concatenate image and text features along the last dimension
                fused_features = torch.cat((image_features, text_features), dim=-1)  # Shape: [B, 1024]
            else:
                # Element-wise sum of image and text features
                fused_features = image_features + text_features  # Shape: [B, 512]

        # Project fused features to the required size (output_channels * target_width)
        x = self.fc(fused_features)  # Shape: [B, output_channels * target_width]
        
        # Reshape to (B, output_channels, 1, target_width)
        x = x.view(x.size(0), self.output_channels, 1, self.target_width)
        
        return x

# image_tensor = torch.randn(2, 3, 224, 224).to("cuda")  # Example batch of 2 images
# text_input = clip.tokenize(["This is an example sentence.", "Another example sentence."]).to("cuda").float()

# model = CLIPEncoder(output_channels=480, target_width=150, fusion_method='concat').to("cuda").float()

# output = model(image_tensor, text_input)

# print(output.shape)  
