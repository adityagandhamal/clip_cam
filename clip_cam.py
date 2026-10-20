import torch
from clip import clip
from torch.nn import functional as F
from einops import rearrange
import utils
import argparse

def parse_args():    
        parser = argparse.ArgumentParser(description="CLIP Image-Text Matching")
        parser.add_argument('--model_name', type=str, default="ViT-B/16", help='Name of the CLIP model')
        parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
        parser.add_argument('--text', type=str, required=True, help='Text input for matching (e.g., "a photo of a cat")')
        parser.add_argument('--checkpoint', type=str, default=None, help='Path to CLIP checkpoint')
        
        return parser.parse_args()

def main(args):        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model_name = "ViT-B/16"
        model = clip.load_clip(args.model_name, device, checkpoint=args.checkpoint)

        image = utils.process_input(args.image_path, device=device)

        vis_f = model.encode_image(image, dense=True)
        vis_f = F.normalize(vis_f, dim=-1)

        text = [str(args.text)]
        tokens = clip.tokenize(text).to(device)
        text_embeddings = model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.unsqueeze(1)

        H, W = image.shape[-2:]
        grid_size = int((vis_f.shape[1] - 1) ** 0.5)
        img_txt_matching = vis_f[:,1:] @ torch.permute(text_embeddings, (1,2,0))
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h', w=grid_size)
        img_txt_matching = F.interpolate(img_txt_matching, size=(H, W), mode='bilinear')

        img_txt_matching = utils.min_max(img_txt_matching)

        utils.visualize(image, text, img_txt_matching)

if __name__ == "__main__":
        args = parse_args()
        main(args)