import torch
from clip import clip
from torch.nn import functional as F
from einops import rearrange
import utils
import argparse
import os

def parse_args():    
        parser = argparse.ArgumentParser(description="CLIP Image-Text Matching")
        parser.add_argument('--model_name', type=str, default="ViT-B/16", help='Name of the CLIP model')
        parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
        parser.add_argument('--text', type=str, required=True, help='Text input for matching (e.g., "a photo of a cat")')
        parser.add_argument('--checkpoint', type=str, default=None, help='Path to CLIP checkpoint')
        parser.add_argument('--resolution', type=int, default=384, help='Input image resolution')
        parser.add_argument('--compare_checkpoint', type=str, default=None,
                            help='Path to a second CLIP checkpoint (e.g. fine-tuned) to compare side by side')

        return parser.parse_args()

def similarity_map(model, image, text):
        """Dense image-text similarity map, min-max normalized per prompt. Shape: (1, num_prompts, H, W)."""
        device = image.device
        vis_f = model.encode_image(image, dense=True)
        vis_f = F.normalize(vis_f, dim=-1)

        tokens = clip.tokenize(text).to(device)
        text_embeddings = model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.unsqueeze(1)

        H, W = image.shape[-2:]
        grid_size = int((vis_f.shape[1] - 1) ** 0.5)
        img_txt_matching = vis_f[:,1:] @ torch.permute(text_embeddings, (1,2,0))
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h', w=grid_size)
        img_txt_matching = F.interpolate(img_txt_matching, size=(H, W), mode='bilinear')

        return utils.min_max(img_txt_matching)

def checkpoint_label(checkpoint):
        if checkpoint:
                return os.path.basename(checkpoint)        
        else:
                return 'pretrained'

def main(args=None):
        if args is None:
                args = parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = utils.process_input(args.image_path, device=device, resolution=args.resolution)
        text = [str(args.text)]

        model = clip.load_clip(args.model_name, device, checkpoint=args.checkpoint)
        img_txt_matching = similarity_map(model, image, text)

        if args.compare_checkpoint is None:
                utils.visualize(image, text, img_txt_matching)
                return

        del model
        compare_model = clip.load_clip(args.model_name, device, checkpoint=args.compare_checkpoint)
        compare_matching = similarity_map(compare_model, image, text)
        labels = [checkpoint_label(args.checkpoint), checkpoint_label(args.compare_checkpoint)]
        utils.visualize_compare(image, text, [img_txt_matching, compare_matching], labels)

if __name__ == "__main__":
        args = parse_args()
        main(args)