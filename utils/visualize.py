from PIL import Image
from matplotlib import pyplot as plt
import cv2
import torch
import numpy as np
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def to_pil(image, size):
        if isinstance(image, Image.Image):
            return image.resize(size)
        elif isinstance(image, torch.Tensor):
            if image.ndim > 3:
                image = image.squeeze(0)
            image_unormed = (image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                             + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]  # undo the normalization
            return Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL
        else:
            raise TypeError(f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}')

def overlay(image, logits, alpha=0.6):
        """Blend a jet heatmap per prompt over the image. Returns a list of RGB uint8 arrays."""
        if logits.ndim > 3:
            logits = logits.squeeze(0)
        logits = logits.detach().cpu().numpy()

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logits = (logits * 255).astype('uint8')
        heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logits]

        vizs = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]
        return [cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB) for viz in vizs]

def visualize(image, text, logits):
        W, H = logits.shape[-2:]
        image = to_pil(image, (W, H))

        for viz, cls_name in zip(overlay(image, logits), text):
            plt.imshow(viz)
            plt.title(cls_name)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{cls_name}_.png', bbox_inches='tight')
            plt.show()
            plt.close()

def visualize_compare(image, text, logits_list, labels):
        """Show the maps of several models side by side, one figure per prompt."""
        W, H = logits_list[0].shape[-2:]
        image = to_pil(image, (W, H))
        vizs_per_model = [overlay(image, logits) for logits in logits_list]

        for i, cls_name in enumerate(text):
            fig, axes = plt.subplots(1, len(logits_list), figsize=(5 * len(logits_list), 5))
            for ax, vizs, label in zip(axes, vizs_per_model, labels):
                ax.imshow(vizs[i])
                ax.set_title(label)
                ax.axis('off')
            fig.suptitle(cls_name)
            plt.tight_layout()
            plt.savefig(f'{cls_name}_compare.png', bbox_inches='tight')
            plt.show()
            plt.close()
