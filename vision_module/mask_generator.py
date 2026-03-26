"""
Adapted from: https://github.com/fitz0401/tasc/blob/main/tasc/vision_module/mask_generator.py
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import groundingdino.datasets.transforms as T
from segment_anything import build_sam, SamPredictor
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, predict, annotate
from torchvision.ops import nms
from PIL import Image
from pathlib import Path

file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add XMem2 to path for mask tracking
XMEM_PATH = '/home/u0177383/XMem2'
if os.path.exists(XMEM_PATH) and XMEM_PATH not in sys.path:
    sys.path.insert(0, XMEM_PATH)

class MaskGenerator:
    def __init__(self):
        self.ckpt_path = '/home/u0177383/doi_policy/ckpts'
        self.config_path = '/home/u0177383/doi_policy/foci_real_world/configs'
        self.device = device
        self.dino, self.sam = self._prepare_dino_and_sam()

    def _prepare_dino_and_sam(self):
        DINO_WEIGHTS_PATH = os.path.join(self.ckpt_path, "groundingdino_swint_ogc.pth")
        DINO_CONFIG_PATH = os.path.join(self.config_path, "grounding_dino.py")
        dino = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)
        SAM_WEIGHTS_PATH = os.path.join(self.ckpt_path, "sam_vit_h_4b8939.pth")
        sam = build_sam(checkpoint=SAM_WEIGHTS_PATH)
        sam.to(self.device)
        return dino, sam
    
    def _load_pil_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_np = np.asarray(image_pil)
        image_transformed, _ = transform(image_pil, None)
        self.H, self.W, _ = image_np.shape
        return image_np, image_transformed

    @staticmethod
    def _show_mask(masks, frame, random_color=True):
        annotated_frame_pil = Image.fromarray(frame).convert("RGBA")
        for i in range(masks.shape[0]):
            mask = masks[i][0]
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
            annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
        return np.array(annotated_frame_pil)
    
    def get_scene_object_bboxes(self, image, texts, visualize=False, logdir=None):
        # mask out gripper region
        image_np, image_transformed = self._load_pil_image(image)
        # get bounding box
        BOX_THRESHOLD = 0.3
        TEXT_THRESHOLD = 0.22 # can be tuned for different tasks
        all_boxes, all_logits, all_phrases = [], [], []
        for text in texts:
            boxes, logits, phrases = predict(
                model=self.dino,
                image=image_transformed,
                caption=text,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            id = torch.argmax(logits)
            all_boxes.append(boxes[id])
            all_logits.append(logits[id])
            all_phrases.append(phrases[id])
        boxes, logits, phrases = torch.stack(all_boxes, dim=0), torch.stack(all_logits, dim=0), all_phrases
        
        if visualize:
            annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
            # %matplotlib inline
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            annotated_frame_pil = Image.fromarray(annotated_frame)
            if logdir is not None:
                annotated_frame_pil.save(os.path.join(logdir, 'bbox.png'))
            plt.imshow(annotated_frame_pil)
            plt.show()
        return boxes, logits, phrases

    def filter_masks(self, boxes, scores, phrases, iou_threshold=0.5):
        # filter out masks by IOU
        device = boxes.device
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([self.W, self.H, self.W, self.H], device=device)
        keep_indices = nms(boxes_xyxy, scores, iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        phrases = [phrases[i] for i in keep_indices.tolist()]
        return boxes, scores, phrases

    def get_segmentation_masks(self, image, boxes, logits, phrases, visualize=False, save_path=None):
        image_np, _ = self._load_pil_image(image)
        device = boxes.device
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([self.W, self.H, self.W, self.H], device=device)
        # get segmentation based on bounding box
        sam_predictor = SamPredictor(self.sam)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(self.device)
        sam_predictor.set_image(image_np)
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            masks = masks.cpu().numpy()
        if visualize:
            annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
            # %matplotlib inline
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            mask_annotated_img = self._show_mask(masks, annotated_frame)
            mask_annotated_img_pil = Image.fromarray(mask_annotated_img)
            plt.imshow(mask_annotated_img_pil)
            plt.show()
            if save_path is not None:
                mask_annotated_img_pil.save(save_path)
        return masks[:, 0]
    
    def track_masks_with_xmem(self, demo_dir, object_name, initial_mask, verbose=False):
        """ Run XMem++ tracking on a demo directory. """
        if not os.path.isdir(XMEM_PATH):
            if verbose:
                print("Warning: XMem++ not available, falling back to mask reuse")
            return {'masks': {}, 'success': False}
        
        demo_path = Path(demo_dir).expanduser().resolve()
        imgs_path = str(demo_path / 'color')
        masks_path = str(demo_path / 'mask' / object_name)
        output_path = str(demo_path / f'xmem_output_{object_name}')
        
        # Create mask directory and save initial mask
        mask_dir = Path(masks_path)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial mask for frame 0.
        mask_with_id = (initial_mask > 0).astype(np.uint8)
        frame0_candidates = sorted((demo_path / 'color').glob('*.png'))
        frame0_name = frame0_candidates[0].name if frame0_candidates else '00000.png'
        initial_mask_path = mask_dir / frame0_name
        cv2.imwrite(str(initial_mask_path), mask_with_id)
        
        if verbose:
            print(f"Running XMem++ tracking for {object_name}...")
            print(f"  Input images: {imgs_path}")
            print(f"  Initial mask: {initial_mask_path}")
            print(f"  Output: {output_path}")
        
        # Run XMem++ tracking with first frame annotation
        frames_with_masks = [0]  # Only annotate first frame

        # Use subprocess with spawn start method to avoid: "Cannot re-initialize CUDA in forked subprocess" from DataLoader workers.
        payload = {
            'imgs_path': imgs_path,
            'masks_path': masks_path,
            'output_path': output_path,
            'frames_with_masks': frames_with_masks,
            'xmem_path': XMEM_PATH,
        }
        subprocess_code = (
            "import json, os, sys, multiprocessing as mp\n"
            "mp.set_start_method('spawn', force=True)\n"
            "payload = json.loads(os.environ['XMEM_PAYLOAD'])\n"
            "xmem_path = payload['xmem_path']\n"
            "if xmem_path not in sys.path:\n"
            "    sys.path.insert(0, xmem_path)\n"
            "import inference.run_on_video as rov\n"
            "from inference.data.video_reader import VideoReader\n"
            "from torch.utils.data import DataLoader\n"
            "def _create_dataloaders_single_worker(imgs_in_path, masks_in_path, config):\n"
            "    vid_reader = VideoReader('', imgs_in_path, masks_in_path, size=config['size'], use_all_masks=True)\n"
            "    loader = DataLoader(vid_reader, batch_size=None, shuffle=False, num_workers=0, collate_fn=VideoReader.collate_fn_identity)\n"
            "    vid_length = len(loader)\n"
            "    config['enable_long_term_count_usage'] = (\n"
            "        config['enable_long_term'] and\n"
            "        (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) * config['num_prototypes']) >= config['max_long_term_elements']\n"
            "    )\n"
            "    return vid_reader, loader\n"
            "rov._create_dataloaders = _create_dataloaders_single_worker\n"
            "rov.run_on_video(payload['imgs_path'], payload['masks_path'], payload['output_path'], payload['frames_with_masks'])\n"
            "print('__XMEM_OK__')\n"
        )

        env = os.environ.copy()
        env['XMEM_PAYLOAD'] = json.dumps(payload)
        try:
            result = subprocess.run(
                [sys.executable, '-c', subprocess_code],
                cwd=XMEM_PATH,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout or '').strip()
                print(f"XMem++ tracking failed: {err}")
                return {'masks': {}, 'success': False}
            if verbose and result.stdout:
                print(result.stdout.strip())
        except Exception as e:
            print(f"XMem++ tracking failed: {e}")
            return {'masks': {}, 'success': False}
        
        # Load tracked masks from the default XMem output folder.
        tracked_masks = {}
        output_mask_dir = Path(output_path) / 'masks'
        if not output_mask_dir.exists():
            if verbose:
                print(f"Warning: XMem++ masks not found at {output_mask_dir}")
            return {'masks': {}, 'success': False}
        mask_files = sorted(output_mask_dir.glob('*.png'))
        for mask_file in mask_files:
            try:
                frame_idx = int(mask_file.stem)
            except ValueError:
                continue
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # Convert from object IDs to binary mask (0/1)
            mask_binary = (mask > 0).astype(np.uint8)
            tracked_masks[frame_idx] = mask_binary
        
        if verbose:
            print(f"  Loaded {len(tracked_masks)} tracked masks")
        
        return {'masks': tracked_masks, 'success': len(tracked_masks) > 0}