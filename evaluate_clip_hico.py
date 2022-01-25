import clip
import torch
import argparse
from pathlib import Path
from datasets.hico_dataset import HICODataset, get_clip_transforms
from evaluators.clip_evaluator import CLIPEvaluator
from metrics.hicomap import HICOmAP

parser = argparse.ArgumentParser()

parser.add_argument('--ko', default=False, type=bool,
                    help='Evaluate in known object mode')
parser.add_argument('--exclude_no_interaction', default=False, type=bool,
                    help='Exclude no interaction classes')
parser.add_argument('--train', default=False, type=bool,
                    help='Use the train set for evaluation')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size for the evaluation')
parser.add_argument('--backbone', default='ViT-B/32', type=str,
                    help='Backbone employed by CLIP')
parser.add_argument('--ir_threshold', default=0.2, type=float,
                    help='Backbone employed by CLIP')


args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

_, transforms = clip.load(args.backbone, device)

hico_dataset = HICODataset(Path(__file__).parent / 'data' / 'hico_20150920',
                           train=args.train,
                           transform=get_clip_transforms(300),
                           exclude_no_interaction=args.exclude_no_interaction)

hico_map = HICOmAP(hico_dataset=hico_dataset, known_object_mode=args.ko).to(device)

evaluator = CLIPEvaluator(device=device, dataset=hico_dataset, backbone=args.backbone,
                          metric=hico_map, batch_size=args.batch_size, ir_threshold=args.ir_threshold)
evaluator.evaluate_in_batches(32)
hico_map.export_to_json(filename='evaluation_report')

print(evaluator.final_metric.item())
