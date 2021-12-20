import clip
import torch
import argparse
from pathlib import Path
from hico_dataset import HICODataset
from clip_evaluator import CLIPEvaluator
from hicomap import HICOmAP

parser = argparse.ArgumentParser()

parser.add_argument('--ko', default=False, type=bool,
                    help='Evaluate in known object mode')
parser.add_argument('--exclude_no_interaction', default=True, type=bool,
                    help='Exclude no interaction classes')
parser.add_argument('--train', default=False, type=bool,
                    help='Use the train set for evaluation')
parser.add_argument('--batch_size', default=512, type=int,
                    help='Batch size for the evaluation')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

hico_dataset = HICODataset(Path(__file__).parent / 'hico_20150920',
                           train=args.train,
                           transform=preprocess,
                           exclude_no_interaction=args.exclude_no_interaction)

hico_map = HICOmAP(hico_dataset=hico_dataset, known_object_mode=args.ko).to(device)

evaluator = CLIPEvaluator(device=device, dataset=hico_dataset,
                          metric=hico_map, batch_size=args.batch_size)
evaluator.evaluate()
hico_map.export_to_json(filename='evaluation_report')

print(evaluator.final_metric.item())
