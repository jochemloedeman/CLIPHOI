import torch
import argparse
from pathlib import Path
from datasets.hico_dataset import HICODataset, get_testing_transforms
from evaluators.clip_evaluator import CLIPEvaluator
from metrics.hicomap import HICOmAP

parser = argparse.ArgumentParser()

parser.add_argument('--ko', default=False, type=bool,
                    help='Evaluate in known object mode')
parser.add_argument('--exclude_no_interaction', default=True, type=bool,
                    help='Exclude no interaction classes')
parser.add_argument('--train', default=False, type=bool,
                    help='Use the train set for evaluation')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for the evaluation')
parser.add_argument('--backbone', default='ViT-L/14', type=str,
                    help='Backbone employed by CLIP')
parser.add_argument('--prob_fn', default='softmax', type=str,
                    help='Probability function used to process logits')
parser.add_argument('--ir_threshold', default=.4, type=float,
                    help='Backbone employed by CLIP')
parser.add_argument('--resolution', default=300, type=int,
                    help='resolution of shortest side')
parser.add_argument('--normalize', default=True, type=bool,
                    help='Apply normalization during preprocessing')
parser.add_argument('--center_crop', default=False, type=bool,
                    help='Apply a center crop during preprocessing')
parser.add_argument('--five_crop', default=True, type=bool,
                    help='Apply five crop')
parser.add_argument('--save_predictions', default=False, type=bool,
                    help='Save predictions for inspection')
parser.add_argument('--include_image', default=False, type=bool,
                    help='Save predictions for inspection')


args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

testing_transforms = get_testing_transforms(resolution=args.resolution,
                                            normalize=args.normalize,
                                            center_crop=args.center_crop,
                                            five_crop=args.five_crop)

hico_dataset = HICODataset(Path(__file__).parent / 'data' / 'hico_20150920',
                           train=args.train,
                           transform=testing_transforms,
                           exclude_no_interaction=args.exclude_no_interaction,
                           save_predictions=args.save_predictions)

hico_map = HICOmAP(hico_dataset=hico_dataset, known_object_mode=args.ko).to(device)

evaluator = CLIPEvaluator(device=device, dataset=hico_dataset,
                          backbone=args.backbone, metric=hico_map,
                          batch_size=args.batch_size, prob_fn=args.prob_fn,
                          ir_threshold=args.ir_threshold, five_crop=args.five_crop,
                          center_crop=args.center_crop, save_predictions=args.save_predictions,
                          include_image=args.include_image)
evaluator.evaluate()

print(evaluator.final_metric.item())
hico_map.export_to_json('evaluation.report')
