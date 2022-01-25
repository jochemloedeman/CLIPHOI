from pathlib import Path

from matplotlib import pyplot as plt

from datasets.hico_dataset import HICODataset

hico_training_set = HICODataset(Path(__file__).parent / 'data' / 'hico_20150920',
                                train=True,
                                exclude_no_interaction=False)

dimensions = []
aspect_ratios = []
for index in range(len(hico_training_set)):
    image, label = hico_training_set[index]
    dimensions.append((image.shape[1], image.shape[2]))
    aspect_ratios.append(image.shape[2] / image.shape[1])

plt.hist(aspect_ratios, bins=60)
plt.show()
