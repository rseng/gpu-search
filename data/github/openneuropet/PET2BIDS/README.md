# https://github.com/openneuropet/PET2BIDS

```console
metadata/schema.json:      "LabelingPulseAverageB1": {
metadata/schema.json:        "name": "LabelingPulseAverageB1",
metadata/schema.json:        "description": "The average B1-field strength of the RF labeling pulses, in microteslas.\nAs an alternative, `\"LabelingPulseFlipAngle\"` can be provided.\n",
metadata/schema.json:      "LabelingPulseAverageGradient": {
metadata/schema.json:        "name": "LabelingPulseAverageGradient",
metadata/schema.json:      "LabelingPulseDuration": {
metadata/schema.json:        "name": "LabelingPulseDuration",
metadata/schema.json:      "LabelingPulseFlipAngle": {
metadata/schema.json:        "name": "LabelingPulseFlipAngle",
metadata/schema.json:        "description": "The flip angle of a single labeling pulse, in degrees,\nwhich can be given as an alternative to `\"LabelingPulseAverageB1\"`.\n",
metadata/schema.json:      "LabelingPulseInterval": {
metadata/schema.json:        "name": "LabelingPulseInterval",
metadata/schema.json:      "LabelingPulseMaximumGradient": {
metadata/schema.json:        "name": "LabelingPulseMaximumGradient",
metadata/schema.json:            "LabelingPulseAverageGradient": "recommended",
metadata/schema.json:            "LabelingPulseMaximumGradient": "recommended",
metadata/schema.json:            "LabelingPulseAverageB1": "recommended",
metadata/schema.json:            "LabelingPulseDuration": "recommended",
metadata/schema.json:            "LabelingPulseFlipAngle": "recommended",
metadata/schema.json:            "LabelingPulseInterval": "recommended"

```
