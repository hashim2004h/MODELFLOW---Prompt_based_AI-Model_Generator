import json
from pathlib import Path

base_dir = Path(r"c:\Users\VICTUS\Downloads\Telegram Desktop\modelflow\modelflow")
metadata_path = base_dir / "models_storage" / "trained" / "models_metadata.json"

with open(metadata_path, 'r') as f:
    data = json.load(f)

# The user trained an image model with id '2d39de92-ccfe-416b-bbff-442096f20c1e'
item = data.get('2d39de92-ccfe-416b-bbff-442096f20c1e')
if item:
    item['prompt'] = 'Classify images of different dog breeds'
    item['name'] = 'Classify images of different dog breeds'

with open(metadata_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Updated prompt for latest image classification model")
