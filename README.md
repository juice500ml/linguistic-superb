## Set up environment
```sh
conda create -p ./envs python=3.10
conda activate ./envs
pip install -r requirements.txt
```

## Push tasks
```sh
HF_TOKEN=YOUR_HF_TOKEN python3 TASK_NAME.py
```

## How to use utils
```python
from utils import validate_and_push_dataset

rows = [
    "audio": ["/path/to/audio1.wav", "/path/to/audio2.wav", "/path/to/audio3.wav", ],
    "instruction": ["inst1", "inst2", "inst3"],
    "label": ["l1", "l2", "l3"],
]
validate_and_push_dataset("your/repo", rows)
```
