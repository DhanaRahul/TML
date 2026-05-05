Membership Inference Attack
1. Requirement & Version tested 
Python  ≥ 3.10 
PyTorch ≥ 2.1 (CUDA recommended)
CUDA 11.8 / 12.x

2. Download the Dataset and Target Model
"https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
"https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
"https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"

Place `pub.pt`, `priv.pt`, and `model.pt` in the same directory as `task_template.py`.
3. Train Shadow Models (Required for LiRA)
python train_shadows.py

Hyperparameters
Architecture: ResNet-18 (modified for 32×32 input) |
Epochs 100
Batch size 128 
Num shadow models 64 

4. Run the Attack and Submit
python task_template.py
