# [IEEE-JBHI 2025] Pytorch Implementation of the Paper "MedFILIP: Medical Fine-Grained Language-Image Pre-Training"

## Requirements

- python=3.10.12
- pytorch-cuda=11.7
- tensorflow=2.14.0
- transformers=4.24.0

## Code Architecture

### downstream
Contains modules for fine-tuning and inference:
- `classifi.py`: Fine-tuning for classification tasks
- `models.py`: Contrastive learning models and segmentation models
- `retrieve.py`: Zero-shot retrieval tasks
- `segment.py`: Fine-tuning for segmentation tasks

### GPT-IE
Information extraction using GPT-3.5 and related preprocessing and post-processing:
- `GPT-IE.py`: Entity extraction using GPT-3.5
- `post_process.py`: Post-processing of extracted entities
- `pre_process.py`: Preprocessing of diagnostic reports
- `run.py`: Multithreaded execution of GPT-IE

### LLaMA-IE
Information extraction using LLaMA-3-8B
- **data folder**: Houses instruction fine-tuning dataset for LLaMA-3-8B
- **inference.py**: Code for inference using the fine-tuned LLaMA-3-8B
- **instruction_generator.py**: Code for constructing instruction fine-tuning dataset
- **llama3_sft.sh**: Command-line code for LLaMA-3-8B fine-tuning
  - Configuration file: `.\LLM\ckpt\sft_args.json`
- **post_process.py**: Post-processes LLaMA-3-8B's output, converting structured disease information to JSON format

### train
Training of contrastive learning models and related configurations:
- `constants.py`: Sets of disease categories, disease severity levels, disease locations, and disease-description mapping dictionaries
- `models.py`: Contrastive learning models
- `data_GPT.json`: Entity extracted by GPT-3.5
- `data_llama3_8B.json`: Entity extracted by LLAMA-3-8B
- `train.py`: Training script for contrastive learning models

## Citation

If you use this project in your research, please consider citing it. Below is the BibTeX entry for referencing this work:

```bibtex
@article{liang2025medfilip,
  title={MedFILIP: Medical Fine-Grained Language-Image Pre-Training},
  author={Liang, Xinjie and Li, Xiangyu and Li, Fanding and Jiang, Jie and Dong, Qing and Wang, Wei and Wang, Kuanquan and Dong, Suyu and Luo, Gongning and Li, Shuo},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}