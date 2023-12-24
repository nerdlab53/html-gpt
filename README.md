# html-gpt
# Fine tuned GPT2 540M for HTML Code generation


## About
- This is fine tuned GPT2 which generates HTML code for a given piece of prompt.
- Model chosen for fine-tuning : [GPT2 540M](https://huggingface.co/gpt2)
- Trained on retr0sushi04/html_pre_processed can be found on HuggingFace : [dataset](https://huggingface.co/datasets/retr0sushi04/html_pre_processed) which is a preprocessed version of [raw dataset](https://huggingface.co/datasets/jawerty/html_dataset).

## Contents 
- The individual Python scripts in this repo contains chunks of code individually as well as a notebook for easy implementation of fine tuning and inference.
- The dataset and the preprocessed dataset are also available as csv files.

## Requirements
- Requirements are listed in requirements.txt and are as follows :
- Installation :
- ```Python
    !pip install torch==2.1.0 transformers==4.31.0 matplotlib==3.8.2 seaborn argparse numpy datasets
  ```
- **NOTE**: Install torch with CUDA support if using on GPU for even faster training.
