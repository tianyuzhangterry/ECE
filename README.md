# ECE
Explainable and Efficient Editing 

- Code for [Explainable and Efficient Editing for Large Language Models] 

- Large Language Models (LLMs) possess remarkable capabilities in storing and retrieving vast factual knowledge but often retain outdated or incorrect information from web corpora. While full
retraining is costly, locate-and-edit model editing methods offer an feasible alternative. Current methods typically follow a two-stage paradigm: (1) identifying critical layers for knowledge storage and (2) updating their parameters to store new knowledge. However, both of these two phases have their inherent limitations. In stage 1, layers identification is independent of the to-be-updated knowledge, ignoring the varying storage patterns of different knowledge types. Meanwhile, Stage 2 suffers from high computational overhead due to independent gradient descent for each piece of knowledge. To solve these, we propose an Explainable and effiCient model Editing method, termed ECE. Specifically, in Stage 1, ECE integrates the concept of LLMs explainability into the editing process, enabling the adaptive identification of the crucial neurons based on the input knowledge. In Stage 2, ECE clusters similar knowledge based on the explanation results, allowing batch optimization in a single gradient step, significantly reducing time consumption without sacrificing effectiveness. Extensive experiments demonstrate that ECE can achieve superior performance while delivering a 3.27× speedup in editing efficiency, showcasing the potential of explainability-driven editing methods for LLMs.

# ![alt text](resource/)
# *Figure: This is the overall architecture of our ECE method.*

## Requirements
**At least one A40 48G GPU.**

- pytorch==1.12.1
- einops==0.4.0
- higher==0.2.1
- hydra-core==1.2.0
- transformers==4.23.1
- datasets==1.18.3
- matplotlib==3.6.1
- spacy==3.4.1
- scipy==1.9.2
- scikit-learn==1.0.2
- nltk==3.7

## Quick Start
### An example for editing GPT2-XL on counterfact dataset using ECE_WI
#### 1. Edit Llama3 (8B) model 
 
    python experiments/evaluate.py --alg_name ECE_WI --model_name gpt2-xl --hparams_fname gpt2-xl.json --ds_name=mcf --dataset_size_limit=2000 --num_edits=100 --downstream_eval_steps=5 

This command runs an evaluation script for the NSE algorithm using the Llama3-8b-instruct. Below are the explanations for each argument:

- `--alg_name=ECE_WI`: Specifies the name of the algorithm being used, which is ECE_WI in this case.
- `--model_name= gpt2-xl`: Indicates the name of the model being evaluated, here it is  gpt2-xl.
- `--hparams_fname=gpt2-xl.json`: Points to the JSON file containing hyperparameters specific to the  gpt2-xl model.
- `--ds_name=mcf`: Specifies the dataset name, in this case, "mcf".
- `--dataset_size_limit=2000`: Sets the total number of editing samples to 2000.
- `--num_edits=100`: Defines the batch size for each round of editing, meaning 100 edits will be performed in each batch. 
- `--downstream_eval_steps=5`: indicates that a test of general capabilities is conducted after every 5 rounds of editing.
#### 2. Summarize the results

    python summarize.py --dir_name=ECE_WI --runs=run_<run1>,run_<run2>

## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git).
