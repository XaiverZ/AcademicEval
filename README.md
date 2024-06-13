# AcademicEval: Live Long-Context LLM Benchmark


![Method](./model.png)


**You can download our collected data at [AcademicEval-HF](https://huggingface.co/datasets/AcademicEval/AcademicEval)**


## Environment Setup

### Python Package

```bash
# python==3.10
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install arxiv
pip install tqdm
pip install rouge_score
pip install textstat
pip install transformers
pip install langchain
pip install PyMuPDF
pip install faiss-gpu
pip install openai==0.28.0
```

### Model Weights

The following is the model weight download address needed in the experiment:
- [Contriever](https://huggingface.co/facebook/contriever)
- [BERT-Base-Uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [RoBERTa](https://huggingface.co/FacebookAI/roberta-large)
- [BART](https://huggingface.co/facebook/bart-large-cnn)
- [DeBERTa](https://huggingface.co/microsoft/deberta-xlarge-mnli)

You can download them as needed to conduct corresponding experiments.


### LLM Tokenizers

We additionally need the tokenizer configuration files for LLMs to ensure correct and accurate truncation.
- [Gemma](https://huggingface.co/google/gemma-7b-it)
- [LLaMA](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [Qwen](https://huggingface.co/Qwen/Qwen1.5-72B-Chat)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [Nous Hermes](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)

You only need to download the tokenizer configuration files for each LLM, no model weight files are needed, because we access LLMs through the API.




## Experiments

**Note: Since we use the LLM API provided by [together.ai](https://www.together.ai/) to access LLMs, you need to prepare your own API KEY.**

Some script examples are shown below:


```bash
# title-10K
python exp_comparison.py --setting title_short --llm_model google/gemma-7b-it --cuda 3
# title-30K
python exp_comparison.py --setting title_long --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
# title-31K-G
python exp_comparison.py --setting title_long_graph --llm_model Qwen/Qwen1.5-72B-Chat --cuda 3
```





## Benchmark Construction


**Coming Soon**

