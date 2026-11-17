# GoEmotions classification with DistilRoBERTa and LoRA

Jiarui Zha and Linxuan Li

We compared a frozen DistilRoBERTa encoder with a trained classification head against a rank-16 LoRA adapter and a trained classification head. The task is to predict any of the 28 GoEmotions labels for an English Reddit comment.

## Test results

| Model | Micro-F1 | Macro-F1 |
| --- | ---: | ---: |
| [Frozen encoder baseline](DS_project_code.ipynb) | 0.3394 | 0.1245 |
| [LoRA](LoRA_code.ipynb) | 0.5223 | 0.3811 |

These are the outputs saved in the notebooks for the GoEmotions `simplified` test split of 5,427 examples. Both notebooks use the dataset's predefined train, validation and test splits and a sigmoid threshold of 0.5. The best checkpoints are selected by validation Macro-F1. The comparison measures LoRA against a frozen encoder, and the repo contains no full fine-tuning run.

Micro-F1 pools predictions across labels. Macro-F1 averages per-label F1 with equal weight, giving rare labels the same weight as common ones.

## Train and run inference

```bash
git clone https://github.com/LinxuanLi-DS/Fine-Tuning-with-DistilRoBERTa.git
cd Fine-Tuning-with-DistilRoBERTa
python -m pip install -r requirements.txt datasets scikit-learn tqdm jupyterlab
python -m jupyterlab
```

Run [LoRA_code.ipynb](LoRA_code.ipynb) from the repository root. It downloads GoEmotions and DistilRoBERTa, trains the adapter and writes `best_LoRA_model.pt`. If training in Colab, copy that file to the repository root before running inference or building the Docker image. The checkpoint is generated locally.

```bash
python lora_inference.py --model-path best_LoRA_model.pt --device cpu --text "I finally finished the project!"
```

For the FastAPI server and Docker setup, see the [deployment guide](deployment_guide.md).

## References

- [GoEmotions dataset paper](https://aclanthology.org/2020.acl-main.372/)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
