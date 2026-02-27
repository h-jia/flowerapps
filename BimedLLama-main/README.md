# FlowerTune LLM on Medical Dataset (Bio-Medical-Llama)

Federated instruction tuning with a pretrained [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards). This app uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition, and preprocess the dataset.

## Quickstart

```
flwr new @heart-ai-lab/flowertune-med-bimedllama
```

## Fetch the App

Install Flower:

```
pip install flwr
```

Fetch the app:

```
flwr new @heart-ai-lab/flowertune-med-bimedllama
```

This will create a new directory with the following structure:

```
flowertune-med-bimedllama/
├── flowertune_med
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── dataset.py      # Data loading for Simulation and Deployment
│   ├── models.py       # Model definition
│   └── strategy.py     # FedProx strategy
├── flowertune-eval-medical/
│   └── ...             # Evaluation scripts and PEFT adapters
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

## Run the App

You can run your Flower App in both simulation and deployment mode without making changes to the code. If you are starting with Flower, we recommend using the simulation mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

**Tip:** Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations, how to use more virtual SuperNodes, and how to configure CPU/GPU usage in your ClientApp.

Install the dependencies defined in pyproject.toml as well as the flowertune_med package:

```
cd flowertune-med-bimedllama && pip install -e .
```

Run with default settings:

```
flwr run .
```

You can also override some of the settings for your ClientApp and ServerApp defined in pyproject.toml. For example:

```
flwr run . --run-config "num-server-rounds=5"
```

### Run with the Deployment Engine

To run this App using Flower's Deployment Engine we recommend first creating some demo data using [Flower Datasets](https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html). For example:

```
# Install Flower datasets
pip install flwr-datasets

# Create dataset partitions and save them to disk
flwr-datasets create medalpaca/medical_meadow_medical_flashcards --num-partitions 2 --out-dir demo_data
```

The above command will create two IID partitions of the Medical Meadow Medical Flashcards dataset and save them in a `demo_data` directory. Next, you can pass one partition to each of your SuperNodes like this:

```
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/demo_data/partition_0"
```

Finally, ensure the environment of each SuperNode has all dependencies installed. Then, launch the run via `flwr run` but pointing to a SuperLink connection that specifies the SuperLink your SuperNode is connected to:

```
flwr run . <SUPERLINK-CONNECTION> --stream
```

**Tip:** Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine.

## Methodology

This experiment performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library. The clients' models are aggregated with FedProx strategy.

### PEFT Adapter

The fine-tuning results have been submitted as a PEFT adapter and can be accessed here:

- [Bio-Medical-Llama-3-8B-PEFT-Adapter](https://github.com/h-jia/BimedLLama/tree/main/flowertune-eval-medical/peft-adapter)

### Bio-Medical-Llama-3-8B Configuration

- **Precision**: bf16 for model weights, tf32 for gradients and optimizer states.
- **Quantization**: 4-bit quantization for reduced memory usage.
- **Optimizer**: Paged AdamW 8-bit for effective optimization under constrained resources.
- **LoRA Configuration**: Rank (r): 8, Alpha: 32
- **Training Configuration**: Batch size: 16, Max steps: 6, Warmup steps: 2, Total rounds: 10, Fraction fit per round: 0.15
- **Learning Rate Scheduler**: Cosine annealing (1e-5 max, 1e-6 min)
- **Strategy**: FedProx

### Evaluation Results

Hardware: NVIDIA A100 (1x GPU)

- **pubmedqa**: 0.6580
- **medqa**: 0.6031
- **medmcqa**: 0.6834
- **careqa**: 0.5367
- **average**: 0.6203

### Model saving

The global PEFT model checkpoints are saved every 2 rounds after aggregation on the server side as default, which can be specified with `train.save-every-round` under `[tool.flwr.app.config]` entry in pyproject.toml.

> **Note:** Please provide the last PEFT checkpoint if you plan to participate in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

## Acknowledgements

This code is based on the following repositories:
- [FlowerTune-Qwen2.5-7B-Instruct-Medical](https://github.com/ethicalabs-ai/FlowerTune-Qwen2.5-7B-Instruct-Medical)
- [GCCL-Medical-LLM2-FlowerTune](https://github.com/gachon-CCLab/GCCL-Medical-LLM2-FlowerTune)

We thank the authors for their valuable contributions to the medical LLM fine-tuning community.
