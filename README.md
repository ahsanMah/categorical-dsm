# Denoising Score Matching for Categorical Data

This is the official repo for the work [Anomaly Detection via Gumbel Noise Score Matching](https://www.frontiersin.org/articles/10.3389/frai.2024.1441205). If you use this repository for research purposes, please cite our work using the citation at the end of this document.

## Training on a new dataset

This section explains how to train the model on a new dataset. You'll need to modify two main files: `dataconfigs.py` and `dataloader.py`. Follow these steps to add your new dataset:

### Step 1: Update `dataconfigs.py`

1. Add a new `ConfigDict` for your dataset. Use the existing configs as a template.
2. Fill in the following fields:
   - `dataset`: The name of your dataset (string)
   - `categories`: A list of integers representing the number of classes for each categorical feature
   - `numerical_features`: The number of numerical features in your dataset (integer)
   - `label_column`: The name of the column containing the class labels (string)
   - `anomaly_label`: The label used for anomalies in your dataset (string)

Example:

```python
new_dataset = ml_collections.ConfigDict()
new_dataset.dataset = "new_dataset"
new_dataset.categories = [3, 4, 2, 5]  # Example: 4 categorical features with 3, 4, 2, and 5 classes respectively
new_dataset.numerical_features = 2
new_dataset.label_column = "class"
new_dataset.anomaly_label = "anomaly"

# Add your new dataset to the main config
config.new_dataset = new_dataset
```

### Step 2: Update `dataloader.py`

1. Add your dataset to the `tabular_datasets` dictionary:

```python
tabular_datasets = {
    # ... existing datasets ...
    "new_dataset": "new_dataset.csv",
}
```

2 (optional). If your dataset requires special handling, modify the `load_dataset` function. Look for the section that handles specific datasets (e.g., "cars", "mushrooms", "nursery") and add your logic there.

Example:

```python
if name == "new_dataset":
    df = pd.read_csv(f"data/{tabular_datasets[name]}")
    # Add any necessary preprocessing steps here
    # For example:
    # df = df.drop(columns=['unnecessary_column'])
    # df[label_name] = df[label_name].map({'normal': '0', 'anomaly': '1'})
```

### Step 3: Prepare the dataset file

1. Ensure your dataset file (e.g., "new_dataset.csv") is in the correct format (CSV for new datasets).
2. Place the file in the appropriate directory (usually the `data/` folder).
3. If your dataset requires any preprocessing, consider doing it beforehand to simplify the loading process.

### Step 4: Run the training

To train the model with your new dataset, use the following command:

```bash
python main.py --config=configs/your_config.py --mode=train --workdir=/path/to/your/workdir
```

> **Make sure to update `your_config.py` with `config.data = get_data_config("new_dataset")` and include the necessary parameters for your new dataset.**

### Additional Notes

- Ensure that the data types in your CSV file match the expectations of the loader. Categorical data should be strings, and numerical data should be integers or floats.
- If your dataset has a different file format (e.g., ARFF), you may need to modify the `load_dataset` function in `dataloader.py` to handle it correctly.
- Always test your changes with a small subset of your data before running a full training session.

## Model Training Configuration

This section explains the purpose and structure of the model training configuration files, and how to create a new configuration for your experiments.

### Purpose of Configuration Files

Configuration files in this project serve several important purposes:
1. They centralize all hyperparameters and settings for an experiment.
2. They allow for easy reproducibility of experiments.
3. They facilitate hyperparameter tuning and ablation studies.
4. They provide a clear overview of the experimental setup.

### Structure of Configuration Files

The configuration system is built using `ml_collections.ConfigDict`, which allows for nested configurations and easy access to parameters.

There are two main types of configuration files:

1. **Base Configuration** (`base_config.py`): This file contains default settings for *all* experiments.
2. **Dataset-specific Configuration** (e.g., `cars_config.py`): These files inherit from the base configuration and specify settings for a particular dataset or experiment.

### Key Configuration Sections

#### Training

The `training` section includes parameters such as:
- `batch_size`: Size of training batches
- `n_steps`: Total number of training steps
// freq is measured in number of trianing steps //
- `log_freq`: Frequency of logging metrics (to tensorboard or wandb)
- `eval_freq`: Frequency of evaluation
- `checkpoint_freq`: Frequency of saving checkpoints
- `snapshot_freq`: Frequency of saving snapshots
- `resume`: Whether to resume training from a checkpoint

#### Data

The `data` section is typically loaded from a separate data configuration file (e.g., `dataconfigs.py`) and includes dataset-specific information. See previous section for more information.

#### Model

The `model` section specifies the architecture and hyperparameters of the model, including:
- `name`: Model architecture (e.g., "tab-transformer", "tab-resnet")
- `ndims`: Number of dimensions in the model
- `layers`: Number of layers
- `dropout`: Dropout rate
- `attention_heads` and `attention_dim_head`: For transformer-based models
- Other architecture-specific parameters

#### Optimization

The `optim` section includes optimization-related settings such as:
- `optimizer`: Optimizer type (e.g., "AdamW")
- `lr`: Learning rate
- `weight_decay`: L2 regularization strength
- `beta1` and `beta2`: Adam optimizer parameters
- `grad_clip`: Gradient clipping value
- `scheduler`: Learning rate scheduler type

#### Evaluation

The `eval` section specifies evaluation-related parameters, such as the batch size for evaluation.

#### Hyperparameter Sweeps

The `sweep` section defines the configuration for hyperparameter tuning, including:
- Parameters to sweep over
- Sweep method (e.g., "bayes" for Bayesian optimization)
- Metric to optimize
- Early termination criteria

This project maskes use of the Weights and Biases (`wandb`) library to perform and log hyperparameter sweeps.

### Creating a New Configuration File

To create a new configuration file for your experiment:

1. Create a new Python file (e.g., `my_experiment_config.py`).
2. Import the necessary modules and the base configuration:

   ```python
   import ml_collections
   from configs.base_config import get_config as get_base_config
   from configs.dataconfigs import get_config as get_data_config
   ```

3. Define a `get_config()` function that returns your custom configuration:

   ```python
   def get_config():
       config = get_base_config()
       
       # Modify or add configuration parameters
       config.training.batch_size = 256
       config.training.n_steps = 1000000
       
       # Set the data configuration
       # The dataconfigs.py file should have already been updated
       # by following information in the previous section
       config.data = get_data_config("my_dataset")
       
       # Modify model configuration
       config.model.name = "my_custom_model"
       config.model.ndims = 512
       config.model.layers = 8
       
       # Modify optimization configuration
       config.optim.lr = 1e-4
       config.optim.weight_decay = 1e-5
       
       return config
   ```

4. Customize the configuration as needed for your experiment, overriding default values from the base configuration.

### Using the Configuration in the Training Process

The configuration is used in the `runner.py` file to set up the training process. The main steps are:

1. The configuration is loaded using the command-line argument `--config`.
2. The appropriate model is instantiated based on the `config.model.name`.
3. The dataset is loaded using parameters from `config.data`.
4. The optimizer and learning rate scheduler are set up using `config.optim`.
5. The training loop uses various parameters from the configuration, such as `config.training.n_steps` and `config.training.eval_freq`.

To run training with your new configuration:

```bash
python main.py --config=configs/my_experiment_config.py --mode=train --workdir=/path/to/your/workdir
```

### Tips for Experimenting with Configurations

1. Start with a copy of an existing configuration file and modify it for your needs.
2. Use the `sweep` section to define hyperparameter searches for your experiment.
3. Keep track of different configurations by using clear, descriptive filenames.
4. Comment your configuration files, especially when using non-standard settings.
5. When running experiments, save the configuration along with the results for future reference.
6. Use the `devtest` flag in the configuration for quick testing of your setup before running full experiments.

Remember that changes to the configuration structure might require corresponding updates in the `runner.py` file to ensure all new parameters are properly utilized during training.

# Citing
```
@article{10.3389/frai.2024.1441205,
   AUTHOR={Mahmood, Ahsan  and Oliva, Junier  and Styner, Martin A.},
   TITLE={Anomaly Detection via Gumbel Noise Score Matching},
   JOURNAL={Frontiers in Artificial Intelligence},
   VOLUME={7},
   YEAR={2024},
   URL={https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1441205},
   DOI={10.3389/frai.2024.1441205},
   ISSN={2624-8212},
}
```
