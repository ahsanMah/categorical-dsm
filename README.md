# categorical-dsm
Denoising Score Matching for Categorical Data

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
