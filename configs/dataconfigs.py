import ml_collections

def get_config(name):
    config = ml_collections.ConfigDict()

    bank =  ml_collections.ConfigDict()
    bank.categories = [12, 4, 8, 3, 3, 3, 2, 10, 5, 3]
    bank.cont_dims = 0
    bank.label_column = "y"
    bank.anomaly_label = "yes"

    census =  ml_collections.ConfigDict()
    census.dataset = "census"
    census.categories = [9, 16, 7, 15, 6, 5, 2, 42]
    census.cont_dims = 6

    config.census = census
    config.bank = bank
    
    return config[name]