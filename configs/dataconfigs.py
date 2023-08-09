import ml_collections


def get_config(name):
    config = ml_collections.ConfigDict()

    bank = ml_collections.ConfigDict()
    bank.dataset = "bank"
    bank.categories = [12, 4, 8, 3, 3, 3, 2, 10, 5, 3]
    bank.numerical_features = 0
    bank.label_column = "y"
    bank.anomaly_label = "yes"

    chess = ml_collections.ConfigDict()
    chess.dataset = "chess"
    chess.categories = [4, 4, 8, 8, 8, 8]
    chess.numerical_features = 0
    chess.label_column = "class"
    chess.anomaly_label = "1"

    census = ml_collections.ConfigDict()
    census.dataset = "census"
    census.categories = [
        9,
        17,
        3,
        7,
        24,
        15,
        5,
        10,
        2,
        3,
        6,
        8,
        6,
        6,
        51,
        38,
        8,
        10,
        9,
        10,
        3,
        4,
        5,
        43,
        43,
        43,
        5,
        3,
    ]
    census.numerical_features = 5
    census.label_column = "class"
    census.anomaly_label = "50000+."

    kdd_u2r = ml_collections.ConfigDict()
    kdd_u2r.dataset = "u2r"
    kdd_u2r.categories = [3, 23, 8, 2, 2, 2]
    kdd_u2r.numerical_features = 0
    kdd_u2r.label_column = "class"
    kdd_u2r.anomaly_label = "1"

    kdd_probe = ml_collections.ConfigDict()
    kdd_probe.dataset = "probe"
    kdd_probe.categories = [3, 47, 11, 2, 2, 2]
    kdd_probe.numerical_features = 0
    kdd_probe.label_column = "class"
    kdd_probe.anomaly_label = "1"

    solar = ml_collections.ConfigDict()
    solar.dataset = "solar"
    solar.categories = [6, 4, 2, 3, 3, 2, 2, 2, 8, 6, 3]
    solar.numerical_features = 0
    solar.label_column = "class"
    solar.anomaly_label = "1"

    cmc = ml_collections.ConfigDict()
    cmc.dataset = "cmc"
    cmc.categories = [4, 4, 2, 2, 4, 4, 2, 3]
    cmc.numerical_features = 0
    cmc.label_column = "class_numberofchildren"
    cmc.anomaly_label = "1"

    celeba = ml_collections.ConfigDict()
    celeba.dataset = "celeba"
    celeba.categories = [2] * 39
    celeba.numerical_features = 0
    celeba.label_column = "class"
    celeba.anomaly_label = "1"
    
    cars = ml_collections.ConfigDict()
    cars.dataset = "cars"
    cars.categories = [4, 4, 4, 3, 3, 3]
    cars.numerical_features = 0
    cars.label_column = "unacc"
    cars.anomaly_label = "1"

    config.census = census
    config.bank = bank
    config.chess = chess
    config.probe = kdd_probe
    config.u2r = kdd_u2r
    config.solar = solar
    config.cmc = cmc
    config.celeba = celeba
    config.cars = cars

    return config[name]
