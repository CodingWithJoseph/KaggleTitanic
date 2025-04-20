import dill as pickle


def save_model(model, name, trained_features=None):
    bundle = {'model': model, 'features': trained_features}
    try:
        with open(f'../models/{name}', "wb") as f:
            pickle.dump(bundle, f)
    except Exception as e:
        print(f"Error saving model '{name}': {e}")


def open_model(name):
    try:
        with open(f'../models/{name}', "rb") as f:
            bundle = pickle.load(f)
            return bundle.get('model'), bundle.get('features')

    except Exception as e:
        print(f"Error loading model '{name}': {e}")
    return None, None
