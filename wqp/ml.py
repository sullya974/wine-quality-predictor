from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

def build_wine_predictor_model() -> Pipeline:

    model: Pipeline = Pipeline(steps=[
        ('regression', ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False))
    ])
    return model

