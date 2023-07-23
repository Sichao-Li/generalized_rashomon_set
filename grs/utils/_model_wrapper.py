import torch
from PIL import Image
class model_wrapper:
    def __init__(
            self,
            model,
            wrapper_for_torch,
            softmax,
            preprocessor=None,
    ):
        self.model = model
        self.wrapper_for_torch = wrapper_for_torch
        self.softmax = softmax
        self.preprocessor = preprocessor

    def predict(self, X):
        if self.wrapper_for_torch:
            if hasattr(self.model, 'predict'):
                X = torch.tensor(X).float()
                return self.model.predict(X).detach().numpy()
            else:
                if self.softmax:
                    if hasattr(X, 'shape'):
                        X = Image.fromarray(X)
                    X = self.preprocessor(X)
                    return self.model(X).squeeze(0).softmax(0)

                else:
                    X = torch.tensor(X).float()
                    return self.model(X).detach().numpy()

        else:
            X = X
            if hasattr(self.model, 'predict'):
                if not hasattr(self.model, 'predict_proba'):
                    return self.model.predict(X)
                else:
                    return self.model.predict_proba(X)[:, 1]
            else:
                return self.model(X)