import torch
from PIL import Image
import numpy as np


class ModelWrapper:
    def __init__(self, model, predict_proba=False):
        self.model = model
        self.predict_proba = predict_proba

    def predict(self, X):
        if self.predict_proba:
            return self._predict_proba(X)
        else:
            return self._predict(X)

    def _predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise NotImplementedError("predict_proba is not available for this model")

    def _predict(self, X):
        if hasattr(self.model, 'predict'):
            pred = self.model.predict(X)
        elif isinstance(self.model, torch.nn.Module):
            X = torch.as_tensor(X, dtype=torch.float32)
            with torch.no_grad():
                pred = self.model(X)
        else:
            raise TypeError("Unsupported model type")

        return self._to_numpy(pred)


    @staticmethod
    def _to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}")


class ImageModelWrapper(ModelWrapper):
    def __init__(self, model, preprocessor=None):
        super().__init__(model)
        self.preprocessor = preprocessor

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if X.dim() == 3:
            X = X.unsqueeze(0)  # Add batch dimension if missing
        if self.preprocessor:
            X = self.preprocessor(X)
        with torch.no_grad():
            pred = self.model(X)
        return self._to_numpy(pred.squeeze(0))


class BinaryOutputModelWrapper(ModelWrapper):
    def predict(self, X):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(X)
            _, predicted = torch.max(pred, 1)
        return self._to_numpy(predicted)