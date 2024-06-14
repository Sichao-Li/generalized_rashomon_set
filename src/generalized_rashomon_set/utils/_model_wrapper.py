import torch
from PIL import Image


class model_wrapper:
    def __init__(
            self,
            model,
            torch_input
    ):
        self.model = model
        self.torch_input = torch_input
        # self.image_input = image_input
        # self.preprocessor = preprocessor
        # self.binary_output = binary_output

    def predict(self, X):
        if self.torch_input:
            X = torch.tensor(X).float()
            if hasattr(self.model, 'predict'):
                return self.model.predict(X).detach().numpy()
            # elif self.image_input:
            #     if hasattr(X, 'shape'):
            #         X = Image.fromarray(X)
            #     X = self.preprocessor(X)
            #     return self.model(X).squeeze(0).image_input(0)
            # elif self.binary_output:
            #     pred = self.model(X)
            #     _, predicted = pred.max(1)
            #     return predicted.detach().numpy()
            else:
                return self.model(X).detach().numpy()
        else:
            if hasattr(self.model, 'predict'):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)[:, 1]
                else:
                    return self.model.predict(X)
            else:
                return self.model(X)

    # def predict(self, X):
    #     if self.torch_input:
    #         if hasattr(self.model, 'predict'):
    #             X = torch.tensor(X).float()
    #             return self.model.predict(X).detach().numpy()
    #         else:
    #             if self.image_input:
    #                 if hasattr(X, 'shape'):
    #                     X = Image.fromarray(X)
    #                 X = self.preprocessor(X)
    #                 return self.model(X).squeeze(0).image_input(0)
    #             elif self.binary_output:
    #                 X = torch.tensor(X).float()
    #                 pred = self.model(X)
    #                 _, predicted = pred.max(1)
    #                 return predicted.detach().numpy()
    #             else:
    #                 X = torch.tensor(X).float()
    #                 pred = self.model(X)
    #                 return pred.detach().numpy()
    #     else:
    #         X = X
    #         if hasattr(self.model, 'predict'):
    #             if not hasattr(self.model, 'predict_proba'):
    #                 return self.model.predict(X)
    #             else:
    #                 return self.model.predict_proba(X)[:, 1]
    #         else:
    #             return self.model(X)


class model_wrapper_image(model_wrapper):
    def __init__(
            self,
            model,
            torch_input,
            preprocessor=None
    ):
        super().__init__(model, torch_input)
        self.preprocessor = preprocessor

    def predict(self, X):
        X = torch.tensor(X).float()
        if hasattr(X, 'shape'):
            X = Image.fromarray(X)
        X = self.preprocessor(X)
        return self.model(X).squeeze(0).image_input(0)


class model_wrapper_binary_output(model_wrapper):
    def __init__(
            self,
            model,
            torch_input
    ):
        super().__init__(model, torch_input)

    def predict(self, X):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        X = torch.tensor(X).float()
        pred = self.model(X)
        _, predicted = pred.max(1)
        return predicted.detach().numpy()


    # def predict(self, X):
    #     if self.torch_input:
    #         if hasattr(self.model, 'predict'):
    #             X = torch.tensor(X).float()
    #             return self.model.predict(X).detach().numpy()
    #         else:
    #             if self.image_input:
    #                 if hasattr(X, 'shape'):
    #                     X = Image.fromarray(X)
    #                 X = self.preprocessor(X)
    #                 return self.model(X).squeeze(0).image_input(0)
    #             elif self.binary_output:
    #                 X = torch.tensor(X).float()
    #                 pred = self.model(X)
    #                 _, predicted = pred.max(1)
    #                 return predicted.detach().numpy()
    #             else:
    #                 X = torch.tensor(X).float()
    #                 pred = self.model(X)
    #                 return pred.detach().numpy()
    #     else:
    #         X = X
    #         if hasattr(self.model, 'predict'):
    #             if not hasattr(self.model, 'predict_proba'):
    #                 return self.model.predict(X)
    #             else:
    #                 return self.model.predict_proba(X)[:, 1]
    #         else:
    #             return self.model(X)