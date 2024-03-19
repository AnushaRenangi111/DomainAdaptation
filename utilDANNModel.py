import torch
import utilModels
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

class DANNModel(nn.Module):
    def __init__(self, model_number, input_shape, nb_classes, batch_size, grl='auto', summary=False):
        super(DANNModel, self).__init__()
        self.learning_phase = 1
        self.model_number = model_number
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.batch_size = batch_size

        self.opt = optim.SGD(lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

        self.clsModel = getattr(utilModels, "ModelV" + str(model_number))(input_shape)

        self.dann_model, self.label_model, self.tsne_model = self.__build_dann_model()

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def compile(self):
        pass

    def __build_dann_model(self):
        branch_features = self.clsModel.get_model_features()

        # Build domain model...
        self.grl_layer = GradientReversal(1.0)
        branch_domain = self.grl_layer(branch_features)
        branch_domain = self.clsModel.get_model_domains(branch_domain)
        branch_domain = nn.Sequential(nn.Linear(branch_domain.shape[1], 2), nn.Softmax(dim=1))

        # Build label model...
        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        branch_label = branch_features[:self.batch_size // 2]

        # Build label model...
        branch_label = self.clsModel.get_model_labels(branch_label)
        branch_label = nn.Linear(branch_label.shape[1], self.nb_classes)
        branch_label = nn.Softmax(dim=1)

        dann_model = nn.Sequential(self.clsModel.input, [branch_domain, branch_label])
        label_model = nn.Sequential(self.clsModel.input, branch_label)
        tsne_model = nn.Sequential(self.clsModel.input, branch_features)

        return dann_model, label_model, tsne_model
