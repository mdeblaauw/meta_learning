import unittest
import torch
from typing import Dict
from few_shot.meta_learning.models.base_model import \
    BaseModel


class TestModel(BaseModel):
    """A simple one layer feed-forward model to test
    the model class.
    """
    def __init__(self, configuration: Dict):
        """Initialization of the model with parameters and
        necessary settings.

        Arguments:
            configuration {Dict} -- The config file with parameters.
        """
        super().__init__(configuration)

        self.netsimple = torch.nn.Linear(3, 1, bias=False)
        # Initialse weights to zero for test
        self.netsimple.weight = torch.nn.Parameter(torch.zeros(1, 3))
        self.netsimple = self.netsimple.to(self.device)

        self.network_names = ['simple']
        self.loss_names = ['simple']

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.netsimple.parameters(),
                                         lr=configuration['lr'])
        self.optimizers = [self.optimizer]

    def forward(self):
        """Forward through simple network.
        """
        self.output = self.netsimple(self.input)

    def compute_loss(self):
        """Compute loss with forward output.
        """
        self.loss_simple = self.mse_loss(self.output, self.label)

    def optimizer_parameters(self):
        """Perform backward pass to update weights.
        """
        self.loss_simple.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()


class TestBaseModel(unittest.TestCase):
    """Test the base model script.
    """
    def test_base_model_train(self):
        """Test the base model script with a simple model
        on a training iteration.
        """
        input_data = torch.zeros((1, 3))
        label = torch.Tensor([[0]])

        config = {
            'is_train': True,
            'checkpoint_path': 'dummy',
            'load_checkpoint': 0,
            'lr': 1,
            'lr_policy': 'dummy'
        }
        model = TestModel(config)

        model.setup()
        model.train()
        model.set_input((input_data, label))

        self.assertTrue(torch.all(model.input.eq(input_data)))
        self.assertTrue(torch.all(model.label.eq(label)))

        model.forward()

        self.assertTrue(torch.all(model.output.eq(label)))

        model.compute_loss()

        self.assertTrue(torch.all(model.loss_simple.eq(torch.Tensor([[0]]))))
        self.assertEqual(type(model.get_current_losses()['simple']), float)
        self.assertEqual(model.get_current_losses()['simple'], 0.0)

        model.optimizer_parameters()

        # check if print net runs
        model.print_networks()

    def test_base_model_lr_scheduler(self):
        """Test if the scheduler works correctly.
        """
        input_data = torch.zeros((1, 3))
        label = torch.Tensor([[0]])

        config = {
            'is_train': True,
            'checkpoint_path': 'dummy',
            'load_checkpoint': 0,
            'lr': 1,
            'lr_policy': 'step',
            'lr_decay_iters': 2
        }
        model = TestModel(config)
        model.setup()

        for i in range(2):
            model.train()
            model.set_input((input_data, label))
            model.forward()
            model.compute_loss()
            model.optimizer_parameters()
            model.update_learning_rate()
            if i == 0:
                self.assertEqual(
                    model.optimizers[0].param_groups[0]['lr'], 1.0
                )
            else:
                self.assertEqual(
                    model.optimizers[0].param_groups[0]['lr'], 0.3
                )

        def test_base_model_load_and_save(self):
            # TODO
            dummy


if __name__ == '__main__':
    unittest.main()
