import unittest
import torch
import tempfile
import shutil
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

    def optimize_parameters(self):
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

        model.optimize_parameters()

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
            model.optimize_parameters()
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
        """Test if the load, save and setup function work.
        """
        input_data = torch.zeros((1, 3))
        label = torch.Tensor([[0]])

        temp_save_dir = tempfile.mkdtemp()

        config = {
            'is_train': True,
            'checkpoint_path': temp_save_dir,
            'load_checkpoint': 0,
            'epochs': 2,
            'lr': 1,
            'lr_policy': 'step',
            'lr_decay_iters': 2
        }
        model = TestModel(config)
        model.setup()

        # check save and load functions.
        for i in range(config['epochs']):
            model.train()
            model.set_input((input_data, label))
            model.forward()
            model.compute_loss()
            model.optimize_parameters()
            model.update_learning_rate()

        model.save_networks(epoch=2)
        model.save_optimizers(epoch=2)

        new_model = TestModel(config)
        new_model.load_networks(epoch=2)
        new_model.load_optimizers(epoch=2)

        self.assertEqual(
            new_model.optimizers[0].param_groups[0]['lr'], 0.3
        )

        # check setting lr and step count using setup() function is correct.
        config['load_checkpoint'] = 2
        new_model = TestModel(config)
        new_model.setup()
        self.assertEqual(
                new_model.optimizers[0].param_groups[0]['lr'], 0.3
            )
        self.assertEqual(new_model.schedulers[0]._step_count, 3)

        # check if step count for scheduler is correct.
        for i in range(2):
            new_model.train()
            new_model.set_input((input_data, label))
            new_model.forward()
            new_model.compute_loss()
            new_model.optimize_parameters()
            new_model.update_learning_rate()
            if i > 0:
                self.assertEqual(
                    new_model.optimizers[0].param_groups[0]['lr'], 0.09
                )

        shutil.rmtree(temp_save_dir)


if __name__ == '__main__':
    unittest.main()
