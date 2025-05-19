import unittest
from unittest.mock import mock_open, patch, MagicMock
import json
from torch.utils import data
from torch.utils.data import Dataset
from custom_agent.agent_dataset import AgentDataset, DataCollatorWithPadding


class TestAgentDataset(unittest.TestCase):

    @patch("builtins.open")
    def setUp(self, mock_open_func):
        """
        设置测试环境。
        - 模拟 tokenizer
        - 模拟文件打开和读取
        - 构造一个简单的 AgentDataset 实例
        """
        self.tokenizer = MagicMock()
        self.data_path = "dummy.json"
        self.max_length = 128

        mock_file = mock_open()
        mock_file_content = '{"question": "What studies used Variational Autoencoders (VAEs) in training generative deep learning models on large-scale datasets?", "answer": ["Expressive Body Capture: 3D Hands, Face, and Body from a Single Image", "HuMoR: 3D Human Motion Model for Robust Pose Estimation", "Learning Motion Priors for 4D Human Body Capture in 3D Scenes", "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE"], "answer_arxiv_id": ["1904.05866", "2105.04668", "2108.10399", "2104.05670"], "source_meta": {"published_time": "20240305"}, "qid": "AutoScholarQuery_train_33550"}'
        mock_file.return_value.readlines.return_value = [mock_file_content]
        mock_open_func.return_value = mock_file.return_value

        self.dataset = AgentDataset(self.data_path, self.tokenizer)

    def test_init_success(self):
        """
        TC01: 测试初始化成功加载数据
        """
        self.assertEqual(len(self.dataset.messages), 1)


if __name__ == "__main__":
    unittest.main()
