import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datasets import Dataset

# Ensure the project root is in the sys.path so we can import gsm
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gsm.datasets.datasets import GSM8KDataset

from gsm.datasets.datasets import (
    GSM8KDataset,
    create_math_dataset,
    format_math_dataset,
    create_sft_dataset,
    create_rl_dataset
)



class MockTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # A simple simulated chat template output
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

class TestGSM8KDataset(unittest.TestCase):
    def test_sft_format(self):
        dataset = GSM8KDataset(split="train", max_samples=2, format_type="sft")
        
        self.assertEqual(len(dataset), 2)
        
        # Test __getitem__
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("completion", sample)
        self.assertIn("text", sample)
        
        # Ensure required prompt structure exists
        self.assertIn("Question:", sample["prompt"])
        self.assertIn("Let's solve this step by step:", sample["prompt"])

    def test_rl_format_without_tokenizer(self):
        dataset = GSM8KDataset(split="train", max_samples=2, format_type="rl")
        
        self.assertEqual(len(dataset), 2)
        
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("ground_truth", sample)
        self.assertIn("question", sample)
        self.assertIn("full_answer", sample)
        
        # Since tokenizer=None, prompt should be the raw content
        self.assertIn("Question:", sample["prompt"])

    def test_create_math_dataset_sft(self):
        # SFT format
        ds_sft = create_math_dataset(dataset_name="gsm8k", split="test", max_samples=2, format_type="sft")
        self.assertTrue(isinstance(ds_sft, Dataset))
        self.assertIn("prompt", ds_sft.features)
        
    @patch('gsm.datasets.datasets.AutoTokenizer.from_pretrained')
    def test_format_math_dataset(self, mock_from_pretrained):
        # Set up mock tokenizer
        mock_from_pretrained.return_value = MockTokenizer()
        
        # Create a mock dataset
        mock_data = {
            "question": ["What is 2+2?", "If I have 3 apples..."],
            "answer": ["2+2 is 4.\n#### 4", "3-1=2.\n#### 2"]
        }
        raw_dataset = Dataset.from_dict(mock_data)
        
        # Test SFT format
        sft_ds = format_math_dataset(raw_dataset, format_type="sft", model_name="dummy_model")
        self.assertEqual(len(sft_ds), 2)
        self.assertEqual(sft_ds[0]["prompt"], "Question: What is 2+2?\n\nLet's solve this step by step:\n")
        self.assertEqual(sft_ds[0]["completion"], "2+2 is 4.\n\nFinal Answer: 4")
        
        # Test RL format
        rl_ds = format_math_dataset(raw_dataset, format_type="rl", model_name="dummy_model")
        self.assertEqual(len(rl_ds), 2)
        self.assertEqual(rl_ds[0]["ground_truth"], "4")
        self.assertIn("<|im_start|>user", rl_ds[0]["prompt"])
        
    def test_create_sft_dataset_helper(self):
        ds = create_sft_dataset(max_samples=2, split="train")
        self.assertEqual(len(ds), 2)
        
    @patch('gsm.datasets.datasets.AutoTokenizer.from_pretrained')
    def test_create_rl_dataset_helper(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MockTokenizer()
        ds = create_rl_dataset(max_samples=2, split="train", model_name="dummy_model")
        self.assertEqual(len(ds), 2)

if __name__ == '__main__':
    unittest.main()
    # from datasets import load_dataset

    # dataset = load_dataset("json", data_dir="data/GSM8K_zh", split="test")
    # print(dataset[0])


    
    

