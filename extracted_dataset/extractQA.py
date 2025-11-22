import json
import logging
import os
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Process datasets and extract questions/answers in SFT format."""
    
    def __init__(self, output_dir: str):
        """Initialize the processor with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_shp_dataset(self) -> Dict:
        """Load Stanford Human Preferences dataset."""
        logger.info("Loading Stanford Human Preferences (SHP) dataset")
        try:
            dataset = load_dataset("stanfordnlp/SHP")
            logger.info(f"Successfully loaded SHP dataset")
            logger.info(f"Train: {len(dataset['train'])} samples")
            logger.info(f"Validation: {len(dataset['validation'])} samples")
            logger.info(f"Test: {len(dataset['test'])} samples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading SHP dataset: {str(e)}")
            raise
    
    def load_openmath_dataset(self, cache_dir: Optional[str] = None) -> Dict:
        """Load OpenMathInstruct-1 dataset."""
        logger.info("Loading OpenMathInstruct-1 dataset")
        try:
            dataset = load_dataset("nvidia/OpenMathInstruct-1", cache_dir=cache_dir)
            logger.info(f"Successfully loaded OpenMathInstruct-1 dataset")
            
            # Create validation split if it doesn't exist
            if "validation" not in dataset:
                logger.info("Creating validation split (98% train, 2% validation)")
                split_dataset = dataset["train"].train_test_split(test_size=0.02, shuffle=True, seed=42)
                dataset = {
                    "train": split_dataset["train"],
                    "validation": split_dataset["test"],
                    "test": split_dataset["test"]  # Using validation as test for consistency
                }
            
            logger.info(f"Train: {len(dataset['train'])} samples")
            logger.info(f"Validation: {len(dataset['validation'])} samples")
            logger.info(f"Test: {len(dataset.get('test', []))} samples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading OpenMathInstruct-1 dataset: {str(e)}")
            raise
    
    def format_shp_data(self, example: Dict) -> List[Tuple[str, str]]:
        """Format SHP examples and extract question-answer pairs using the correct fields."""
        qa_pairs = []
        
        for i in range(len(example['history'])):
            question = example['history'][i]
            # Choose the preferred answer based on the label
            if example['labels'][i] == 1:
                # A is preferred
                answer = example['human_ref_A'][i]
            else:
                # B is preferred
                answer = example['human_ref_B'][i]
            
            # Store the question and preferred answer
            qa_pairs.append((question, answer))
            
        return qa_pairs
    
    def format_openmath_data(self, example: Dict) -> List[Tuple[str, str]]:
        """Format OpenMathInstruct examples and extract question-answer pairs."""
        qa_pairs = []
        
        for i in range(len(example['question'])):
            question = example['question'][i]
            answer = example['generated_solution'][i] if example['generated_solution'][i] is not None else ""
            
            # Store the original question and answer
            qa_pairs.append((question, answer))
            
        return qa_pairs
    
    def format_sft_shp_question(self, question: str) -> str:
        """Format a question into the SFT format for SHP (question part only)."""
        return f"Question: {question}\nAnswer: "
    
    def format_sft_shp_answer(self, answer: str) -> str:
        """Format an answer into the SFT format for SHP (answer part only)."""
        return answer
    
    def format_sft_openmath_question(self, question: str) -> str:
        """Format a question into the SFT format for OpenMathInstruct (question part only)."""
        return f"### Instruction:\n{question}\n\n### Response:\n"
    
    def format_sft_openmath_answer(self, answer: str, eos_token: str = "") -> str:
        """Format an answer into the SFT format for OpenMathInstruct (answer part only)."""
        return f"{answer}{eos_token}"
    
    def format_sft_full(self, question: str, answer: str, dataset_name: str, eos_token: str = "") -> str:
        """Format a full question-answer pair into SFT format (for reference)."""
        if dataset_name.lower() == "shp":
            return f"Question: {question}\nAnswer: {answer}"
        else:  # openmath
            return f"### Instruction:\n{question}\n\n### Response:\n{answer}{eos_token}"
    
    def extract_and_save(self, dataset_name: str, split: str = "train", 
                         cache_dir: Optional[str] = None, eos_token: str = "") -> None:
        """Extract questions and answers from a dataset and save them in SFT format."""
        if dataset_name.lower() == "shp":
            dataset = self.load_shp_dataset()
            format_func = self.format_shp_data
            sft_q_func = self.format_sft_shp_question
            sft_a_func = self.format_sft_shp_answer
        elif dataset_name.lower() == "openmath":
            dataset = self.load_openmath_dataset(cache_dir)
            format_func = self.format_openmath_data
            sft_q_func = self.format_sft_openmath_question
            sft_a_func = lambda a: self.format_sft_openmath_answer(a, eos_token)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in the dataset")
        
        # Process the dataset
        all_qa_pairs = []
        for i in range(0, len(dataset[split]), 100):  # Process in batches to avoid memory issues
            end_idx = min(i + 100, len(dataset[split]))
            batch = dataset[split][i:end_idx]
            qa_pairs = format_func(batch)
            all_qa_pairs.extend(qa_pairs)
        
        # Prepare output files
        original_questions_file = os.path.join(self.output_dir, f"{dataset_name}_{split}_original_questions.txt")
        original_answers_file = os.path.join(self.output_dir, f"{dataset_name}_{split}_original_answers.txt")
        sft_questions_file = os.path.join(self.output_dir, f"{dataset_name}_{split}_sft_questions.txt")
        sft_answers_file = os.path.join(self.output_dir, f"{dataset_name}_{split}_sft_answers.txt")
        sft_full_file = os.path.join(self.output_dir, f"{dataset_name}_{split}_sft_full.txt")
        json_file = os.path.join(self.output_dir, f"{dataset_name}_{split}_qa_pairs.json")
        
        # Save original and SFT-formatted questions and answers
        with open(original_questions_file, 'w', encoding='utf-8') as oqf, \
             open(original_answers_file, 'w', encoding='utf-8') as oaf, \
             open(sft_questions_file, 'w', encoding='utf-8') as sqf, \
             open(sft_answers_file, 'w', encoding='utf-8') as saf, \
             open(sft_full_file, 'w', encoding='utf-8') as sff:
            
            for i, (question, answer) in enumerate(all_qa_pairs):
                # Write original question and answer
                oqf.write(f"[{i}] {question}\n\n")
                oaf.write(f"[{i}] {answer}\n\n")
                
                # Write SFT formatted question and answer separately
                sft_question = sft_q_func(question)
                sft_answer = sft_a_func(answer)
                
                sqf.write(f"[{i}] {sft_question}\n\n")
                saf.write(f"[{i}] {sft_answer}\n\n")
                
                # Write full SFT format (for reference)
                sft_full = self.format_sft_full(question, answer, dataset_name, eos_token)
                sff.write(f"[{i}] {sft_full}\n\n")
        
        # Save as JSON for easier processing
        json_data = [
            {
                "id": i,
                "sft_question": sft_q_func(q),
                "sft_answer": sft_a_func(a),
            }
            for i, (q, a) in enumerate(all_qa_pairs)
        ]
        
        with open(json_file, 'w', encoding='utf-8') as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(all_qa_pairs)} question-answer pairs from {dataset_name} {split} split")
        logger.info(f"Original questions saved to: {original_questions_file}")
        logger.info(f"Original answers saved to: {original_answers_file}")
        logger.info(f"SFT formatted questions saved to: {sft_questions_file}")
        logger.info(f"SFT formatted answers saved to: {sft_answers_file}")
        logger.info(f"Full SFT format saved to: {sft_full_file} (for reference)")
        logger.info(f"JSON data saved to: {json_file}")

def main():
    """Main function to extract and save data from different datasets."""
    output_dir = "extracted_data"
    processor = DatasetProcessor(output_dir)
    
    # Process SHP dataset
    logger.info("Processing SHP dataset...")
    processor.extract_and_save("shp", "train")
    processor.extract_and_save("shp", "validation")
    processor.extract_and_save("shp", "test")
    
    # Process OpenMathInstruct dataset with EOS token (if needed)
    logger.info("Processing OpenMathInstruct dataset...")
    cache_dir = "./hf_cache"
    eos_token = ""  
    processor.extract_and_save("openmath", "train", cache_dir, eos_token)
    processor.extract_and_save("openmath", "validation", cache_dir, eos_token)
    
    logger.info("All data extracted and saved successfully!")

if __name__ == "__main__":
    main()