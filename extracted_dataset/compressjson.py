import json
import os
import random
import glob

def reduce_json_size(input_dir, output_dir, reduction_factor=2):
    """

    
    Args:
        input_dir: 원본 JSON 파일이 있는 디렉토리
        output_dir: 축소된 JSON 파일을 저장할 디렉토리
        reduction_factor: 축소 비율 (기본값 10 = 10분의 1로 줄임)
    """

    os.makedirs(output_dir, exist_ok=True)
    

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        output_file = os.path.join(output_dir, f"reduced_{file_name}")
        

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_size = len(data)
        print(f"Processing {file_name}: {original_size} items")
        

        sample_size = max(1, original_size // reduction_factor)
        reduced_data = random.sample(data, sample_size)
        

        for i, item in enumerate(reduced_data):
            item['id'] = i
        

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reduced_data, f, ensure_ascii=False, indent=2)
        

        original_file_size = os.path.getsize(json_file) / (1024 * 1024)  # MB 단위
        reduced_file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB 단위
        
        print(f"  Original: {original_size} items, {original_file_size:.2f} MB")
        print(f"  Reduced:  {len(reduced_data)} items, {reduced_file_size:.2f} MB")
        print(f"  Reduction ratio: {original_file_size/reduced_file_size:.2f}x")
        print(f"  Saved to: {output_file}")
        print("-" * 50)

if __name__ == "__main__":
    input_dir = "test"  
    output_dir = "reduced_data_test"   
    
    reduce_json_size(input_dir, output_dir)
    print("All JSON files have been successfully reduced!")