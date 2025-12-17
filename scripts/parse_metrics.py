import json
import re

def parse_caption_metrics(file_path, output_path):
    data = {
        "metrics": {},
        "bleu_raw_stats": {},
        "images": []
    }
    
    current_image = None
    state = None  # 'generated', 'references', 'none'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        # Parse Metrics Header
        if "BLEU-1 | BLEU-2" in line:
            headers = [h.strip() for h in line.split('|')]
            i += 1
            values = [float(v.strip()) for v in lines[i].split('|')]
            for h, v in zip(headers, values):
                data["metrics"][h] = v
            i += 1
            continue
            
        # Parse BLEU Raw Statistics
        if line.startswith("BLEU Raw Statistics:"):
            i += 1
            stats_line = lines[i].strip()
            # It looks like a python dict string, let's try to parse it safely or use regex
            # Example: {'testlen': 804116, 'reflen': 520145, ...}
            # We can use json.loads if we replace single quotes, or ast.literal_eval
            import ast
            try:
                data["bleu_raw_stats"] = ast.literal_eval(stats_line)
            except Exception as e:
                print(f"Warning: Could not parse BLEU stats: {e}")
            i += 1
            continue
            
        if line.startswith("ratio:"):
            data["bleu_raw_stats"]["ratio"] = float(line.split(":")[1].strip())
            i += 1
            continue

        # Parse Image Blocks
        image_id_match = re.match(r"Image ID:\s*(.+)", line)
        if image_id_match:
            if current_image:
                data["images"].append(current_image)
            
            current_image = {
                "id": image_id_match.group(1),
                "generated_caption": "",
                "reference_captions": []
            }
            state = 'none'
            i += 1
            continue
            
        if line == "Generated caption:":
            state = 'generated'
            i += 1
            # The next line is the caption. It might be indented.
            if i < len(lines):
                current_image["generated_caption"] = lines[i].strip()
            i += 1
            continue
            
        if line == "Reference captions:":
            state = 'references'
            i += 1
            continue
            
        if state == 'references':
            if line.startswith("-"):
                ref = line[1:].strip()
                current_image["reference_captions"].append(ref)
            elif line.startswith("Image ID"):
                # Should be caught by the main loop condition ideally, 
                # but if we fall through, we decrement i to re-process this line as a start of new block
                i -= 1 # Go back to let the main loop handle Image ID
                state = 'none'
            # If it's just text but not starting with -, it might be a continuation or garbage
            # meaningful lines in references start with -
            
        i += 1
        
    # Append the last image
    if current_image:
        data["images"].append(current_image)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully parsed {len(data['images'])} images to {output_path}")

if __name__ == "__main__":
    parse_caption_metrics("../images/caption_metrics_output.txt", "caption_metrics_output.json")

