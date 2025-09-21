import json
import requests
import argparse
from typing import Dict
from prompt import RD, NA, format


class LLMQueryProcessor:
    def __init__(self, api_key: str, model: str = "openai/gpt-3.5-turbo", api_base: str = None):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base or "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    def query_llm(self, prompt: str, system_message: str = None) -> str:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.api_base,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                raise Exception(result)
            
        except requests.exceptions.RequestException as e:
            raise Exception(e)
        except KeyError as e:
            raise Exception(e)
        except Exception as e:
            raise Exception(e)
    
    def process_three_stage_query(self, sw_data: str, nw_data: str) -> Dict[str, str]:

        print("\n=== RD ===")
        stage1_prompt = f"{RD}\n\ndata:\n{sw_data}"
        
        stage1_result = self.query_llm(stage1_prompt)
        print(f"\n{stage1_result}")
        
        print("\n=== Format ===")
        stage2_prompt = f"This is your previous answer:\n{stage1_result}\n{format}"
        
        stage2_result = self.query_llm(stage2_prompt)
        print(f"\n{stage2_result}")
        
        print("\n=== NA ===")
        stage3_input = f"Architecture:\n{stage2_result}\nNode Weights:\n{nw_data}"
        stage3_prompt = f"{NA}\n\n{stage3_input}"
        
        stage3_result = self.query_llm(stage3_prompt)
        print(f"\n{stage3_result}")
        
        return {
            "stage2_format_result": stage2_result,
            "stage3_na_result": stage3_result,
        }
    
    def save_results(self, results: Dict[str, str], output_file: str = "llm_query_results.json"):
        print(f"\nResults saved to: {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save results: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai/gpt-3.5-turbo")
    parser.add_argument("--api_base", type=str, default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--sw_data", type=str)
    parser.add_argument("--nw_data", type=str)
    parser.add_argument("--output", type=str, default="llm_query_results.json")

    args = parser.parse_args()
    
    processor = LLMQueryProcessor(
        api_key=args.api_key,
        model=args.model,
        api_base=args.api_base
    )
    
    if args.sw_data and args.nw_data:
        sw_data = args.sw_data
        nw_data = args.nw_data
    else:
        return
    
    try:
        results = processor.process_three_stage_query(sw_data, nw_data)
        processor.save_results(results, args.output)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
