from typing import List
import json
import sys
import numpy as np
import time
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from swift.llm import (
    get_model_tokenizer,
    get_template,
    inference,
    ModelType,
    get_default_template_type,
)
from swift.utils import seed_everything
from swift.tuners import Swift

prompt = "Now you need to use your medical knowledge to help me. I will give you some medical diagnosis reports, and you need to extract disease-related information from the reports. The output should only contain the extracted disease information and should not include any other text. When extracting information, you need to follow six rules as follows: 1. Each piece of disease-related information extracted must meet the format: {severity or stage of disease}{location or organ of disease}{category of disease}, for example, the sentence is \u2018New nodular opacities are clustered within the left upper lobe.\u2019, then the disease-related information extracted should be \u2018{New}{left upper lobe}{nodular opacities}\u2019. 2. Some sentences may lack some information, in this case, you need to use {mask} instead, for example, if the sentence is \u2018There is left lung pneumonia.\u2019, then the disease-related information extracted should be {mask}{left lung}{pneumonia}. 3. If the disease was negatively mentioned in the report, for example, if the sentence is \u2018There is no pneumothorax or left-sided pleural effusion.\u2019, then the disease-related information extracted should be \u2018{No}{mask}{pneumothorax}\u2019 and \u2018{No}{left-sided pleural}{effusion}\u2019. 4. There may be multiple disease descriptions in one sentence, you need to find them all and extract disease-related information. 5. Ignore words irrelevant to disease description, for example, the sentence is \u2018The heart size is normal.\u2019, and there is no disease-related information, so you don\u2019t need to extract information from this sentence. 6. Separate information with commas. \nNext, you can extract information from the report.\n report:\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--checkpoint", type=str, help="checkpoint_path", default="./ckpt/"
    )
    parser.add_argument(
        "--reports", type=str, help="reports_path", default="./reports/"
    )
    parser.add_argument(
        "--extracted_entity",
        type=str,
        help="results_path",
        default="./extracted_entity/llama3_fine_tuned/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    chekpoint_path = args.checkpoint
    reports_path = args.reports
    results_path = args.results

    model_type = ModelType.llama3_8b_instruct
    template_type = get_default_template_type(model_type)
    print(f"template_type: {template_type}")

    kwargs = {}
    # kwargs['use_flash_attn'] = True  # 使用flash_attn

    model_type = ModelType.llama3_8b_instruct
    template_type = get_default_template_type(model_type)

    model, tokenizer = get_model_tokenizer(
        model_type, model_kwargs={"device_map": "auto"}
    )

    model = Swift.from_pretrained(model, chekpoint_path, inference_mode=True)
    template = get_template(template_type, tokenizer)
    seed_everything(42)
    for group in range(10, 20):
        group = "p" + str(group)
        reports = reports_path + group + "/reports.npy"
        reports = np.load(reports, allow_pickle=True)
        results_path = results_path + group + "/results.npy"
        if not os.path.exists(results_path + group):
            os.makedirs(results_path + group, exist_ok=True)

        try:
            results = np.load(results_path, allow_pickle=True)
        except:
            results = np.array([])

        for report in reports[len(results) :]:
            query = report["text"]
            response, history = inference(model, template, prompt + query)
            history.pop()
            results = np.append(
                results,
                {
                    "path": report["path"],
                    "report": report["text"],
                    "response": response,
                },
            )
            print(report["path"])
            print(query)
            print(response)
            print(len(results))
