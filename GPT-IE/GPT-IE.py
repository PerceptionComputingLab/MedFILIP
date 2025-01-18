from typing import List
import json
import sys
import numpy as np
import time
import argparse
import openai

prompts = [
    "Now you need to use your medical knowledge to help me. I will give you some medical image diagnosis reports, and you need to extract disease-related information from the reports sentence by sentence. When extracting information, you need to follow six rules as follows: \n 1. Each piece of disease-related information extracted must meet the format: {descriptor of disease}{location of disease}{organ of disease}{category of disease}, for example, the sentence is 'New nodular opacities are clustered within the left upper lobe.', then the disease-related information extracted should be '{New}{left upper}{lobe}{nodular opacities}'.\n2. Some sentences may lack some information, in this case, you need to use {NA} instead, for example, if the sentence is 'There is left lung pneumonia.', then the disease-related information extracted should be {NA}{left}{lung}{pneumonia}.\n3. If the disease was negatively mentioned in the report, for example, if the sentence is 'There is no pneumothorax or left-sided pleural effusion.', then the disease-related information extracted should be '{No}{NA}{NA}{pneumothorax}' and '{No}{left-sided}{pleural}{effusion}'.\n4. There may be multiple disease descriptions in one sentence, you need to find them all and extract disease-related information.\n5. Ignore words irrelevant to disease description, for example, the sentence is 'The heart size is normal.', and there is no disease-related information, so you don't need to extract information from this sentence .\n6. Separate information with commas.",
    "Now you are a medical professional.Each report describes up to 13 diseases: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia ,Pneumothorax, Pleural Other, Support Devices and No Finding. I will give you some reports, you need to label these 13 diseases, if the label is not 3, you need to provide the relevant text in the report to support the reason for this label. The rules for labeling are as follows: \nThe label has four values: 0,1,-1,3. These values have the following interpretation: \n1: The disease was positively mentioned in the report, for example, ‘A large pleural effusion’. 0: The disease was negatively mentioned in the report, for example, ‘No pneumothorax.’. -1: The disease was mentioned with uncertainty in the report, for example, ‘The cardiac size cannot be evaluated.’ , or mentioned with ambiguous language in the report and it is unclear if the pathology exists or not, for example, ‘The cardiac contours are stable.’. 3: No mention of the disease was made in the report.",
    "Now you are a medical professional.I will give you some reports, you need to label the diseases mentioned in the reports. The rules for labeling are as follows: \nThe label has three values: 0,1,-1. These values have the following interpretation: \n1: The disease was positively mentioned in the report, for example, ‘A large pleural effusion’.\n 0: The disease was negatively mentioned in the report, for example, ‘No pneumothorax.’.\n -1: The disease was mentioned with uncertainty in the report, for example, ‘The cardiac size cannot be evaluated.’ , or mentioned with ambiguous language in the report and it is unclear if the pathology exists or not, for example, ‘The cardiac contours are stable.’.\n You need to give the disease type and its label and the relevant text in the report to support the reason for the label, for example, 'pleural effusion:1 (Small pleural effusion in the right middle fissure is new)'",
    "Now you are a medical professional. I will give you some reports, you need to extract information from the report according to the following prompt template:{disease adjective}{disease location}{disease type}.For example,{acute}{heart}{heart consolidation}.Note that multiple diseases may be described in one report, you need to find them all and extract them with the prompt template.In addition, if 'disease adjective' or 'disease location' information is missing, use {NA} to indicate, but the 'disease type' shoule not be {NA}.If disease was negatively mentioned in the report, for example,‘No pneumothorax.’,the {disease adjective} should be {No}. Common types of diseases include: Engorgement,Consolidation, Opacity, Aerate, Deformity, Fractures, Thicken, Calcification, Aspiration, Pneumonia, Effusion, Pneumothorax. You should strictly follow the prompt template. One message per line, don't put multiple diseases in one message, ignore all text in the report except the disease description.I will use the information you provide for image classification, try to generate information that fits my task.",
]


class SimChatGPT:

    def __init__(self, api_key: str, messages: List = None):
        openai.api_key = api_key
        if messages:
            self.messages = messages
        else:
            self.messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompts[0]},
            ]

    def ask(self) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=self.messages,
            temperature=0,
        )
        response_content = response["choices"][0]["message"]["content"]
        return response_content

    def predict(self, report: str) -> str:
        self.messages.append({"role": "user", "content": report})
        response_content = self.ask()
        self.messages.pop()
        return response_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", type=str, help="p10-p19")
    args = parser.parse_args()
    # Reports for test
    reports = [
        "The lung volumes are low.  The cardiac, mediastinal and hilar contours appear unchanged, allowing for differences in technique. There are a number of round nodular densities projecting over each upper lung, but more numerous and discretely visualized in the left upper lobe, similar to prior study.  However, in addition, there is a more hazy widespread opacity projecting over the left mid upper lung which could be compatible with a coinciding pneumonia.  Pulmonary nodules in the left upper lobe are also not completely characterized on this study.  There is no pleural effusion or pneumothorax. Post-operative changes are similar along the right chest wall.",
        "Lung volumes remain low. There are innumerable bilateral scattered small pulmonary nodules which are better demonstrated on recent CT. Mild pulmonary vascular congestion is stable. The cardiomediastinal silhouette and hilar contours are unchanged. Small pleural effusion in the right middle fissure is new. There is no new focal opacity to suggest pneumonia. There is no pneumothorax.",
    ]

    api_key = "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"
    sim_chatgpt = SimChatGPT(api_key=api_key)
    print(sim_chatgpt.ask())
    reports_path = "./reports/" + args.group + "/reports.npy"
    reports = np.load(reports_path, allow_pickle=True)
    results_path = "./results/" + args.group + "/results.npy"
    resume_path = "./results/" + args.group + "/resumes.npy"

    try:
        results = np.load(results_path, allow_pickle=True)
    except:
        try:
            results = np.load(resume_path, allow_pickle=True)
        except:
            results = np.array([])

    for report in reports[len(results) :]:
        try:
            if len(results) % 100 == 0 and len(results) > 0:
                np.save(results_path, results)
                print(sim_chatgpt.ask())
            if len(results) % 1000 == 0 and len(results) > 0:
                np.save(resume_path, results)

            contents = sim_chatgpt.predict(report["text"])
            results = np.append(
                results,
                {"path": report["path"], "report": report["text"], "prompt": contents},
            )
            print(report["path"])
            print(report["text"])
            print(contents)
            print(len(results))
        except Exception as e:
            print(f"Error occurred: {e}")
            print(report["path"])
            if len(results) > 0:
                np.save(resume_path, results)
            break

    if len(results) > 0:
        np.save(results_path, results)
    if len(results) == len(reports):
        time.sleep(1000)
