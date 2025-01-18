import numpy as np

reports_path = "~/DILLIM/reports_process/reports/p18/reports.npy"
reports = np.load(reports_path, allow_pickle=True)
prompt = "Now you need to use your medical knowledge to help me. I will give you some medical diagnosis reports, and you need to extract disease-related information from the reports. The output should only contain the extracted disease information and should not include any other text. When extracting information, you need to follow six rules as follows: 1. Each piece of disease-related information extracted must meet the format: {severity or stage of disease}{location or organ of disease}{category of disease}, for example, the sentence is ‘New nodular opacities are clustered within the left upper lobe.’, then the disease-related information extracted should be ‘{New}{left upper lobe}{nodular opacities}’. 2. Some sentences may lack some information, in this case, you need to use {mask} instead, for example, if the sentence is ‘There is left lung pneumonia.’, then the disease-related information extracted should be {mask}{left lung}{pneumonia}. 3. If the disease was negatively mentioned in the report, for example, if the sentence is ‘There is no pneumothorax or left-sided pleural effusion.’, then the disease-related information extracted should be ‘{No}{mask}{pneumothorax}’ and ‘{No}{left-sided pleural}{effusion}’. 4. There may be multiple disease descriptions in one sentence, you need to find them all and extract disease-related information. 5. Ignore words irrelevant to disease description, for example, the sentence is ‘The heart size is normal.’, and there is no disease-related information, so you don’t need to extract information from this sentence. 6. Separate information with commas. \nNext, you can extract information from the report.\n report:\n"

k = 25
results = np.array([])
for p in range(int(1000 / k)):
    content = []
    m, n = 1, 76
    with open(
        "~/DILLIM/reports_process/examples/reports" + str(p) + ".txt", "r"
    ) as f:
        for line in f:
            content.append(line)
    for i in range(25):
        m = m + 3
        n = n + 2
        query = content[m]
        response = content[n]
        if query[-1] == "\n":
            query = query[:-1]
        if response[-1] == "\n":
            response = response[:-1]
        print(m, n, "query", query, "response", response)
        results = np.append(results, {"query": prompt + query, "response": response})
    # print('reports'+str(n)+':'+str(len(content)))

print(len(results))
import json

results_json = results.tolist()
results_json = json.dumps(results_json)
with open("~/DILLIM/reports_process/data/instruction.json", "w") as file:
    file.write(results_json)
