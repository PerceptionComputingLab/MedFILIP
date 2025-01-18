import csv
import numpy as np
import re
import os
import copy
from collections import Counter

disease_type = [
    "ascites",
    "pulmonary embolism",
    "patchy opacity",
    "pleural fluid",
    "nodular opacity",
    "pseudoaneurysm",
    "pulmonary nodules",
    "pneumoperitoneum",
    "dislocation",
    "inflammation",
    "hyperinflation",
    "tracheostomy tube",
    "pulmonary fibrosis",
    "infectious process",
    "patchy opacities",
    "cardiac enlargement",
    "bronchiectasis",
    "lesion",
    "cyst",
    "abscess",
    "embolism",
    "pneumomediastinum",
    "nodular opacities",
    "interstitial opacities",
    "density",
    "granuloma",
    "large effusion",
    "adenopathy",
    "pacemaker",
    "interstitial pulmonary edema",
    "subcutaneous emphysema",
    "parenchymal opacities",
    "hemorrhage",
    "dilatation",
    "nodules",
    "opacifications",
    "infiltrate",
    "aneurysm",
    "tension",
    "interstitial markings",
    "scoliosis",
    "fibrosis",
    "copd",
    "hernia",
    "hematoma",
    "hiatal hernia",
    "lymphadenopathy",
    "vascular engorgement",
    "occlusion",
    "airspace opacities",
    "thrombosis",
    "consolidations",
    "engorgement",
    "aeration",
    "nodule",
    "calcification",
    "calcifications",
    "scarring",
    "infection",
    "pulmonary vascular congestion",
    "fractures",
    "fracture",
    "pleural effusions",
    "emphysema",
    "pneumothoraces",
    "abnormalities",
    "congestion",
    "focal consolidation",
    "vascular congestion",
    "opacification",
    "pleural effusion",
    "consolidation",
    "opacities",
    "opacity",
    "edema",
    "cardiomegaly",
    "pulmonary edema",
    "pneumonia",
    "effusions",
    "atelectasis",
    "effusion",
    "pneumothorax",
]
disease_type_ = [
    [
        "effusion",
        "effusions",
        "pleural effusion",
        "pleural effusions",
        "large effusion",
    ],
    ["edema", "pulmonary edema", "interstitial pulmonary edema"],
    [
        "opacity",
        "nodular opacity",
        "patchy opacity",
        "patchy opacities",
        "opacities",
        "airspace opacities",
        "nodular opacities",
        "interstitial opacities",
        "parenchymal opacities",
    ],
    ["opacification", "opacifications"],
    ["consolidation", "consolidations", "focal consolidation"],
    ["vascular congestion", "congestion", "pulmonary vascular congestion"],
    ["fracture", "fractures"],
    ["calcification", "calcifications"],
]
disease_location_organ = [
    "costophrenic sinus",
    "left base",
    "ventricle",
    "mediastinal contour",
    "left basal",
    "right lung apex",
    "hemidiaphragms",
    "bronchovascular",
    "lobe",
    "parenchymal",
    "hila",
    "stomach",
    "bony structures",
    "right upper lobe",
    "vein",
    "pulmonary vascularity",
    "vessels",
    "aortic",
    "aortic valve",
    "thoracic",
    "pectoral",
    "left hemidiaphragm",
    "bony",
    "hemidiaphragmatic contour",
    "right atrium",
    "both lower",
    "apical",
    "aortic knob",
    "displaced",
    "abnormalities",
    "mediastinal and hilar contours",
    "left chest wall",
    "right hemidiaphragm",
    "distal",
    "left apical",
    "tube",
    "basilar",
    "low",
    "right apex",
    "port-a-cath",
    "et tube",
    "silhouette",
    "nasogastric tube",
    "right middle lobe",
    "vasculature",
    "pleura",
    "thoracic spine",
    "upper zone",
    "surfaces",
    "picc line",
    "diaphragm",
    "silhouettes",
    "subcutaneous",
    "pulmonary vasculature",
    "pleural",
    "mediastinal",
    "osseous",
    "pulmonary artery",
    "right hilar",
    "alveolar",
    "lower thoracic",
    "lateral",
    "right lower lobe",
    "lingula",
    "aorta",
    "venous",
    "right apical",
    "right chest wall",
    "bilateral lower",
    "upper",
    "right infrahilar",
    "cardiopulmonary",
    "hiatal",
    "perihilar",
    "contour",
    "lower",
    "carinal",
    "focal",
    "mid thoracic",
    "vertebral body",
    "central",
    "both",
    "costophrenic angles",
    "region",
    "chest",
    "left mid and lower",
    "interstitial",
    "internal jugular",
    "lymphadenopathy",
    "left upper",
    "airspace",
    "multifocal",
    "lung base",
    "pulmonary venous",
    "hilar",
    "ribs",
    "left lower lobe",
    "base",
    "lung",
    "right basal",
    "cardiomediastinal silhouette",
    "right middle and lower",
    "right lower",
    "catheter",
    "bilateral",
    "trachea",
    "hemi thorax",
    "right lateral",
    "chest wall",
    "bone",
    "contours",
    "descending",
    "overt",
    "right lung",
    "bibasal",
    "veins",
    "mid and lower",
    "endotracheal tube",
    "retrocardiac",
    "humerus",
    "left retrocardiac",
    "sternal wires",
    "right mid lung",
    "tracheostomy tube",
    "left",
    "subclavian",
    "left lung base",
    "left subclavian",
    "abdomen",
    "descending aorta",
    "right upper quadrant",
    "left mid",
    "carina",
    "lower lobes",
    "intrathoracic",
    "apex",
    "lungs",
    "right-sided",
    "cavoatrial junction",
    "middle",
    "lower lobe",
    "spine",
    "bones",
    "right upper",
    "mediastinal and hilar",
    "na",
    "adenopathy",
    "hilus",
    "clavicle",
    "upper lobe",
    "thoracic aorta",
    "mid",
    "right",
    "cardiac silhouette",
    "right perihilar",
    "cardiac",
    "right ventricle",
    "line",
    "lingular",
    "left perihilar",
    "adjacent",
    "underlying",
    "right base",
    "hilum",
    "aortic arch",
    "pericardial",
    "right basilar",
    "left-sided",
    "pulmonary",
    "pulmonary arteries",
    "osseous structures",
    "lobes",
    "shoulder",
    "right mid and lower",
    "zone",
    "left lung",
    "mid svc",
    "subdiaphragmatic",
    "mediastinum",
    "posterior",
    "vascular",
    "left mid lung",
    "thoracolumbar junction",
    "rib",
    "lung bases",
    "cardiomediastinal",
    "pulmonary venous pressure",
    "basal",
    "heart",
    "ng tube",
    "areas",
    "biapical",
    "right lung bases",
    "hemithorax",
    "left lateral",
    "atrium",
    "hilar and mediastinal",
    "left lower",
    "upper abdomen",
    "left upper quadrant",
    "pulmonary vascular",
    "bases",
    "lymph nodes",
    "both bases",
    "pericardium",
    "costophrenic angle",
    "mediastinal contours",
    "right lung base",
    "infrahilar",
    "bibasilar",
    "left pectoral",
    "soft tissues",
    "right middle",
    "esophagus",
    "lung apices",
    "pulmonary vessels",
    "hemidiaphragm",
    "right mid",
    "right internal jugular",
    "cardiac and mediastinal",
    "left basilar",
    "lymph node",
    "cardiomediastinal and hilar",
    "lung parenchyma",
    "both lung bases",
]
disease_adjective = [
    "concurrent",
    "vague",
    "probably",
    "dilated",
    "biapical",
    "small if any",
    "less prominent",
    "widening",
    "bibasal",
    "resolving",
    "consistent with",
    "subsequent",
    "smaller",
    "decreasing",
    "minimally increased",
    "postoperative",
    "healed",
    "worse",
    "worrisome",
    "significant",
    "potentially",
    "old healed",
    "asymmetrical",
    "scarring",
    "similar",
    "complete",
    "little",
    "acute",
    "prior",
    "moderate-to-severe",
    "perihilar",
    "loculated",
    "progressive",
    "possibility of",
    "basilar",
    "slightly improved",
    "infectious",
    "ill-defined",
    "lungs",
    "central",
    "supervening",
    "subsegmental",
    "massive",
    "hyperinflation",
    "partial",
    "crowding",
    "some",
    "moderate to large",
    "minor",
    "tortuosity",
    "vascular",
    "degenerative",
    "subcutaneous",
    "consolidation",
    "heterogeneous",
    "right lower",
    "presumed",
    "slightly increased",
    "moderately severe",
    "mediastinal",
    "small-to-moderate",
    "standard",
    "atelectatic",
    "upper",
    "concerning",
    "diffuse bilateral",
    "hazy",
    "interstitial",
    "top-normal",
    "moderately",
    "infection",
    "pre-existing",
    "associated",
    "blunting",
    "layering",
    "streaky",
    "left-sided",
    "moderately enlarged",
    "early",
    "dense",
    "widespread",
    "right-sided",
    "intact",
    "old",
    "interval improvement",
    "tiny",
    "improvement",
    "tortuous",
    "decrease",
    "left lower",
    "elevated",
    "elevation",
    "unremarkable",
    "interval increase",
    "moderate to severe",
    "previous",
    "marked",
    "pleural",
    "asymmetric",
    "small to moderate",
    "hyperinflated",
    "increase",
    "prominence",
    "resolved",
    "focal",
    "developing",
    "prominent",
    "improving",
    "volume loss",
    "worsened",
    "mild-to-moderate",
    "interval",
    "residual",
    "adjacent",
    "aspiration",
    "compressive",
    "retrocardiac",
    "lower",
    "constant",
    "linear",
    "known",
    "underlying",
    "subtle",
    "slight",
    "multiple",
    "likely",
    "trace",
    "continued",
    "mild to moderate",
    "calcified",
    "mildly enlarged",
    "mildly",
    "probable",
    "multifocal",
    "superimposed",
    "enlargement",
    "extensive",
    "diffuse",
    "increasing",
    "clear",
    "patchy",
    "substantial",
    "chronic",
    "enlarged",
    "borderline",
    "worsening",
    "large",
    "possible",
    "decreased",
    "bibasilar",
    "severe",
    "improved",
    "persistent",
    "bilateral",
    "left",
    "right",
    "increased",
    "minimal",
    "stable",
    "low",
    "normal",
    "new",
    "unchanged",
    "moderate",
    "small",
    "mild",
    "no",
]
disease_adjective = disease_adjective[:-1]
disease_type_2 = [
    "ascites",
    "pulmonary embolism",
    "pleural fluid",
    "pseudoaneurysm",
    "pulmonary nodules",
    "pneumoperitoneum",
    "dislocation",
    "inflammation",
    "hyperinflation",
    "tracheostomy tube",
    "pulmonary fibrosis",
    "infectious process",
    "cardiac enlargement",
    "bronchiectasis",
    "lesion",
    "cyst",
    "abscess",
    "embolism",
    "pneumomediastinum",
    "density",
    "granuloma",
    "adenopathy",
    "pacemaker",
    "subcutaneous emphysema",
    "hemorrhage",
    "dilatation",
    "nodules",
    "infiltrate",
    "aneurysm",
    "tension",
    "interstitial markings",
    "scoliosis",
    "fibrosis",
    "copd",
    "hernia",
    "hematoma",
    "hiatal hernia",
    "lymphadenopathy",
    "vascular engorgement",
    "occlusion",
    "thrombosis",
    "engorgement",
    "aeration",
    "nodule",
    "calcification",
    "scarring",
    "infection",
    "fracture",
    "emphysema",
    "pneumothoraces",
    "abnormalities",
    "vascular congestion",
    "opacification",
    "consolidation",
    "opacity",
    "cardiomegaly",
    "edema",
    "pneumonia",
    "atelectasis",
    "effusion",
    "pneumothorax",
]
view_position = ["AP", "PA", "LATERAL", "LL", ""]


def post_process(root_dir):
    """
    Post-process the processing results, including reading metadata, creating directories, loading and processing results, and saving the processed results.

    Parameters:
    root_dir: The root directory path used to construct image paths.
    """
    # Initialize a dictionary to store metadata
    paths_dict = {}
    # Open and read the metadata CSV file
    with open("./mimic-cxr-2.0.0-metadata.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Store metadata combined by subject_id and study_id
            if str(row[1]) + str(row[2]) not in paths_dict:
                paths_dict[str(row[1]) + str(row[2])] = [
                    [str(row[0]), str(row[1]), str(row[2]), str(row[4])]
                ]
            else:
                paths_dict[str(row[1]) + str(row[2])].append(
                    [str(row[0]), str(row[1]), str(row[2]), str(row[4])]
                )

    # Create directories to store post-processed results
    for i in range(10, 20):
        path = "./post_processed_results/" + "p" + str(i)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # Load and process each result file
    for p in range(10, 20):
        results = np.load("./results/p" + str(p) + "/results.npy", allow_pickle=True)
        results_not_empty = np.array([])
        save_path = "./post_processed_results/p" + str(p) + "/results.npy"
        for i in range(len(results)):
            # Split the text by commas or periods
            parts = re.split("[,.]", results[i]["prompt"])
            checked_result = []
            # Traverse the split parts and check if they meet the requirements
            for part in parts:
                if len(part) < 1:
                    continue
                # Organize the results returned by GPT
                if len(re.findall("\{[a-zA-Z\s\-/]+\}", part.strip())) == 4:
                    result = list(
                        filter(
                            lambda s: s != "", re.split("[{}]", part.strip().lower())
                        )
                    )
                    if result[3] == "na" and result[2] in disease_type:
                        result[3] = result[2]
                        result[2] = "na"

                    if result[0] == "na" and result[1] in disease_adjective:
                        result[0] = result[1]
                        result[1] = "na"

                    # Structured labels need to include disease description, disease location, disease organ, and disease type information, and these labels must be within specific ranges
                    if (
                        result[3] not in disease_type
                        or result[2] not in disease_location_organ
                        or result[1] not in disease_location_organ
                        or result[0] not in disease_adjective
                    ):
                        continue
                    # Group different descriptions of the same disease into one category
                    for original_disease_type in disease_type_:
                        if result[3] in original_disease_type:
                            result[3] = original_disease_type[0]
                    checked_result.append(result)
            results[i]["result"] = checked_result

            # Retain only records with valid results and add additional information
            if len(results[i]["result"]) >= 1:
                if len(results[i]["result"]) > 10:
                    print(results[i]["path"])
                    print(len(results[i]["result"]))
                    if len(results[i]["result"]) > 100:
                        print(results[i]["result"])
                # Remove results that do not contain structured labels after processing, and add image paths and view information
                subject_id_and_study_id = str(results[i]["path"][25:33]) + str(
                    results[i]["path"][35:43]
                )
                for j in range(len(paths_dict[subject_id_and_study_id])):
                    result = copy.deepcopy(results[i])
                    result["image_path"] = (
                        root_dir
                        + "p"
                        + paths_dict[subject_id_and_study_id][j][1][:2]
                        + "/p"
                        + paths_dict[subject_id_and_study_id][j][1]
                        + "/s"
                        + paths_dict[subject_id_and_study_id][j][2]
                        + "/"
                        + paths_dict[subject_id_and_study_id][j][0]
                        + ".jpg"
                    )
                    result["view_position"] = paths_dict[subject_id_and_study_id][j][3]
                    if (
                        result["path"]
                        == "/root/reports/files/p10/p10002559/s52212843.txt"
                    ):
                        print(result)
                    results_not_empty = np.append(results_not_empty, result)
        # Save the processed results
        np.save(save_path, results_not_empty)


def concat_npy():
    # Combine into one .npy file
    data_dir = "./post_processed_results"
    folder_names = os.listdir(data_dir)
    result = np.empty((0,))

    for folder_name in folder_names:
        file_path = os.path.join(data_dir, folder_name, "results.npy")
        data = np.load(file_path, allow_pickle=True)
        result = np.concatenate((result, data))
    np.save("./mimic.npy", result)


if __name__ == "__main__":
    # Set the root directory to the MIMIC-CXR dataset file path
    root_dir = "~/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

    # Call the post_process function to post-process the data
    post_process(root_dir)

    # Load the post-processed result file
    path = "./post_processed_results/p10/results.npy"
    results = np.load(path, allow_pickle=True)

    # Initialize an empty list to store disease information
    disease = []

    # Display the processed label information (only display results for a specific path)
    for i in range(len(results)):
        if results[i]["path"] == "/root/reports/files/p10/p10002559/s52212843.txt":
            print(results[i])

    # Traverse all results, extract and record disease information from each result
    for i in range(len(results)):
        for result in results[i]["result"]:
            disease.append(result[3])  # Assume result[3] is the disease name

    # Count the occurrences of each disease
    counts = Counter(disease)

    # Sort the counts in descending order
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Initialize three lists to store the top 100 diseases, their occurrences, and disease types
    disease_100 = []
    disease_num = []
    disease_type = []

    # Print and record the information of the last 50 diseases (i.e., the 50 diseases with the fewest occurrences)
    for i in range(len(sorted_counts) - 50, len(sorted_counts)):
        print(
            sorted_counts[len(sorted_counts) - 1 - i][0],
            sorted_counts[len(sorted_counts) - 1 - i][1],
        )
        disease_100.append(
            [
                sorted_counts[len(sorted_counts) - 1 - i][0],
                sorted_counts[len(sorted_counts) - 1 - i][1],
            ]
        )
        disease_num.append(sorted_counts[len(sorted_counts) - 1 - i][1])
        disease_type.append(sorted_counts[len(sorted_counts) - 1 - i][0])

    # Define the headers for the CSV file
    headers = ["disease", "occurrences"]

    # Write the information of the top 100 diseases to a CSV file
    with open("disease_100.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(disease_100)

    # Call the concat_npy function to merge Numpy files
    concat_npy()

    # Load the merged Numpy file
    results = np.load("./mimic.npy", allow_pickle=True)

    # Print the first 10 loaded results
    print(results[:10])
