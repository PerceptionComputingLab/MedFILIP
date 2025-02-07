import csv
import numpy as np
import re
import os
import copy
import argparse
from collections import Counter

view_position = ["AP", "PA", "LATERAL", "LL", ""]
disease_adjective = [
    "new large",
    "minimal residual",
    "slightly larger",
    "similar",
    "significant",
    "slightly more prominent",
    "interval development",
    "mild elevation",
    "small amount",
    "slightly worse",
    "more pronounced",
    "more prominent",
    "persistent mild",
    "difficult to exclude",
    "borderline size",
    "very small",
    "interval",
    "potential",
    "minimal decrease",
    "suspected",
    "slightly lower",
    "volume loss",
    "almost completely resolved",
    "slightly worsened",
    "less prominent",
    "unchanged small",
    "new moderate",
    "suggestive of",
    "persistent small",
    "substantial increase",
    "resolution of",
    "little",
    "resolution",
    "borderline enlarged",
    "slightly enlarged",
    "moderate size",
    "mild increase",
    "minimal improvement",
    "larger",
    "ill-defined",
    "moderate-sized",
    "near complete",
    "markedly enlarged",
    "widespread",
    "slight worsening",
    "possible mild",
    "newly appeared",
    "moderate-to-large",
    "bilateral small",
    "interval decrease",
    "substantially decreased",
    "resolving",
    "no change",
    "slight decrease",
    "faint",
    "asymmetric",
    "mildly",
    "left greater than right",
    "scattered",
    "interval increase",
    "mildly increased",
    "substantial decrease",
    "loculated",
    "worrisome for",
    "worse",
    "likely small",
    "small residual",
    "minimally improved",
    "progressed",
    "hyperexpansion",
    "massive",
    "lungs",
    "interval improvement",
    "presumed",
    "minimal increase",
    "complete",
    "subsequent",
    "improvement",
    "minimally increased",
    "minimally decreased",
    "stable small",
    "probable small",
    "top-normal",
    "decreasing",
    "dense",
    "developing",
    "heterogeneous",
    "stable mild",
    "partial",
    "some",
    "slight improvement",
    "superimposed",
    "normal",
    "smaller",
    "adjacent",
    "moderate-to-severe",
    "tortuosity",
    "mild to moderately enlarged",
    "possible trace",
    "small-to-moderate",
    "mild improvement",
    "bibasal",
    "slight",
    "known",
    "unchanged mild",
    "blunting",
    "minor",
    "tortuous",
    "moderately enlarged",
    "slightly decreased",
    "slightly improved",
    "moderately severe",
    "new mild",
    "multiple",
    "multifocal",
    "little change",
    "possible small",
    "constant",
    "upper limits of normal",
    "focal",
    "residual",
    "new small",
    "layering",
    "slight increase",
    "stable moderate",
    "diffuse",
    "hazy",
    "calcified",
    "increase",
    "elevated",
    "atelectatic",
    "marked",
    "decrease",
    "moderate to large",
    "associated",
    "unchanged moderate",
    "slightly increased",
    "lower",
    "basilar",
    "moderate to severe",
    "small bilateral",
    "right basilar",
    "hyperinflated",
    "improving",
    "streaky",
    "left retrocardiac",
    "elevation",
    "small to moderate",
    "mild-to-moderate",
    "continued",
    "probable",
    "right",
    "left basilar",
    "linear",
    "subtle",
    "extensive",
    "borderline",
    "left",
    "tiny",
    "enlargement",
    "mild to moderate",
    "pleural",
    "compressive",
    "possible",
    "chronic",
    "trace",
    "likely",
    "mildly enlarged",
    "worsening",
    "resolved",
    "increasing",
    "retrocardiac",
    "worsened",
    "substantial",
    "patchy",
    "persistent",
    "decreased",
    "enlarged",
    "severe",
    "large",
    "bilateral",
    "stable",
    "bibasilar",
    "low",
    "minimal",
    "new",
    "increased",
    "atelectasis",
    "improved",
    "unchanged",
    "moderate",
    "small",
    "mask",
    "mild",
]
disease_location_organ = [
    "left lung base laterally",
    "parenchymal",
    "upper lobe",
    "central pulmonary vasculature",
    "bronchial wall",
    "posteriorly",
    "bibasilar lung",
    "bilateral alveolar",
    "retrocardiac lung areas",
    "right costophrenic sulcus",
    "bilateral basilar",
    "right subpulmonic",
    "pulmonary interstitial",
    "biapical",
    "costophrenic angle",
    "right mid lung field",
    "left costophrenic sinus",
    "pericardial",
    "bilateral lung",
    "bilateral upper lobe",
    "bilateral bases",
    "bilateral hilar",
    "bilateral basal",
    "bibasilar subsegmental",
    "left lung basis",
    "right lung basis",
    "main pulmonary artery",
    "right perihilar region",
    "left retrocardiac region",
    "mediastinal and hilar",
    "diffuse bilateral pulmonary",
    "left lateral chest wall",
    "right major fissure",
    "right pleural space",
    "widespread parenchymal",
    "left mid to lower lung",
    "left infrahilar",
    "left hemidiaphragmatic",
    "left ventricular",
    "bilateral costophrenic angles",
    "right upper zone",
    "cardiomediastinal",
    "lung",
    "right medial lung base",
    "right mid zone",
    "lung parenchyma",
    "left chest wall",
    "right mid-to-lower lung",
    "middle lobe",
    "right base medially",
    "costophrenic angles",
    "left and right lung base",
    "upper zone",
    "bilateral basal parenchymal",
    "heart size",
    "lingular",
    "ascending aorta",
    "left midlung",
    "hila",
    "aortic arch",
    "left pneumothorax",
    "small bilateral",
    "small right",
    "retrocardiac lung regions",
    "small left",
    "both lower lungs",
    "opacities",
    "right and left lung bases",
    "heart size and mediastinum",
    "mediastinal",
    "right mid to lower lung",
    "bilateral lower lung",
    "bilateral interstitial",
    "bilateral lung bases",
    "hemidiaphragms",
    "right cardiophrenic angle",
    "base",
    "both lower lobes",
    "lung apices",
    "left and right lung bases",
    "central pulmonary vascular",
    "bilateral airspace",
    "bilateral pleural effusions",
    "right chest wall",
    "left mid lung field",
    "right middle and lower lobes",
    "base of the left lung",
    "right lower hemithorax",
    "stomach",
    "left hilus",
    "interstitial markings",
    "bibasilar opacities",
    "bilateral lungs",
    "lower lung",
    "right hemidiaphragmatic",
    "bilateral effusions",
    "left hilar",
    "bilateral perihilar",
    "interstitial pulmonary",
    "right middle and lower lobe",
    "bibasilar airspace",
    "right hilus",
    "right midlung",
    "base of the right lung",
    "pulmonary vasculature",
    "left lung apex",
    "upper lobes",
    "biapical pleural",
    "left base retrocardiac",
    "hilar",
    "both bases",
    "thoracic spine",
    "left heart border",
    "right perihilar",
    "right hilum",
    "aortic knob",
    "vascular",
    "left apex",
    "right heart border",
    "right infrahilar region",
    "lower lobe",
    "left mid and lower lung",
    "small bilateral pleural",
    "pulmonary vessels",
    "lower lobes",
    "descending aorta",
    "cardiomediastinal silhouette",
    "upper lungs",
    "right mid and lower lung",
    "right hilar",
    "left-sided",
    "right apex",
    "perihilar",
    "right infrahilar",
    "left upper lung",
    "bibasal",
    "pleural effusions",
    "right greater than left",
    "small right pleural",
    "pulmonary venous",
    "right lung apex",
    "right pleural effusion",
    "aorta",
    "lower lungs",
    "both lungs",
    "apical",
    "lingula",
    "small left pleural",
    "mediastinum",
    "bilateral lower lobe",
    "right-sided",
    "retrocardiac region",
    "thoracic aorta",
    "bilaterally",
    "left perihilar",
    "right lung bases",
    "cardiomegaly",
    "bilateral pulmonary",
    "left lung bases",
    "right costophrenic angle",
    "right upper lung",
    "left costophrenic angle",
    "right hemithorax",
    "bilateral parenchymal",
    "left pleural effusion",
    "basilar",
    "lungs",
    "right mid lung",
    "left hemithorax",
    "both lung bases",
    "left basal",
    "left mid lung",
    "right-sided pleural",
    "interstitial",
    "left-sided pleural",
    "left hemidiaphragm",
    "pleural",
    "right basal",
    "left retrocardiac",
    "left lower lung",
    "right basilar",
    "pulmonary",
    "right middle lobe",
    "bases",
    "left upper lobe",
    "left apical",
    "left basilar",
    "right hemidiaphragm",
    "pulmonary vascular",
    "right lower lung",
    "left lung",
    "heart",
    "lung bases",
    "bilateral",
    "right apical",
    "retrocardiac",
    "right lung",
    "right upper lobe",
    "left lung base",
    "right base",
    "right",
    "right lung base",
    "left",
    "left base",
    "cardiac",
    "bibasilar",
    "right lower lobe",
    "cardiac silhouette",
    "bilateral pleural",
    "left lower lobe",
    "right pleural",
    "left pleural",
    "mask",
]
disease_type_original = [
    "opacity with air bronchograms",
    "effusion with atelectasis",
    "central lymph node enlargement",
    "osseous metastatic disease",
    "dextroscoliosis",
    "perihilar edema",
    "tube",
    "space",
    "hemorrhage",
    "opacification concerning for pneumonia",
    "consistent with pulmonary disease",
    "pulmonary vascular re-distribution",
    "cardiomyopathy or pericardial effusion",
    "hiatus hernia",
    "central pulmonary vascular congestion",
    "lesion",
    "interstitial opacity",
    "ground-glass opacification",
    "opacities compatible with pneumonia",
    "centralized pulmonary edema",
    "septal thickening",
    "underlying atelectasis",
    "distention",
    "scar",
    "nodular densities",
    "consistent with copd",
    "component",
    "poorly defined opacity",
    "mediastinal venous engorgement",
    "thoracic aorta",
    "hila",
    "cavitation",
    "volume overload",
    "metastatic disease",
    "opacities represent atelectasis",
    "streaks",
    "engorged",
    "tension",
    "pulmonary vascularity",
    "mediastinal and pulmonary vascular engorgement",
    "rib fracture",
    "early pneumonia",
    "overt pulmonary edema",
    "lung disease",
    "consolidations concerning for pneumonia",
    "pneumonic infiltrate",
    "opacity reflecting atelectasis",
    "hematoma",
    "enlarging effusions",
    "compression fracture",
    "underlying consolidation",
    "opacities due to atelectasis",
    "versus scarring",
    "bronchial inflammation",
    "mediastinal vascular engorgement",
    "pulmonary arteries",
    "hyperinflated",
    "opacification consistent with pneumonia",
    "thickening or effusion",
    "opacification consistent with effusion and atelectasis",
    "of the thoracic aorta",
    "heart failure",
    "opacities consistent with pneumonia",
    "opacity suggesting pneumonia",
    "postoperative appearance",
    "ards",
    "relative elevation",
    "cardiac decompensation",
    "pulmonary hemorrhage",
    "impression",
    "gas",
    "fluid collection",
    "prominence of interstitial markings",
    "abnormalities",
    "picc line",
    "rounded density",
    "biapical thickening",
    "diameter",
    "airspace process",
    "densities",
    "appearance",
    "intrathoracic malignancy",
    "flattening",
    "venous pressure",
    "pulmonary vascular redistribution",
    "plaque",
    "eventration",
    "dilatation",
    "of the costophrenic angle",
    "interstitial changes",
    "collapsed",
    "air",
    "osteopenia",
    "hilar congestion",
    "compression deformity",
    "alveolar opacities",
    "volume",
    "chf findings",
    "haziness",
    "hyperexpanded",
    "acute intrathoracic process",
    "catheter",
    "vascular redistribution",
    "clear",
    "peribronchial cuffing",
    "apical thickening",
    "lung nodules",
    "lucency",
    "pulmonary venous hypertension",
    "or pneumonia",
    "perihilar opacities",
    "acute cardiopulmonary abnormality",
    "mediastinum",
    "lung opacities",
    "obscuration",
    "pulmonary hypertension",
    "aspiration",
    "shift",
    "emphysematous changes",
    "silhouette",
    "process",
    "pulmonary arterial hypertension",
    "pulmonary abnormality",
    "of the lungs",
    "markings",
    "masses",
    "interstitial prominence",
    "infiltrative pulmonary abnormality",
    "atherosclerotic calcifications",
    "lobe pneumonia",
    "pulmonary vasculature",
    "heart enlargement",
    "granulomas",
    "alveolar infiltrate",
    "radiodensity",
    "infiltrates",
    "elongation",
    "relevant change",
    "air-fluid level",
    "contour",
    "bronchiectasis",
    "of pulmonary venous pressure",
    "and opacities",
    "collection",
    "airspace disease",
    "confluent opacity",
    "contours",
    "interstitial pulmonary abnormality",
    "pulmonary opacities",
    "redistribution",
    "obstructive pulmonary disease",
    "degenerative changes",
    "mediastinal widening",
    "calcification",
    "aorta",
    "pulmonary congestion",
    "vascular engorgement",
    "indistinctness",
    "pulmonary opacifications",
    "fullness",
    "adenopathy",
    "fracture",
    "bronchial wall thickening",
    "fractures",
    "interstitial lung disease",
    "infection",
    "hydropneumothorax",
    "fibrosis",
    "plaques",
    "ventilation",
    "widening",
    "scoliosis",
    "rib fractures",
    "pulmonary disease",
    "lymphadenopathy",
    "pulmonary nodules",
    "pressure",
    "pneumoperitoneum",
    "reticular opacities",
    "acute cardiopulmonary disease",
    "airspace opacity",
    "change",
    "overinflation",
    "peribronchial opacification",
    "pneumomediastinum",
    "plate-like atelectasis",
    "calcifications",
    "granuloma",
    "pulmonary venous pressure",
    "interstitial markings",
    "interstitial abnormality",
    "congestive heart failure",
    "pulmonary fibrosis",
    "airspace consolidation",
    "pulmonary vascular engorgement",
    "infiltrate",
    "abnormality",
    "parenchymal opacity",
    "interstitial opacities",
    "acute cardiopulmonary process",
    "tortuosity",
    "hyperexpansion",
    "engorgement",
    "subcutaneous emphysema",
    "heart",
    "lungs",
    "density",
    "opacifications",
    "chf",
    "hyperinflation",
    "fluid",
    "atelectaxic changes",
    "blunting",
    "heart size",
    "prominence",
    "airspace opacities",
    "parenchymal opacities",
    "mass",
    "enlarged",
    "hiatal hernia",
    "copd",
    "interstitial pulmonary edema",
    "volume loss",
    "consolidations",
    "emphysema",
    "changes",
    "aeration",
    "scarring",
    "collapse",
    "thickening",
    "size",
    "fluid overload",
    "elevation",
    "congestion",
    "vascular congestion",
    "nodule",
    "lung volumes",
    "pulmonary vascular congestion",
    "opacification",
    "enlargement",
    "consolidation",
    "pneumonia",
    "cardiac",
    "edema",
    "opacity",
    "pneumothorax",
    "atelectasis",
    "effusion",
]

disease_type = [
    "degenerative changes",
    "adenopathy",
    "collection",
    "calcification",
    "airspace disease",
    "mediastinum",
    "rib fractures",
    "vascular engorgement",
    "interstitial pulmonary abnormality",
    "mediastinal widening",
    "plaques",
    "hydropneumothorax",
    "bronchial wall thickening",
    "calcifications",
    "pulmonary opacifications",
    "scoliosis",
    "infection",
    "ventilation",
    "lymphadenopathy",
    "pneumoperitoneum",
    "reticular opacities",
    "airspace opacity",
    "granuloma",
    "overinflation",
    "peribronchial opacification",
    "pneumomediastinum",
    "parenchymal opacity",
    "airspace consolidation",
    "congestive heart failure",
    "pulmonary fibrosis",
    "infiltrate",
    "engorgement",
    "interstitial abnormality",
    "pulmonary vascular engorgement",
    "subcutaneous emphysema",
    "interstitial opacities",
    "pulmonary venous pressure",
    "interstitial markings",
    "tortuosity",
    "hyperexpansion",
    "acute cardiopulmonary process",
    "fluid",
    "prominence",
    "opacifications",
    "chf",
    "hyperinflation",
    "airspace opacities",
    "blunting",
    "parenchymal opacities",
    "atelectaxic changes",
    "mass",
    "consolidations",
    "copd",
    "emphysema",
    "thickening",
    "hiatal hernia",
    "volume loss",
    "interstitial pulmonary edema",
    "scarring",
    "aeration",
    "collapse",
    "fluid overload",
    "nodule",
    "vascular congestion",
    "opacification",
    "consolidation",
    "pneumonia",
    "opacity",
    "edema",
    "pneumothorax",
    "atelectasis",
    "effusion",
]

if "mask" in disease_type:
    disease_type = disease_type.remove("mask")

disease_divide = {
    "effusion and atelectasis": ["effusion", "atelectasis"],
    "collapse and/or consolidation": ["collapse", "consolidation"],
    "consolidation compatible with pneumonia": ["consolidation", "pneumonia"],
    "consolidation concerning for pneumonia": ["consolidation", "pneumonia"],
    "consolidative opacity": ["consolidation", "opacity"],
    "consolidative opacities": ["consolidation", "opacity"],
    "opacity compatible with pneumonia": ["opacity", "pneumonia"],
    "opacity consistent with pneumonia": ["opacity", "pneumonia"],
    "opacities concerning for pneumonia": ["opacity", "pneumonia"],
    "opacity concerning for pneumonia": ["opacity", "pneumonia"],
    "opacity compatible with atelectasis": ["opacity", "atelectasis"],
    "opacities suggestive of atelectasis": ["opacity", "atelectasis"],
    "opacities reflect atelectasis": ["opacity", "atelectasis"],
    "opacities atelectasis": ["opacity", "atelectasis"],
    "opacities reflecting atelectasis": ["opacity", "atelectasis"],
    "nodular opacification": ["nodular", "opacification"],
}

disease_type_repeat = [
    [
        "atelectasis",
        "volume loss/infiltrate",
        "basal atelectasis",
        "plate atelectasis",
        "scarring or atelectasis",
        "areas of atelectasis",
        "lobe atelectasis",
        "atelectasis/scarring",
        "platelike atelectasis",
        "plate-like atelectasis",
        "subsegmental atelectasis",
    ],
    [
        "effusion",
        "to effusions",
        "effusion or thickening",
        "to effusion",
        "and effusion",
        "pericardial effusion",
        "effusions",
        "effusion",
    ],
    [
        "consolidation",
        "region of consolidation",
        "and/or consolidation",
        "areas of consolidation",
        "or consolidation",
        "pulmonary consolidation",
        "airspace consolidation",
        "consolidations",
        "consolidation",
    ],
    [
        "pneumonia",
        "aspiration pneumonia",
        "pneumonia",
        "acute pneumonia",
    ],
    [
        "pneumothorax",
        "apical pneumothorax",
        "hydro pneumothorax",
        "pneumothoraces",
        "pneumothorax",
    ],
    [
        "opacity",
        "rounded opacity",
        "airspace opacification",
        "pulmonary opacification",
        "nodular opacification",
        "reticulonodular opacities",
        "pulmonary opacities",
        "confluent opacity",
        "opacities",
        "opacification",
        "opacifications",
        "opacity",
    ],
    [
        "scarring",
        "fibrotic changes",
        "scarring or atelectasis",
        "atelectasis/scarring",
        "or scarring",
        "scarring",
    ],
    [
        "cardiac",
        "cardiomediastinal silhouette",
        "to cardiomegaly",
        "cardiac silhouette enlargement",
        "of the cardiac silhouette",
        "cardiac silhouette",
        "cardiac enlargement",
        "cardiomegaly",
    ],
    [
        "edema",
        "pulmonary interstitial edema",
        "to pulmonary edema",
        "interstitial edema",
        "pulmonary edema",
        "edema",
    ],
    [
        "nodule",
        "lung nodule",
        "nodules",
        "pulmonary nodule",
        "nodular density",
        "nodular opacities",
        "pulmonary nodules",
        "nodular opacity",
        "nodular",
        "nodule",
    ],
    [
        "vascular congestion",
        "pulmonary vascular congestion",
        "congestion",
        "pulmonary congestion",
    ],
    ["pulmonary fibrosis", "fibrosis"],
]

# disease_type_repeat = [
#   [
#       "atelectasis",
#       "volume loss/infiltrate",
#       "region of consolidation",
#       "opacity compatible with atelectasis",
#       "basal atelectasis",
#       "opacities suggestive of atelectasis",
#       "plate atelectasis",
#       "opacities reflect atelectasis",
#       "scarring or atelectasis",
#       "opacities atelectasis",
#       "areas of atelectasis",
#       "effusion and atelectasis",
#       "lobe atelectasis",
#       "opacities reflecting atelectasis",
#       "atelectasis/scarring",
#       "platelike atelectasis",
#       "plate-like atelectasis",
#       "subsegmental atelectasis",
#   ],
#   [
#       "effusion",
#       "to effusions",
#       "effusion or thickening",
#       "to effusion",
#       "and effusion",
#       "effusion and atelectasis",
#       "pericardial effusion",
#       "effusions",
#       "effusion",
#   ],
#   [
#       "consolidation",
#       "region of consolidation",
#       "and/or consolidation",
#       "areas of consolidation",
#       "or consolidation",
#       "collapse and/or consolidation",
#       "consolidation compatible with pneumonia",
#       "consolidation concerning for pneumonia",
#       "pulmonary consolidation",
#       "consolidative opacity",
#       "consolidation concerning for pneumonia",
#       "consolidative opacities",
#       "consolidation",
#   ],
#   [
#       "pneumonia",
#       "opacity compatible with pneumonia",
#       "opacity consistent with pneumonia",
#       "opacities concerning for pneumonia",
#       "aspiration pneumonia",
#       "consolidation compatible with pneumonia",
#       "consolidation concerning for pneumonia",
#       "pneumonia",
#       "acute pneumonia",
#       "opacity concerning for pneumonia",
#   ],
#   [
#       "pneumothorax",
#       "apical pneumothorax",
#       "hydro pneumothorax",
#       "pneumothoraces",
#       "pneumothorax",
#   ],
#   [
#       "opacity",
#       "opacity compatible with pneumonia",
#       "opacity compatible with atelectasis",
#       "rounded opacity",
#       "opacity consistent with pneumonia",
#       "opacities suggestive of atelectasis",
#       "airspace opacification",
#       "pulmonary opacification",
#       "nodular opacification",
#       "opacities concerning for pneumonia",
#       "opacities reflect atelectasis",
#       "opacities atelectasis",
#       "reticulonodular opacities",
#       "opacities reflecting atelectasis",
#       "opacities concerning for pneumonia",
#       "opacity concerning for pneumonia",
#       "pulmonary opacities",
#       "confluent opacity",
#       "opacities",
#       "opacity",
#   ],
#   [
#       "scarring",
#       "fibrotic changes",
#       "scarring or atelectasis",
#       "atelectasis/scarring",
#       "or scarring",
#       "scarring",
#   ],
#   [
#       "cardiac",
#       "cardiomediastinal silhouette",
#       "to cardiomegaly",
#       "cardiac silhouette enlargement",
#       "of the cardiac silhouette",
#       "cardiac silhouette",
#       "cardiac enlargement",
#       "cardiomegaly",
#   ],
#   [
#       "edema",
#       "pulmonary interstitial edema",
#       "to pulmonary edema",
#       "interstitial edema",
#       "pulmonary edema",
#       "edema",
#   ],
#   [
#       "nodule",
#       "lung nodule",
#       "nodular opacification",
#       "nodules",
#       "pulmonary nodule",
#       "nodular density",
#       "nodular opacities",
#       "pulmonary nodules",
#       "nodular opacity",
#       "nodule",
#   ],
#   [
#       "vascular congestion",
#       "pulmonary vascular congestion",
#       "congestion",
#       "pulmonary congestion",
#   ],
#   ["pulmonary fibrosis", "fibrosis"],
# ]


def post_process(response_dir, image_dir, csv_dir):
    paths_dict = {}
    # 用subject_id(row[1])加study_id(row[2])的方式区分每一个样本,并保存对应图片名称(row[0])和视图信息(row[4])
    with open(csv_dir, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if str(row[1]) + str(row[2]) not in paths_dict:
                paths_dict[str(row[1]) + str(row[2])] = [
                    [str(row[0]), str(row[1]), str(row[2]), str(row[4])]
                ]
            else:
                paths_dict[str(row[1]) + str(row[2])].append(
                    [str(row[0]), str(row[1]), str(row[2]), str(row[4])]
                )

    for i in range(10, 20):
        path = "./post_processed_results/" + "p" + str(i)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for p in range(10, 20):
        results = np.load(
            response_dir + "/p" + str(p) + "/results.npy", allow_pickle=True
        )
        results_not_empty = np.array([])
        save_path = "./post_processed_results/p" + str(p) + "/results.npy"
        for i in range(len(results)):
            # 将文本用逗号或句号分隔
            parts = re.split("[,.]", results[i]["response"])
            checked_result = []
            # 遍历分隔后的部分，判断是否符合格式要求
            for part in parts:
                if len(part) < 1:
                    continue
                if len(re.findall("\{[a-zA-Z\s\-/]+\}", part.strip())) == 3:
                    # 用{}将不同疾病信息分开
                    result = list(
                        filter(
                            lambda s: s != "", re.split("[{}]", part.strip().lower())
                        )
                    )
                    # 处理疾病类别为空，且疾病类别与疾病部位错位的情况
                    if result[2] == "mask" and result[1] in disease_type:
                        result[2] = result[1]
                        result[1] = "mask"
                    # 处理疾病严重程度为空，且疾病严重程度与疾病部位错位的情况
                    if result[0] == "mask" and result[1] in disease_adjective:
                        result[0] = result[1]
                        result[1] = "mask"
                    if result[1] == "mask" and result[0] in disease_location_organ:
                        result[1] = result[0]
                        result[0] = "mask"
                    # 处理疾病严重程度被包括在疾病类别里的情况
                    for words in result[2].split(" "):
                        adjective = ""
                        if words in disease_adjective:
                            result[2] = result[2].replace(words + " ", "")
                            adjective = adjective + words
                        if result[0] == "mask" and adjective != "":
                            result[0] = adjective
                            if "no" in adjective:
                                result[0] = "no"
                    # 把同类疾病的不同形式归为一类
                    for disease_type_repeat_sub in disease_type_repeat:
                        if result[2] in disease_type_repeat_sub:
                            result[2] = disease_type_repeat_sub[0]
                            # 处理疾病类别里包含多种疾病的情况
                    if result[2] in disease_divide:
                        if (
                            result[1] not in disease_location_organ
                            or result[0] not in disease_adjective
                        ):
                            continue
                        checked_result.append(
                            [result[0], result[1], disease_divide[result[2]][0]]
                        )
                        checked_result.append(
                            [result[0], result[1], disease_divide[result[2]][1]]
                        )
                    else:
                        # 结构化标签中需要包含疾病描述、疾病位置疾病类型信息，且这些标签都分别要在特定的范围内
                        if (
                            result[2] not in disease_type
                            or result[1] not in disease_location_organ
                            or result[0] not in disease_adjective
                        ):
                            continue
                        checked_result.append(result)
            results[i]["result"] = checked_result

            if len(results[i]["result"]) >= 1:
                # 剔除处理后不包含结构化标签的结果，加入图片路径以及视图信息
                subject_id_and_study_id = str(results[i]["path"][14:22]) + str(
                    results[i]["path"][24:32]
                )
                for j in range(len(paths_dict[subject_id_and_study_id])):
                    result = copy.deepcopy(results[i])
                    result["image_path"] = (
                        image_dir
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
                    # tokens = tokenizer(
                    #     result["result"], result["view_position"])
                    # padding = 10 - len(tokens)
                    # result["tokens"] = tokens
                    results_not_empty = np.append(results_not_empty, result)
        np.save(save_path, results_not_empty)


def concat_npy(save_path):
    # 合成一个.npy
    data_dir = "./post_processed_results"
    folder_names = os.listdir(data_dir)
    result = np.empty((0,))

    for folder_name in folder_names:
        file_path = os.path.join(data_dir, folder_name, "results.npy")
        data = np.load(file_path, allow_pickle=True)
        result = np.concatenate((result, data))
    np.save(save_path, result)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--image_dir",
        type=str,
        help="physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
        default="~/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        help="mimic-cxr-2.0.0-metadata.csv",
        default="./mimic-cxr-2.0.0-metadata.csv",
    )
    parser.add_argument(
        "--extracted_entity",
        type=str,
        help="extracted_entity",
        default="./extracted_entity/llama3_fine_tuned/",
    )
    parser.add_argument(
        "--save_path", type=str, help="save_path", default="./mimic_cxr.npy"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    csv_dir = args.csv_dir
    extracted_entity = args.extracted_entity
    save_path = args.save_path
    post_process(extracted_entity, image_dir, csv_dir)
    concat_npy(save_path)
    results = np.load(save_path, allow_pickle=True)
    print(results[:10])
