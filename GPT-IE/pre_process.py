import csv
import numpy as np
import re
import os


def read_report(csv_path, root_dir, save_path):
    """
    Read and process medical reports.

    This function reads report information from the given CSV file, filters reports containing specific diseases,
    and saves these reports to the specified path.

    Parameters:
    csv_path: Path to the CSV file containing report information.
    root_dir: Root directory of the original report files.
    save_path: Path to save the processed reports.
    """
    # Initialize a dictionary to store report paths for each patient
    paths_dict = {}
    for i in range(10, 20):
        paths_dict[str(i)] = []

    # Open the CSV file and read report information
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Read reports containing diseases
            if row[10] == "":
                paths_dict[row[0][:2]].append(
                    root_dir + "p" + row[0][:2] + "/p" + row[0] + "/s" + row[1] + ".txt"
                )

    # Traverse each patient's report paths, read and process report contents
    for files in paths_dict:
        reports = []
        for file_path in paths_dict[files]:
            with open(file_path, "r") as f:
                contents = f.read()
            report = contents.replace("\n", "")
            if report.find("FINAL REPORT") != -1:
                report = report.split("FINAL REPORT")[1]
            findings_index = report.find("FINDINGS:")
            if findings_index == -1:
                findings_index = len(report)
            impression_index = report.find("IMPRESSION:")
            if impression_index == -1:
                impression_index = len(report)
            index = min(findings_index, impression_index)
            report = report[index:]
            reports.append({"path": file_path, "text": report})

        # Save the processed reports to the specified path
        reports_path = save_path + "/p" + files
        if not os.path.exists(reports_path):
            os.makedirs(reports_path)
        reports_path = reports_path + "/reports.npy"
        print(reports_path)
        np.save(reports_path, reports)


if __name__ == "__main__":
    root_dir = "/root/reports/files/"
    csv_path = "/root/mimic-cxr-2.0.0-chexpert.csv"
    save_path = "reports.npy"
    save_path = "./reports"
    read_report(csv_path, root_dir, save_path)

    reports = np.load("./reports/p10/reports.npy", allow_pickle=True)
    print(len(reports))
    print(reports[4])