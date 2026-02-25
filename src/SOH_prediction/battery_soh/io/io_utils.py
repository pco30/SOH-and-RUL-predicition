# -*- coding: utf-8 -*-
import os
import pandas as pd

def get_file_creation_time(file_path):
    return os.path.getctime(file_path)

def extract_files(folder_path):
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return excel_files

def process_files(folder_path, sheet_names_to_import):
    file_paths = extract_files(folder_path)
    file_paths.sort(key=get_file_creation_time)
    dfs = []
    last_cycle_index = 0
    for file_path in file_paths:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        for target_sheet in sheet_names_to_import:
            if target_sheet in sheet_names:
                df = pd.read_excel(file_path, sheet_name=target_sheet)
                df['Cycle_Index'] += last_cycle_index
                last_cycle_index = df['Cycle_Index'].max()
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def battery_name_from_path(folder_path):
    return os.path.basename(os.path.normpath(folder_path))