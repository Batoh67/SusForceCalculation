import numpy as np
import pandas as pd


def export_to_excel(front_array, rear_array=None, row_labels=None, filename="output.xlsx"):
    """
    Export one or two NumPy ndarrays to Excel with transposed data and optional row labels.
    The front_array is written to the default sheet, rear_array (if given) is written to a sheet named 'Rear'.

    Parameters:
        front_array (np.ndarray): First dataset (any shape).
        rear_array (np.ndarray, optional): Second dataset to go to 'Rear' sheet.
        row_labels (list of str, optional): Labels for the first column.
        filename (str): Output Excel file name.
    """

    def prepare_data(array, labels):
        # Flatten extra dimensions and transpose
        data_2d = array.reshape(array.shape[0], -1).T
        # Check labels
        if labels is None:
            labels_list = [""] * data_2d.shape[0]
        elif len(labels) != data_2d.shape[0]:
            raise ValueError(
                f"Number of row_labels ({len(labels)}) does not match number of rows ({data_2d.shape[0]}).")
        else:
            labels_list = labels
        # Header
        num_loadcases = data_2d.shape[1]
        header = [""] + [f"loadcase{i + 1}" for i in range(num_loadcases)]
        # Combine labels and data
        rows_with_labels = [[label] + list(row) for label, row in zip(labels_list, data_2d)]
        return pd.DataFrame(rows_with_labels, columns=header)

    # Prepare front sheet
    df_front = prepare_data(front_array, row_labels)

    # Export to Excel with optional rear sheet
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_front.to_excel(writer, sheet_name="Front", index=False)
        if rear_array is not None:
            df_rear = prepare_data(rear_array, row_labels)
            df_rear.to_excel(writer, sheet_name="Rear", index=False)

    print(f"Data exported to {filename} with sheets: Front" + (", Rear" if rear_array is not None else ""))


