"""Canonical downstream tasks used by PSPT."""


def get_class_names(dataset_name):
    if dataset_name == 'bracs':
        return [0, 1, 2], ['Benign', 'Atypical', 'Malignant']
    if dataset_name == 'coad-msi':
        return [0, 1], ['MSS', 'MSI']
    if dataset_name is None:
        return [0, 1], ['0', '1']
    raise ValueError(f'Unsupported classification task: {dataset_name}')
