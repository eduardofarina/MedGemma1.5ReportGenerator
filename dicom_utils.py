"""
DICOM utilities for processing medical imaging studies.
"""
import io
import zipfile
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import pydicom


def extract_dicom_from_zip(zip_bytes: bytes) -> List[Tuple[str, pydicom.Dataset]]:
    """
    Extract DICOM files from a ZIP archive.
    
    Args:
        zip_bytes: Raw bytes of the ZIP file
        
    Returns:
        List of tuples (filename, pydicom Dataset)
    """
    dicom_files = []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.lower().endswith('.dcm'):
                try:
                    file_bytes = zip_ref.read(filename)
                    ds = pydicom.dcmread(io.BytesIO(file_bytes))
                    dicom_files.append((filename, ds))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    
    return dicom_files


def get_modality(ds: pydicom.Dataset) -> str:
    """Get modality from DICOM dataset."""
    return getattr(ds, 'Modality', 'Unknown')


def get_study_info(ds: pydicom.Dataset) -> Dict:
    """Extract study information from DICOM dataset."""
    return {
        'StudyInstanceUID': getattr(ds, 'StudyInstanceUID', 'Unknown'),
        'SeriesInstanceUID': getattr(ds, 'SeriesInstanceUID', 'Unknown'),
        'SOPInstanceUID': getattr(ds, 'SOPInstanceUID', 'Unknown'),
        'Modality': get_modality(ds),
        'SeriesDescription': getattr(ds, 'SeriesDescription', 'Unknown'),
        'StudyDescription': getattr(ds, 'StudyDescription', 'Unknown'),
        'InstanceNumber': getattr(ds, 'InstanceNumber', 0),
        'SliceLocation': getattr(ds, 'SliceLocation', None),
    }


def normalize_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """
    Normalize DICOM pixel array to 8-bit grayscale.
    
    Args:
        ds: pydicom Dataset
        
    Returns:
        Normalized numpy array (0-255)
    """
    pixel_array = ds.pixel_array
    
    # Apply rescale slope and intercept if present
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array * slope + intercept
    
    # Normalize to 0-255
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    
    if pixel_max > pixel_min:
        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)
    
    return normalized


def dicom_to_pil(ds: pydicom.Dataset) -> Image.Image:
    """
    Convert DICOM dataset to PIL Image.
    
    Args:
        ds: pydicom Dataset
        
    Returns:
        PIL Image in RGB format
    """
    normalized = normalize_pixel_array(ds)
    
    # Convert to PIL Image
    if len(normalized.shape) == 2:
        # Grayscale image
        pil_image = Image.fromarray(normalized, mode='L')
        # Convert to RGB for model compatibility
        pil_image = pil_image.convert('RGB')
    elif len(normalized.shape) == 3:
        # RGB or multi-slice - take first slice if 3D
        if normalized.shape[2] <= 4:
            pil_image = Image.fromarray(normalized, mode='RGB')
        else:
            pil_image = Image.fromarray(normalized[:, :, 0], mode='L').convert('RGB')
    else:
        # Fallback to first slice
        pil_image = Image.fromarray(normalized[0], mode='L').convert('RGB')
    
    return pil_image


def organize_by_series(dicom_files: List[Tuple[str, pydicom.Dataset]]) -> Dict[str, List[Tuple[str, pydicom.Dataset]]]:
    """
    Organize DICOM files by series.
    
    Args:
        dicom_files: List of (filename, dataset) tuples
        
    Returns:
        Dictionary mapping SeriesInstanceUID to list of (filename, dataset)
    """
    series_dict = {}
    
    for filename, ds in dicom_files:
        series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
        if series_uid not in series_dict:
            series_dict[series_uid] = []
        series_dict[series_uid].append((filename, ds))
    
    return series_dict


def sort_slices_by_position(series_files: List[Tuple[str, pydicom.Dataset]]) -> List[Tuple[str, pydicom.Dataset]]:
    """
    Sort DICOM slices by their position (InstanceNumber or SliceLocation).
    
    Args:
        series_files: List of (filename, dataset) tuples for a series
        
    Returns:
        Sorted list of (filename, dataset) tuples
    """
    def get_sort_key(item):
        filename, ds = item
        # Try InstanceNumber first, then SliceLocation
        instance_num = getattr(ds, 'InstanceNumber', None)
        if instance_num is not None:
            return (0, int(instance_num))
        
        slice_loc = getattr(ds, 'SliceLocation', None)
        if slice_loc is not None:
            return (1, float(slice_loc))
        
        # Fallback to filename
        return (2, filename)
    
    return sorted(series_files, key=get_sort_key)


def process_dicom_study(zip_bytes: bytes) -> Tuple[str, List[Image.Image], Dict]:
    """
    Process a DICOM study from ZIP file.
    
    Args:
        zip_bytes: Raw bytes of the ZIP file
        
    Returns:
        Tuple of (modality, list of PIL Images, study metadata)
    """
    # Extract DICOM files
    dicom_files = extract_dicom_from_zip(zip_bytes)
    
    if not dicom_files:
        raise ValueError("No valid DICOM files found in the ZIP archive")
    
    # Get modality from first file
    first_ds = dicom_files[0][1]
    modality = get_modality(first_ds)
    
    # Get study info
    study_info = get_study_info(first_ds)
    study_info['TotalFiles'] = len(dicom_files)
    
    images = []
    
    if modality in ['CT', 'MR']:
        # 3D modality - organize by series and sort slices
        series_dict = organize_by_series(dicom_files)
        study_info['SeriesCount'] = len(series_dict)
        
        for series_uid, series_files in series_dict.items():
            # Sort slices within each series
            sorted_files = sort_slices_by_position(series_files)
            
            for filename, ds in sorted_files:
                try:
                    pil_image = dicom_to_pil(ds)
                    images.append(pil_image)
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
    else:
        # 2D modality (CR, DX, etc.) - process all files
        for filename, ds in dicom_files:
            try:
                pil_image = dicom_to_pil(ds)
                images.append(pil_image)
            except Exception as e:
                print(f"Error converting {filename}: {e}")
    
    study_info['ProcessedImages'] = len(images)
    
    return modality, images, study_info
