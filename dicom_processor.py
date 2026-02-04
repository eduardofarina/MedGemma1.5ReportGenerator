"""Utilitários DICOM para processamento de estudos de imagem médica.

Melhorias implementadas:
- Aceita arquivos DICOM mesmo sem extensão .dcm.
- Resumo de séries + seleção padrão mais inteligente.
- Exclusão opcional de localizer/scout (reduz ruído na geração).
- Amostragem inteligente (best-of-3 por "bin") para representar melhor a série
  sem precisar decodificar/empilhar todas as fatias.
- Retorna metadados das séries para alimentar um UI mais rico e prompts melhores.
"""


from __future__ import annotations

import io
import zipfile
from typing import List, Tuple, Dict, Optional, Any, Iterable

import numpy as np
from PIL import Image

import pydicom


# ---------------------------------------------------------------------------
# Helpers: PHI-safe defaults (UI may still choose to display raw fields)
# ---------------------------------------------------------------------------

def mask_patient_id(patient_id: str) -> str:
    """Mascara um identificador de paciente para evitar exposicao de PHI no UI/logs."""
    if not patient_id:
        return "REDACTED"
    s = str(patient_id).strip()
    if len(s) <= 4:
        return "REDACTED"
    return ("*" * (len(s) - 4)) + s[-4:]


# ---------------------------------------------------------------------------
# DICOM reading
# ---------------------------------------------------------------------------

def has_pixel_data(ds: pydicom.Dataset) -> bool:
    return (
        "PixelData" in ds
        or "FloatPixelData" in ds
        or "DoubleFloatPixelData" in ds
    )


def _looks_like_dicom(filename: str) -> bool:
    # Heurística simples pelo nome para pular arquivos claramente não-DICOM dentro do ZIP.
    lower = filename.lower()
    if lower.endswith("/"):
        return False
    if lower.endswith((".txt", ".md", ".json", ".csv", ".xml", ".pdf", ".jpg", ".jpeg", ".png", ".gif")):
        return False
    return True


def extract_dicom_from_zip(zip_bytes: bytes) -> List[Tuple[str, pydicom.Dataset]]:
    """
    Extrai datasets DICOM de um arquivo ZIP.

    A ideia é tentar ler qualquer arquivo que pareça plausível, porque muita exportação
    de PACS vem sem extensão .dcm.
    """
    dicom_files: List[Tuple[str, pydicom.Dataset]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
        for filename in zip_ref.namelist():
            if not _looks_like_dicom(filename):
                continue

            try:
                file_bytes = zip_ref.read(filename)
            except Exception:
                continue

            # Skip tiny files early.
            if len(file_bytes) < 256:
                continue

            try:
                ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            except Exception:
                continue

            # Precisa ter pixel data (ignora SR/dose/etc)
            if not has_pixel_data(ds):
                continue

            dicom_files.append((filename, ds))

    return dicom_files


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_modality(ds: pydicom.Dataset) -> str:
    return str(getattr(ds, "Modality", "Unknown") or "Unknown")


def _get_str(ds: pydicom.Dataset, attr: str) -> str:
    val = getattr(ds, attr, "")
    if val is None:
        return ""
    return str(val).strip()


def _multi_to_first(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if hasattr(val, "__iter__") and not isinstance(val, str):
            return float(val[0])
        return float(val)
    except Exception:
        return None


def get_default_window(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    wc = _multi_to_first(getattr(ds, "WindowCenter", None))
    ww = _multi_to_first(getattr(ds, "WindowWidth", None))
    return wc, ww


def get_study_info(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Coleta informacoes resumidas do estudo (paciente/estudo/serie) para exibir no UI e compor prompt."""
    return {
        "StudyInstanceUID": _get_str(ds, "StudyInstanceUID") or "Unknown",
        "StudyDescription": _get_str(ds, "StudyDescription") or "Unknown",
        "StudyDate": _get_str(ds, "StudyDate") or "Unknown",
        "PatientID": _get_str(ds, "PatientID") or "Unknown",
        "BodyPartExamined": _get_str(ds, "BodyPartExamined"),
        "Manufacturer": _get_str(ds, "Manufacturer"),
        "InstitutionName": _get_str(ds, "InstitutionName"),
        "Modality": get_modality(ds),
    }


def _is_localizer_like(text: str) -> bool:
    t = (text or "").upper()
    needles = [
        "LOCALIZER",
        "SCOUT",
        "SURVEY",
        "TOP",
        "LOC",
    ]
    return any(n in t for n in needles)


def is_localizer_series(series_files: List[Tuple[str, pydicom.Dataset]]) -> bool:
    """Heuristica para identificar series do tipo localizer/scout (geralmente pouco util para laudo)."""
    if not series_files:
        return False

    # SeriesDescription / ImageType are the strongest hints.
    for _, ds in series_files[:10]:
        sd = _get_str(ds, "SeriesDescription")
        if _is_localizer_like(sd):
            return True

        image_type = getattr(ds, "ImageType", None)
        if image_type is not None:
            try:
                parts = [str(x).upper() for x in image_type]
                if any(_is_localizer_like(p) for p in parts):
                    return True
            except Exception:
                pass

    return False


# ---------------------------------------------------------------------------
# Windowing + image conversion
# ---------------------------------------------------------------------------

def apply_windowing(
    pixel_array: np.ndarray,
    ds: pydicom.Dataset,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> np.ndarray:
    """Aplica rescale slope/intercept e windowing no pixel array, devolvendo uint8."""
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)

    arr = pixel_array.astype(np.float32) * slope + intercept

    if window_center is None or window_width is None:
        default_wc, default_ww = get_default_window(ds)
        if window_center is None:
            window_center = default_wc
        if window_width is None:
            window_width = default_ww

    if window_center is not None and window_width is not None and window_width > 0:
        min_val = window_center - window_width / 2.0
        max_val = window_center + window_width / 2.0
        arr = np.clip(arr, min_val, max_val)
        out = ((arr - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        return out

    # Fallback: normalização simples
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi > lo:
        out = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return out
    return np.zeros_like(arr, dtype=np.uint8)


def dicom_to_pil(
    ds: pydicom.Dataset,
    size: Tuple[int, int],
    window_center: Optional[float],
    window_width: Optional[float],
) -> Image.Image:
    """Converte um dataset DICOM (uma fatia) em imagem PIL, aplicando windowing e normalizacao."""
    pixel_array = ds.pixel_array
    normalized = apply_windowing(pixel_array, ds, window_center, window_width)

    # Trata formatos comuns
    if normalized.ndim == 2:
        pil = Image.fromarray(normalized, mode="L")
    elif normalized.ndim == 3:
        # Pode ser multi-frame OU colorido. Heurística:
        # - se a última dimensão <= 4: parece cor
        # - senão: multi-frame -> pega o frame do meio
        if normalized.shape[-1] in (1, 3, 4):
            if normalized.shape[-1] == 1:
                pil = Image.fromarray(normalized[..., 0], mode="L")
            else:
                pil = Image.fromarray(normalized[..., :3], mode="RGB")
        else:
            mid = normalized.shape[0] // 2
            pil = Image.fromarray(normalized[mid], mode="L")
    else:
        # Formato inesperado; tenta achatar e seguir
        pil = Image.fromarray(normalized.reshape(normalized.shape[-2], normalized.shape[-1]), mode="L")

    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    return pil.resize(size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Series organization + sampling
# ---------------------------------------------------------------------------

def organize_by_series(dicom_files: List[Tuple[str, pydicom.Dataset]]) -> Dict[str, List[Tuple[str, pydicom.Dataset]]]:
    """Agrupa datasets DICOM por SeriesInstanceUID, mantendo metadados relevantes."""
    series_dict: Dict[str, List[Tuple[str, pydicom.Dataset]]] = {}
    for filename, ds in dicom_files:
        series_uid = _get_str(ds, "SeriesInstanceUID") or "Unknown"
        series_dict.setdefault(series_uid, []).append((filename, ds))
    return series_dict


def sort_slices_by_position(series_files: List[Tuple[str, pydicom.Dataset]]) -> List[Tuple[str, pydicom.Dataset]]:
    """Ordena fatias pela posicao (ImagePositionPatient) ou InstanceNumber como fallback."""
    def key(item: Tuple[str, pydicom.Dataset]):
        filename, ds = item
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                return (0, int(inst))
            except Exception:
                pass

        slice_loc = getattr(ds, "SliceLocation", None)
        if slice_loc is not None:
            try:
                return (1, float(slice_loc))
            except Exception:
                pass

        # Fallback: filename ordering
        return (2, filename)

    return sorted(series_files, key=key)


def sample_evenly(items: List[Any], k: int) -> List[Any]:
    """Seleciona N indices distribuindo uniformemente ao longo da serie."""
    if k <= 0:
        return []
    if len(items) <= k:
        return items
    if k == 1:
        return [items[len(items) // 2]]

    # k índices em [0, len-1]
    idxs = [int(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idxs]


def _slice_variance_score(ds: pydicom.Dataset, window_center: Optional[float], window_width: Optional[float]) -> float:
    try:
        arr = ds.pixel_array
        arr_u8 = apply_windowing(arr, ds, window_center, window_width)
        # Downsample cheaply to ~64x64 via slicing
        step_r = max(1, arr_u8.shape[0] // 64)
        step_c = max(1, arr_u8.shape[1] // 64)
        small = arr_u8[::step_r, ::step_c].astype(np.float32)
        return float(np.var(small))
    except Exception:
        return 0.0


def sample_smart_best_of_3_per_bin(
    series_slices: List[Tuple[str, pydicom.Dataset]],
    k: int,
    window_center: Optional[float],
    window_width: Optional[float],
) -> List[Tuple[str, pydicom.Dataset]]:
    """
    Amostragem "smart" sem decodificar o volume inteiro:

    Divide em k faixas; em cada faixa avalia 3 candidatos (início/meio/fim) e
    escolhe a fatia com maior variância após aplicar windowing.
    """
    n = len(series_slices)
    if k <= 0:
        return []
    if n <= k:
        return series_slices

    # Bordas das faixas (bins)
    edges = np.linspace(0, n, num=k + 1, dtype=int)

    out: List[Tuple[str, pydicom.Dataset]] = []
    for i in range(k):
        start = int(edges[i])
        end = int(edges[i + 1] - 1)
        end = max(start, min(end, n - 1))
        mid = (start + end) // 2

        candidates = sorted(set([start, mid, end]))
        best_idx = max(
            candidates,
            key=lambda idx: _slice_variance_score(series_slices[idx][1], window_center, window_width),
        )
        out.append(series_slices[best_idx])

    return out


def summarize_series(series_uid: str, series_files: List[Tuple[str, pydicom.Dataset]]) -> Dict[str, Any]:
    """Gera um resumo por serie (modality, numero de imagens, descricao, etc.)."""
    if not series_files:
        return {
            "SeriesInstanceUID": series_uid,
            "SeriesDescription": "Unknown",
            "NumSlices": 0,
            "IsLocalizer": False,
        }

    first_ds = series_files[0][1]
    desc = _get_str(first_ds, "SeriesDescription") or f"Series {(_get_str(first_ds, 'SeriesNumber') or '').strip()}"
    protocol = _get_str(first_ds, "ProtocolName")
    body_part = _get_str(first_ds, "BodyPartExamined")
    series_num = _get_str(first_ds, "SeriesNumber")
    is_loc = is_localizer_series(series_files)

    return {
        "SeriesInstanceUID": series_uid,
        "SeriesDescription": desc or "Unknown",
        "ProtocolName": protocol,
        "BodyPartExamined": body_part,
        "SeriesNumber": series_num,
        "NumSlices": len(series_files),
        "IsLocalizer": bool(is_loc),
    }


def build_series_summary_text(series_summaries: List[Dict[str, Any]], selected_uids: Optional[List[str]]) -> str:
    if not series_summaries:
        return ""

    parts = []
    selected = set(selected_uids or [])
    for s in sorted(series_summaries, key=lambda x: int(x.get("NumSlices") or 0), reverse=True)[:8]:
        desc = str(s.get("SeriesDescription") or "Unknown")
        n = int(s.get("NumSlices") or 0)
        mark = "✓" if (s.get("SeriesInstanceUID") in selected) else " "
        loc = " (localizer)" if s.get("IsLocalizer") else ""
        parts.append(f"[{mark}] {desc} — {n} slices{loc}")
    return "; ".join(parts)


def process_dicom_study(
    zip_bytes: bytes,
    *,
    max_slices_total: int = 500,
    max_slices_per_series: Optional[int] = None,
    image_size: int = 896,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    sampling_strategy: str = "even",
    exclude_localizers: bool = True,
    selected_series_uids: Optional[List[str]] = None,
) -> Tuple[str, List[Image.Image], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Processa um estudo DICOM a partir de um ZIP.

    Retorna:
      modality, images, study_info, series_summaries
    """
    dicom_files = extract_dicom_from_zip(zip_bytes)
    if not dicom_files:
        raise ValueError("No valid DICOM images found in the ZIP archive")

    first_ds = dicom_files[0][1]
    modality = get_modality(first_ds)

    default_wc, default_ww = get_default_window(first_ds)

    series_dict = organize_by_series(dicom_files)
    series_summaries = [summarize_series(uid, files) for uid, files in series_dict.items()]

    # Determine default selection (UI may override)
    if selected_series_uids is None:
        selected_series_uids = [
            s["SeriesInstanceUID"]
            for s in series_summaries
            if int(s.get("NumSlices") or 0) >= 3 and (not exclude_localizers or not s.get("IsLocalizer"))
        ]

    # Filter series
    filtered_series: Dict[str, List[Tuple[str, pydicom.Dataset]]] = {}
    for uid, files in series_dict.items():
        if uid not in set(selected_series_uids):
            continue
        if exclude_localizers and is_localizer_series(files):
            continue
        filtered_series[uid] = files

    if not filtered_series:
        # fallback: se tudo foi filtrado, volta para todas as séries
        filtered_series = series_dict

    total_original_slices = sum(len(files) for files in series_dict.values())
    total_selected_slices = sum(len(files) for files in filtered_series.values())

    # Amostragem
    sampled_slices: List[Tuple[str, pydicom.Dataset]] = []

    strategy = (sampling_strategy or "even").lower().strip()
    use_smart = strategy.startswith("smart")

    if max_slices_per_series is not None and max_slices_per_series > 0:
        for _, files in filtered_series.items():
            sorted_slices = sort_slices_by_position(files)
            if use_smart:
                sampled = sample_smart_best_of_3_per_bin(sorted_slices, max_slices_per_series, window_center, window_width)
            else:
                sampled = sample_evenly(sorted_slices, max_slices_per_series)
            sampled_slices.extend(sampled)
    else:
        # Amostragem global sobre as séries selecionadas
        all_sorted: List[Tuple[str, pydicom.Dataset]] = []
        for _, files in filtered_series.items():
            all_sorted.extend(sort_slices_by_position(files))
        if use_smart:
            sampled_slices = sample_smart_best_of_3_per_bin(all_sorted, max_slices_total, window_center, window_width)
        else:
            sampled_slices = sample_evenly(all_sorted, max_slices_total)

    images: List[Image.Image] = []
    size = (int(image_size), int(image_size))
    for filename, ds in sampled_slices:
        try:
            images.append(
                dicom_to_pil(ds, size=size, window_center=window_center, window_width=window_width)
            )
        except Exception as e:
            # Segue o baile: uma imagem ruim não deve derrubar o estudo inteiro.
            print(f"Erro ao converter {filename}: {e}")

    study_info = get_study_info(first_ds)
    study_info.update(
        {
            "SeriesCount": len(series_dict),
            "SelectedSeriesCount": len(filtered_series),
            "TotalOriginalSlices": total_original_slices,
            "TotalSelectedSlices": total_selected_slices,
            "SampledSlices": len(sampled_slices),
            "ProcessedImages": len(images),
            "ImageSize": int(image_size),
            "DefaultWindowCenter": default_wc,
            "DefaultWindowWidth": default_ww,
            "SamplingStrategy": sampling_strategy,
            "ExcludeLocalizers": bool(exclude_localizers),
            "SelectedSeriesUIDs": selected_series_uids,
        }
    )
    if max_slices_per_series is not None and max_slices_per_series > 0:
        study_info["MaxSlicesPerSeries"] = int(max_slices_per_series)
    else:
        study_info["MaxSlicesTotal"] = int(max_slices_total)

    study_info["SeriesSummary"] = build_series_summary_text(series_summaries, selected_series_uids)

    return modality, images, study_info, series_summaries
