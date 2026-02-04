"""Aplicação principal (Gradio) para rascunho de laudos a partir de estudos DICOM (MedGemma 1.5).

Este fork inclui melhorias voltadas para uso real em produção:
- Seleção de séries + exclusão de localizer/scout.
- Amostragem "smart" para reduzir VRAM sem perder representatividade.
- Pipeline "chunked map-reduce" para estudos grandes (evita estouro de VRAM).
- Passo opcional de refinamento (somente texto) para padronizar o laudo.
- Sugestão automática de parâmetros (auto-fit) baseada na VRAM disponível.
- Reload do modelo com quantização (4/8-bit) quando bitsandbytes estiver disponível.

Observação importante: em Hugging Face Spaces, o import `spaces` deve ocorrer ANTES de torch/transformers.
"""


from __future__ import annotations

# O app Gradio orquestra:
# 1) upload (zip/dicom),
# 2) processamento/preview (séries + amostragem),
# 3) execução do pipeline de laudo,
# 4) exibição de debug/estimativas.

# IMPORTANTE: no Hugging Face Spaces, importe `spaces` ANTES de torch/transformers.
try:
    import spaces  # type: ignore

    SPACES_AVAILABLE = True
except Exception:
    SPACES_AVAILABLE = False

import hashlib
import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

from dicom_processor import mask_patient_id, process_dicom_study
from model_handler import (
    GenerationParams,
    ModelLoadConfig,
    ModelManager,
    PipelineMode,
    generate_report_pipeline,
)
from reporting import (
    Language,
    ReportTemplate,
    build_debug_footer,
    sanitize_findings,
    sanitize_phi_text,
)
from vram import (
    QuantizationMode,
    format_vram_summary,
    get_gpu_info,
    estimate_total_vram_gb,
    suggest_processing_params,
)

# Desativa TF32 para evitar erros do tipo CUBLAS_STATUS_INVALID_VALUE em alguns formatos/GPUs.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass


# -----------------------------------------------------------------------------
# Carregamento global do modelo (útil em HF Spaces / ZeroGPU)
# -----------------------------------------------------------------------------

MODEL_ID = os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it")
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_MANAGER = ModelManager(model_id=MODEL_ID)

# Por padrão: em Spaces, pré-carrega; fora de Spaces, deixa para carregar sob demanda.
_autoload_default = "1" if SPACES_AVAILABLE else "0"
AUTOLOAD_MODEL = os.getenv("AUTOLOAD_MODEL", _autoload_default).strip()

if AUTOLOAD_MODEL == "1":
    print("=" * 60)
    print("Carregando o modelo MedGemma na inicialização...")
    try:
        MODEL_MANAGER.ensure_loaded(hf_token=HF_TOKEN)
        print("Modelo carregado.")
        print(json.dumps(MODEL_MANAGER.describe(), indent=2))
    except Exception as e:
        # Mantém o app no ar; o usuário pode clicar em "Recarregar modelo" e ver o erro.
        print("Falha ao carregar o modelo na inicialização:", repr(e))
        print(traceback.format_exc())
    print("=" * 60)
else:
    print("Inicialização: carregamento do modelo desativado (AUTOLOAD_MODEL=0).")


# -----------------------------------------------------------------------------
# Cache: processed study to avoid re-reading/reprocessing on each generation
# -----------------------------------------------------------------------------

@dataclass
class CachedStudy:
    zip_sha1: Optional[str] = None
    zip_bytes: Optional[bytes] = None

    processing_key: Optional[str] = None
    modality: Optional[str] = None
    images: Optional[List[Image.Image]] = None
    study_info: Optional[Dict[str, Any]] = None
    series_summaries: Optional[List[Dict[str, Any]]] = None


CACHE = CachedStudy()


def _sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def _make_processing_key(
    zip_sha1: str,
    *,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool,
    sampling_strategy: str,
    exclude_localizers: bool,
    selected_series_uids: Optional[List[str]],
) -> str:
    selected = [str(x) for x in (selected_series_uids or []) if str(x).strip()]
    selected_mode = "manual" if selected else "auto"
    selected = sorted(set(selected))

    params = {
        "max_slices_per_series": int(max_slices_per_series),
        "image_size": int(image_size),
        "window_center": float(window_center),
        "window_width": float(window_width),
        "use_auto_window": bool(use_auto_window),
        "sampling_strategy": str(sampling_strategy),
        "exclude_localizers": bool(exclude_localizers),
        "selected_series_mode": selected_mode,
        "selected_series_uids": selected,
    }
    return f"{zip_sha1}:{json.dumps(params, sort_keys=True)}"


def _read_zip_file(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()


def _format_study_info_text(
    *,
    modality: str,
    study_info: Dict[str, Any],
    model_desc: Dict[str, Any],
    quantization: QuantizationMode,
    dtype: str,
    est_total_gb: float,
) -> str:
    # Por padrão, mostramos dados "PHI-safe" (mascarados).
    patient_id_raw = str(study_info.get("PatientID") or "Desconhecido")
    patient_id_masked = mask_patient_id(patient_id_raw)

    wc = study_info.get("DefaultWindowCenter", "N/A")
    ww = study_info.get("DefaultWindowWidth", "N/A")

    series_summary = str(study_info.get("SeriesSummary") or "")

    gpu = get_gpu_info()
    vram_summary = format_vram_summary(gpu=gpu, est_total_gb=est_total_gb)

    lines = [
        "Informações do estudo (exibição segura / PHI-safe)",
        f"Modalidade: {modality}",
        f"Descrição do estudo: {study_info.get('StudyDescription', 'Desconhecido')}",
        f"Data do estudo: {study_info.get('StudyDate', 'Desconhecida')}",
        f"ID do paciente: {patient_id_masked}",
        f"Região: {study_info.get('BodyPartExamined', '')}",
        "",
        f"Qtde. de séries: {study_info.get('SeriesCount', 'N/A')} | Selecionadas: {study_info.get('SelectedSeriesCount', 'N/A')}",
        f"Total de slices (originais): {study_info.get('TotalOriginalSlices', 'N/A')} | Selecionadas: {study_info.get('TotalSelectedSlices', 'N/A')}",
        f"Amostragem: {study_info.get('SamplingStrategy', 'even')} | Excluir localizers: {study_info.get('ExcludeLocalizers', True)}",
    ]
    if "MaxSlicesPerSeries" in study_info:
        lines.append(f"Máx. de slices por série: {study_info.get('MaxSlicesPerSeries')}")
    lines.append(f"Imagens processadas: {study_info.get('ProcessedImages', 0)}")
    lines.append(f"Tamanho das imagens: {study_info.get('ImageSize', 'N/A')}x{study_info.get('ImageSize', 'N/A')}")
    lines.append(f"Janela padrão (do DICOM): WC={wc}, WW={ww}")
    if series_summary:
        lines.append("")
        lines.append("Resumo das séries (top):")
        lines.append(series_summary)

    lines.extend(
        [
            "",
            "Modelo",
            f"Model ID: {model_desc.get('model_id', MODEL_ID)}",
            f"Quantização: {quantization}",
            f"DType: {dtype}",
            f"Device: {model_desc.get('param_device', 'desconhecido')}",
            "",
            "VRAM / Memória",
            vram_summary,
        ]
    )

    return "\n".join(str(x) for x in lines).strip()


def _get_model_runtime_settings(
    quantization_ui: QuantizationMode,
    dtype_ui: str,
    attn_impl_ui: str,
) -> Tuple[QuantizationMode, str, Optional[str]]:
    q = str(quantization_ui).lower().strip()
    if q not in ("auto", "none", "8bit", "4bit"):
        q = "auto"
    d = str(dtype_ui).lower().strip()
    if d == "auto" or not d:
        d = os.getenv("MODEL_DTYPE") or "auto"
    a = str(attn_impl_ui).lower().strip()
    if a == "auto" or not a:
        a = None
    return q, d, a


def reload_model(
    quantization_ui: QuantizationMode,
    dtype_ui: str,
    attn_impl_ui: str,
) -> str:
    try:
        q, d, a = _get_model_runtime_settings(quantization_ui, dtype_ui, attn_impl_ui)
        cfg = ModelLoadConfig(
            model_id=MODEL_ID,
            quantization=q,
            device_map="auto",
            dtype="auto" if d == "auto" else d,
            attn_implementation=a,
        )
        MODEL_MANAGER.ensure_loaded(cfg, hf_token=HF_TOKEN)
        return json.dumps(MODEL_MANAGER.describe(), indent=2)
    except Exception as e:
        return f"Falha ao recarregar o modelo: {e}\n\n{traceback.format_exc()}"


def process_dicom_file(
    file_path: str,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool,
    sampling_strategy: str,
    exclude_localizers: bool,
    selected_series_uids: Optional[List[str]],
    quantization_ui: QuantizationMode,
    dtype_ui: str,
) -> Tuple[str, str, List[Image.Image], Any]:
    """
    Processa um ZIP DICOM e devolve imagens de preview.

    Retorna: status, info_text, images, series_selector_update
    """
    global CACHE

    try:
        if not file_path:
            return "No file uploaded.", "", [], gr.update(choices=[], value=[])

        zip_bytes = _read_zip_file(file_path)
        zip_sha1 = _sha1_bytes(zip_bytes)

        wc = None if use_auto_window else float(window_center)
        ww = None if use_auto_window else float(window_width)

        modality, images, study_info, series_summaries = process_dicom_study(
            zip_bytes,
            max_slices_per_series=int(max_slices_per_series) if int(max_slices_per_series) > 0 else None,
            image_size=int(image_size),
            window_center=wc,
            window_width=ww,
            sampling_strategy=str(sampling_strategy),
            exclude_localizers=bool(exclude_localizers),
            selected_series_uids=selected_series_uids or None,
        )

        # Usa a seleção real aplicada pelo processador (pode incluir defaults).
        actual_selected_series_uids = list(study_info.get("SelectedSeriesUIDs") or [])

        # Cache para reaproveitar na geração (evita reprocessar o ZIP a cada clique).
        processing_key = _make_processing_key(
            zip_sha1,
            max_slices_per_series=int(max_slices_per_series),
            image_size=int(image_size),
            window_center=float(window_center),
            window_width=float(window_width),
            use_auto_window=bool(use_auto_window),
            sampling_strategy=str(sampling_strategy),
            exclude_localizers=bool(exclude_localizers),
            selected_series_uids=sorted(set(actual_selected_series_uids)),
        )

        CACHE = CachedStudy(
            zip_sha1=zip_sha1,
            zip_bytes=zip_bytes,
            processing_key=processing_key,
            modality=modality,
            images=images,
            study_info=study_info,
            series_summaries=series_summaries,
        )

        # Atualiza o seletor de séries (label, value=UID).
        choices: List[Tuple[str, str]] = []
        for s in sorted(series_summaries, key=lambda x: int(x.get("NumSlices") or 0), reverse=True):
            uid = str(s.get("SeriesInstanceUID"))
            desc = str(s.get("SeriesDescription") or "Desconhecida")
            n = int(s.get("NumSlices") or 0)
            loc = " (localizer)" if s.get("IsLocalizer") else ""
            choices.append((f"{desc} — {n} slices{loc}", uid))

        # Seleção padrão: usa o que o processador escolheu (ou o que o usuário marcou).
        default_selected = actual_selected_series_uids

        # VRAM estimate
        q, d, _ = _get_model_runtime_settings(quantization_ui, dtype_ui, "auto")
        model_desc = MODEL_MANAGER.describe()
        est_total = estimate_total_vram_gb(
            num_images=int(study_info.get("ProcessedImages") or 0),
            image_size=int(study_info.get("ImageSize") or image_size),
            quantization=q,
            dtype=("fp16" if d == "auto" else d),
        )

        info_text = _format_study_info_text(
            modality=modality,
            study_info=study_info,
            model_desc=model_desc,
            quantization=q,
            dtype=("auto" if d == "auto" else d),
            est_total_gb=est_total,
        )

        status = f"Processed: {len(images)} images ({modality})"

        return status, info_text, images, gr.update(choices=choices, value=default_selected)

    except Exception as e:
        error_msg = f"Erro ao processar o DICOM: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, "", [], gr.update(choices=[], value=[])


def auto_fit_to_vram(
    target_fraction: float,
    quantization_ui: QuantizationMode,
    dtype_ui: str,
    selected_series_uids: Optional[List[str]],
) -> Tuple[Any, Any, str]:
    """
    Sugere (image_size, max_slices_per_series) com base na VRAM disponível.

    Retorna updates para sliders + uma mensagem de status.
    """
    gpu = get_gpu_info()
    if not gpu.available:
        return gr.update(), gr.update(), "Auto-fit: nenhuma GPU detectada."

    if CACHE.study_info is None:
        return gr.update(), gr.update(), "Auto-fit: processe um estudo primeiro."

    series_count = len(selected_series_uids or []) or int(CACHE.study_info.get("SelectedSeriesCount") or 0) or 1

    target_vram = max(1.0, gpu.total_gb * float(target_fraction))

    q, d, _ = _get_model_runtime_settings(quantization_ui, dtype_ui, "auto")
    dtype = "fp16" if d == "auto" else d

    suggestion = suggest_processing_params(
        series_count=series_count,
        target_vram_gb=target_vram,
        quantization=q,
        dtype=dtype,
        min_slices_per_series=4,
        max_slices_per_series_cap=50,
    )

    if suggestion is None:
        return (
            gr.update(),
            gr.update(),
            f"Auto-fit: não encontrei uma combinação que caiba em {target_vram:.1f} GB. Tente baixar o alvo.",
        )

    msg = (
        f"Auto-fit (alvo {target_vram:.1f} GB): "
        f"image_size={suggestion.image_size}, max_slices_per_series={suggestion.max_slices_per_series} "
        f"(estimativa ~{suggestion.estimated_total_vram_gb:.1f} GB)."
    )
    return (
        gr.update(value=int(suggestion.max_slices_per_series)),
        gr.update(value=int(suggestion.image_size)),
        msg,
    )


def _generate_report_impl(
    file_path: str,
    max_slices_per_series: int,
    image_size: int,
    window_center: float,
    window_width: float,
    use_auto_window: bool,
    sampling_strategy: str,
    exclude_localizers: bool,
    selected_series_uids: Optional[List[str]],
    language: Language,
    template: ReportTemplate,
    clinical_history: str,
    additional_instructions: str,
    prompt_override: str,
    pipeline_mode: str,
    chunk_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    append_debug: bool,
    sanitize_phi: bool,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Gera um rascunho de laudo radiológico.

    Saídas:
      - report text
      - avisos (QC) + debug
      - caminho do .txt
      - caminho do .json de achados (se houver)
    """
    global CACHE

    try:
        if not file_path:
            return "Envie um ZIP DICOM antes de gerar o laudo.", "", None, None

        # Garante que o modelo esteja carregado.
        if MODEL_MANAGER.loaded_config is None:
            MODEL_MANAGER.ensure_loaded(hf_token=HF_TOKEN)

        # Sempre lê o arquivo atual para evitar usar cache de um upload anterior por engano.
        zip_bytes = _read_zip_file(file_path)
        zip_sha1 = _sha1_bytes(zip_bytes)
        if CACHE.zip_sha1 == zip_sha1 and CACHE.zip_bytes is not None:
            zip_bytes = CACHE.zip_bytes

        selected_for_key = [str(x) for x in (selected_series_uids or []) if str(x).strip()]
        selected_for_key = sorted(set(selected_for_key))

        processing_key = _make_processing_key(
            zip_sha1,
            max_slices_per_series=int(max_slices_per_series),
            image_size=int(image_size),
            window_center=float(window_center),
            window_width=float(window_width),
            use_auto_window=bool(use_auto_window),
            sampling_strategy=str(sampling_strategy),
            exclude_localizers=bool(exclude_localizers),
            selected_series_uids=selected_for_key or None,
        )

        if CACHE.processing_key == processing_key and CACHE.images is not None and CACHE.modality and CACHE.study_info:
            modality = CACHE.modality
            images = CACHE.images
            study_info = CACHE.study_info
        else:
            wc = None if use_auto_window else float(window_center)
            ww = None if use_auto_window else float(window_width)
            modality, images, study_info, series_summaries = process_dicom_study(
                zip_bytes,
                max_slices_per_series=int(max_slices_per_series) if int(max_slices_per_series) > 0 else None,
                image_size=int(image_size),
                window_center=wc,
                window_width=ww,
                sampling_strategy=str(sampling_strategy),
                exclude_localizers=bool(exclude_localizers),
                selected_series_uids=selected_series_uids or None,
            )
            CACHE = CachedStudy(
                zip_sha1=zip_sha1,
                zip_bytes=zip_bytes,
                processing_key=processing_key,
                modality=modality,
                images=images,
                study_info=study_info,
                series_summaries=series_summaries,
            )

        # Pipeline mode normalization
        pm = str(pipeline_mode).lower().strip()
        pipeline: PipelineMode
        if pm.startswith("single") and "refine" in pm:
            pipeline = "single_refine"
        elif pm.startswith("single"):
            pipeline = "single"
        elif pm.startswith("chunk") and "refine" in pm:
            pipeline = "chunked_refine"
        else:
            pipeline = "chunked"

        gen = GenerationParams(
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=bool(do_sample),
        )

        # Sanitiza entradas antes de enviar para o modelo (reduz risco de PHI indo pro prompt).
        if sanitize_phi:
            clinical_history, _ = sanitize_phi_text(clinical_history or "", language=language)
            additional_instructions, _ = sanitize_phi_text(additional_instructions or "", language=language)
            prompt_override, _ = sanitize_phi_text(prompt_override or "", language=language)

            safe_study_info = dict(study_info or {})
            for k in ("StudyDescription", "BodyPartExamined", "SeriesSummary"):
                if k in safe_study_info and safe_study_info.get(k):
                    safe_study_info[k], _ = sanitize_phi_text(str(safe_study_info.get(k)), language=language)
        else:
            safe_study_info = study_info

        result = generate_report_pipeline(
            manager=MODEL_MANAGER,
            images=images,
            modality=modality,
            study_info=safe_study_info,
            language=language,
            template=template,
            clinical_history=clinical_history or "",
            additional_instructions=additional_instructions or "",
            prompt_override=prompt_override or "",
            pipeline=pipeline,
            chunk_size=int(chunk_size),
            gen=gen,
        )

        report = result.report_text
        sanitizer_warnings: List[str] = []
        if sanitize_phi:
            report, sanitizer_warnings = sanitize_phi_text(report, language=language)
        if append_debug:
            report = report + "\n" + build_debug_footer(result.debug)

        merged_warnings: List[str] = []
        if sanitizer_warnings:
            merged_warnings.extend(sanitizer_warnings)
        if result.qc_issues:
            merged_warnings.extend(list(result.qc_issues))

        # Save outputs
        txt_path = None
        json_path = None

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tf.write(report.encode("utf-8"))
        tf.flush()
        tf.close()
        txt_path = tf.name

        if result.extracted_findings is not None:
            findings_to_save = result.extracted_findings
            if sanitize_phi:
                findings_to_save, fw = sanitize_findings(findings_to_save, language=language)
                for w in fw:
                    if w not in merged_warnings:
                        merged_warnings.append(w)
            jf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            jf.write(json.dumps({"findings": findings_to_save}, ensure_ascii=False, indent=2).encode("utf-8"))
            jf.flush()
            jf.close()
            json_path = jf.name

        merged_warnings = list(dict.fromkeys(merged_warnings))

        qc_lines: List[str] = []
        if merged_warnings:
            qc_lines.append("QC / Avisos:")
            qc_lines.extend([f"- {x}" for x in merged_warnings])

        qc_lines.append("")
        qc_lines.append("Modelo:")
        qc_lines.append(json.dumps(MODEL_MANAGER.describe(), indent=2))
        qc_lines.append("")
        qc_lines.append("Debug do pipeline:")
        qc_lines.append(json.dumps(result.debug, indent=2))

        qc_text = "\n".join(qc_lines).strip()

        return report, qc_text, txt_path, json_path

    except Exception as e:
        error_msg = f"Erro ao gerar o laudo: {e}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, error_msg, None, None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Aplica o decorator @spaces.GPU quando estiver rodando em Hugging Face Spaces.
if SPACES_AVAILABLE:

    @spaces.GPU(duration=180)  # type: ignore
    def generate_report(*args, **kwargs):
        return _generate_report_impl(*args, **kwargs)

else:

    def generate_report(*args, **kwargs):
        return _generate_report_impl(*args, **kwargs)


def create_interface():
    with gr.Blocks(title="Gerador de Laudo DICOM (MedGemma 1.5) — Enhanced", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Gerador de Laudo DICOM (MedGemma 1.5) — Enhanced")
        gr.Markdown(
            "Envie um arquivo ZIP com imagens DICOM para gerar um rascunho de laudo estruturado. "
            "**Somente para pesquisa/educação — não use para decisão clínica.**"
        )

        with gr.Row():
            # -----------------------------------------------------------------
            # Coluna esquerda: upload e processamento
            # -----------------------------------------------------------------
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Enviar ZIP DICOM",
                    file_types=[".zip"],
                    type="filepath",
                )

                with gr.Accordion("Processamento de imagens", open=True):
                    sampling_strategy = gr.Dropdown(
                        label="Estratégia de amostragem",
                        choices=["even", "smart"],
                        value="even",
                        info="smart = melhor de 3 por faixa (tende a representar melhor sem decodificar tudo)",
                    )
                    exclude_localizers = gr.Checkbox(
                        label="Excluir localizers / scouts (recomendado)",
                        value=True,
                    )
                    series_selector = gr.CheckboxGroup(
                        label="Seleção de séries",
                        choices=[],
                        value=[],
                        info="Clique em 'Processar & pré-visualizar' para listar. Por padrão, seleciona séries não-localizer com >=3 slices.",
                    )

                    max_slices_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Máx. de slices por série",
                        info="0 = amostragem global (não recomendado com muitas séries). Reduza para economizar VRAM.",
                    )
                    image_size_slider = gr.Slider(
                        minimum=224,
                        maximum=1024,
                        value=512,
                        step=32,
                        label="Tamanho da imagem",
                        info="Menor = menos VRAM, menos detalhe visual",
                    )

                    gr.Markdown("**Windowing (TC / raio-X)**")
                    use_auto_window = gr.Checkbox(
                        label="Usar janela automática (metadados DICOM)",
                        value=True,
                    )
                    with gr.Row():
                        window_center_slider = gr.Slider(
                            minimum=-1000,
                            maximum=3000,
                            value=40,
                            step=10,
                            label="Window Center (WC)",
                        )
                        window_width_slider = gr.Slider(
                            minimum=1,
                            maximum=4000,
                            value=400,
                            step=10,
                            label="Window Width (WW)",
                        )

                    with gr.Row():
                        process_btn = gr.Button("Processar & pré-visualizar", variant="primary")
                        auto_fit_btn = gr.Button("Auto-fit para VRAM", variant="secondary")

                    auto_fit_fraction = gr.Slider(
                        minimum=0.5,
                        maximum=0.95,
                        value=0.85,
                        step=0.05,
                        label="Alvo do auto-fit (% da VRAM total)",
                        info="O auto-fit usa VRAM total (não a livre). Reduza se tiver outros apps consumindo VRAM.",
                    )

                status_output = gr.Textbox(label="Status", interactive=False)
                study_info_box = gr.Textbox(
                    label="Informações do estudo & estimativa de memória",
                    interactive=False,
                    lines=18,
                )

                with gr.Accordion("Avançado: modelo (recarregar se precisar)", open=False):
                    quantization = gr.Dropdown(
                        label="Quantização",
                        choices=["auto", "none", "8bit", "4bit"],
                        value=os.getenv("MODEL_QUANTIZATION", "auto"),
                        info="Precisa de bitsandbytes para 4bit/8bit. Se não estiver instalado, volta para none.",
                    )
                    dtype = gr.Dropdown(
                        label="DType",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        value=os.getenv("MODEL_DTYPE", "auto"),
                    )
                    attn_impl = gr.Dropdown(
                        label="Implementação de attention",
                        choices=["auto", "sdpa", "flash_attention_2", "eager"],
                        value="auto",
                        info="Deixe em auto a menos que você saiba que precisa de eager/flash.",
                    )
                    reload_btn = gr.Button("Recarregar modelo", variant="secondary")
                    model_status = gr.Textbox(label="Status do modelo (debug)", interactive=False, lines=8, value=json.dumps(MODEL_MANAGER.describe(), indent=2))

            # -----------------------------------------------------------------
            # Coluna do meio: galeria de preview
            # -----------------------------------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### Pré-visualização")
                gr.Markdown("*Imagens amostradas que serão enviadas ao modelo*")
                image_gallery = gr.Gallery(
                    label="Imagens amostradas",
                    show_label=False,
                    columns=4,
                    rows=3,
                    height=420,
                    object_fit="contain",
                    preview=True,
                )

            # -----------------------------------------------------------------
            # Coluna direita: geração e saída
            # -----------------------------------------------------------------
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    label="Idioma do laudo",
                    choices=["English", "Português (Brasil)", "Español"],
                    value="Português (Brasil)",
                )
                template = gr.Dropdown(
                    label="Template do laudo",
                    choices=["General", "Chest X-ray", "CT (general)", "MRI (general)"],
                    value="General",
                )
                clinical_history = gr.Textbox(
                    label="História clínica / indicação (opcional)",
                    lines=2,
                    placeholder="Ex.: febre e tosse; afastar pneumonia...",
                )
                additional_instructions = gr.Textbox(
                    label="Instruções adicionais (opcional)",
                    lines=2,
                    placeholder="Ex.: seja objetivo; foque em pulmões; impressão numerada...",
                )
                prompt_override = gr.Textbox(
                    label="Prompt personalizado (avançado; opcional)",
                    lines=3,
                    placeholder="Deixe em branco para usar o gerador de prompt estruturado.",
                )

                pipeline_mode = gr.Radio(
                    label="Modo do pipeline",
                    choices=[
                        "single",
                        "single + refine (recomendado se a qualidade variar)",
                        "chunked (map-reduce, recomendado para estudos grandes)",
                        "chunked + refine (melhor qualidade em estudos grandes)",
                    ],
                    value="chunked (map-reduce, recomendado para estudos grandes)",
                )
                chunk_size = gr.Slider(
                    minimum=4,
                    maximum=32,
                    value=16,
                    step=1,
                    label="Tamanho do chunk (imagens por inferência) — só no modo chunked",
                )

                with gr.Accordion("Configurações de geração", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=50,
                        maximum=1200,
                        value=350,
                        step=10,
                        label="Máx. de novos tokens",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperatura",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k",
                    )
                    do_sample_checkbox = gr.Checkbox(
                        label="Ativar sampling",
                        value=True,
                        info="Desmarque para saída determinística.",
                    )

                append_debug = gr.Checkbox(
                    label="Adicionar rodapé de debug no laudo (não-clínico)",
                    value=False,
                )

                sanitize_phi_checkbox = gr.Checkbox(
                    label="Sanitizador de PHI/PII (recomendado)",
                    value=True,
                    info="Redige possíveis identificadores em texto livre (prompt + saída). Heurístico.",
                )

                generate_btn = gr.Button("Gerar laudo", variant="primary", size="lg")

                report_output = gr.Textbox(
                    label="Laudo gerado",
                    interactive=False,
                    lines=18,
                    placeholder="O laudo vai aparecer aqui...",
                )
                qc_output = gr.Textbox(
                    label="QC / Debug",
                    interactive=False,
                    lines=10,
                )

                with gr.Row():
                    download_txt = gr.File(label="Baixar laudo (.txt)")
                    download_findings = gr.File(label="Baixar achados (.json) — modos chunked")

                with gr.Accordion("Presets de janela (TC)", open=False):
                    gr.Markdown("**Clique para aplicar um preset (desativa a janela automática).**")
                    with gr.Row():
                        brain_btn = gr.Button("Cérebro (40/80)", size="sm")
                        subdural_btn = gr.Button("Subdural (75/215)", size="sm")
                        stroke_btn = gr.Button("Stroke (32/8)", size="sm")
                    with gr.Row():
                        lung_btn = gr.Button("Pulmão (-600/1500)", size="sm")
                        mediastinum_btn = gr.Button("Mediastinum (50/350)", size="sm")
                        bone_btn = gr.Button("Bone (400/1800)", size="sm")
                    with gr.Row():
                        abdomen_btn = gr.Button("Abdome (40/400)", size="sm")
                        liver_btn = gr.Button("Liver (60/150)", size="sm")

                # Preset handlers
                def _set_window(wc: float, ww: float):
                    return (wc, ww, False)

                brain_btn.click(lambda: _set_window(40, 80), outputs=[window_center_slider, window_width_slider, use_auto_window])
                subdural_btn.click(lambda: _set_window(75, 215), outputs=[window_center_slider, window_width_slider, use_auto_window])
                stroke_btn.click(lambda: _set_window(32, 8), outputs=[window_center_slider, window_width_slider, use_auto_window])
                lung_btn.click(lambda: _set_window(-600, 1500), outputs=[window_center_slider, window_width_slider, use_auto_window])
                mediastinum_btn.click(lambda: _set_window(50, 350), outputs=[window_center_slider, window_width_slider, use_auto_window])
                bone_btn.click(lambda: _set_window(400, 1800), outputs=[window_center_slider, window_width_slider, use_auto_window])
                abdomen_btn.click(lambda: _set_window(40, 400), outputs=[window_center_slider, window_width_slider, use_auto_window])
                liver_btn.click(lambda: _set_window(60, 150), outputs=[window_center_slider, window_width_slider, use_auto_window])

        # -----------------------------------------------------------------------------
        # Ligações de eventos
        # -----------------------------------------------------------------------------

        process_btn.click(
            fn=process_dicom_file,
            inputs=[
                file_input,
                max_slices_slider,
                image_size_slider,
                window_center_slider,
                window_width_slider,
                use_auto_window,
                sampling_strategy,
                exclude_localizers,
                series_selector,
                quantization,
                dtype,
            ],
            outputs=[status_output, study_info_box, image_gallery, series_selector],
        )

        auto_fit_btn.click(
            fn=auto_fit_to_vram,
            inputs=[auto_fit_fraction, quantization, dtype, series_selector],
            outputs=[max_slices_slider, image_size_slider, status_output],
        )

        reload_btn.click(
            fn=reload_model,
            inputs=[quantization, dtype, attn_impl],
            outputs=[model_status],
        )

        generate_btn.click(
            fn=generate_report,
            inputs=[
                file_input,
                max_slices_slider,
                image_size_slider,
                window_center_slider,
                window_width_slider,
                use_auto_window,
                sampling_strategy,
                exclude_localizers,
                series_selector,
                language,
                template,
                clinical_history,
                additional_instructions,
                prompt_override,
                pipeline_mode,
                chunk_size,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                do_sample_checkbox,
                append_debug,
                sanitize_phi_checkbox,
            ],
            outputs=[report_output, qc_output, download_txt, download_findings],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Modalidades suportadas:** CT, MR, CR, DX. "
            "**Dica:** em estudos muito grandes, use *chunked*; costuma aguentar mais slices sem estourar a VRAM."
        )

    return demo


def main():
    print("Iniciando o Gerador de Laudo DICOM (MedGemma 1.5) — Enhanced...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
