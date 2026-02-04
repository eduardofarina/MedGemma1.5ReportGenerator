"""Carregamento de modelo e inferência (MedGemma 1.5).

Objetivos:
- Centralizar carregamento do modelo (dtype, device, quantização opcional).
- Oferecer pipeline "single-pass" e "chunked map-reduce" para lidar com estudos
  grandes sem estourar VRAM.
- Oferecer um passo opcional de refinamento (somente texto) para melhorar
  consistência e padronização do laudo sem introduzir novos achados.

Quantização via bitsandbytes é opcional: se não estiver instalado, o sistema faz
fallback para carregamento normal (sem quebrar a execução).
"""


from __future__ import annotations

# Este módulo concentra toda a parte pesada (torch/transformers) para:
# - carregar o modelo com configuração consistente,
# - aplicar quantização (quando disponível),
# - rodar geração multimodal e texto-only,
# - executar o pipeline (single-pass ou chunked map-reduce).

import gc
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from reporting import (
    Language,
    ReportTemplate,
    build_chunk_prompt,
    build_default_prompt,
    build_final_prompt_from_findings,
    build_refine_prompt,
    dedupe_findings,
    normalize_report_text,
    parse_findings_bullets,
    parse_findings_json,
    qc_report,
)

from vram import QuantizationMode, supports_bfloat16


PipelineMode = Literal["single", "single_refine", "chunked", "chunked_refine"]


@dataclass(frozen=True)
class GenerationParams:
    max_new_tokens: int = 350
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


@dataclass(frozen=True)
class ModelLoadConfig:
    model_id: str
    quantization: QuantizationMode
    device_map: str
    dtype: str  # "bf16" | "fp16" | "fp32"
    attn_implementation: Optional[str] = None


def _select_default_dtype() -> str:
    if torch.cuda.is_available():
        return "bf16" if supports_bfloat16() else "fp16"
    return "fp32"


def _dtype_from_string(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d == "bf16":
        return torch.bfloat16
    if d == "fp16":
        return torch.float16
    return torch.float32


def _detect_mig() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
        return "MIG" in name.upper()
    except Exception:
        return False


def _build_bnb_config(quantization: QuantizationMode):
    # bitsandbytes is optional.
    if quantization not in ("4bit", "8bit"):
        return None
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except Exception:
        return None

    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    # 4bit
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if supports_bfloat16() else torch.float16,
    )


class ModelManager:
    def __init__(self, model_id: str):
        self._model_id = model_id
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._loaded_config: Optional[ModelLoadConfig] = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call ensure_loaded() first.")
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError("Processor not loaded. Call ensure_loaded() first.")
        return self._processor

    @property
    def loaded_config(self) -> Optional[ModelLoadConfig]:
        return self._loaded_config

    def unload(self) -> None:
        self._processor = None
        self._model = None
        self._loaded_config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_loaded(
        self,
        config: Optional[ModelLoadConfig] = None,
        *,
        hf_token: Optional[str] = None,
    ) -> ModelLoadConfig:
        """
        Carrega modelo/processor se ainda não estiver carregado ou se a configuração mudou.
        """
        if config is None:
            config = ModelLoadConfig(
                model_id=self._model_id,
                quantization=os.getenv("MODEL_QUANTIZATION", "auto"),  # auto|none|8bit|4bit
                device_map="auto",
                dtype=os.getenv("MODEL_DTYPE", _select_default_dtype()),
                attn_implementation=os.getenv("MODEL_ATTN_IMPL") or None,
            )

        # Normalize quantization
        q = str(config.quantization).lower().strip()
        if q not in ("auto", "none", "8bit", "4bit"):
            q = "auto"

        # Normalize dtype
        d = str(config.dtype).lower().strip()
        if d == "auto" or not d:
            d = _select_default_dtype()
        if d not in ("bf16", "fp16", "fp32"):
            d = _select_default_dtype()

        config = ModelLoadConfig(
            model_id=config.model_id,
            quantization=q,  # type: ignore
            device_map=config.device_map,
            dtype=d,
            attn_implementation=config.attn_implementation,
        )

        if self._loaded_config == config and self._model is not None and self._processor is not None:
            return config

        # Recarrega do zero
        self.unload()

        token = hf_token if hf_token is not None else os.getenv("HF_TOKEN")

        # Compatibilidade com MIG: attention eager + fp32 costuma evitar alguns problemas de cublas.
        mig = _detect_mig()
        dtype = config.dtype
        attn_impl = config.attn_implementation

        if mig:
            dtype = "fp32"
            attn_impl = "eager"

        torch_dtype = _dtype_from_string(dtype)

        self._processor = AutoProcessor.from_pretrained(config.model_id, token=token)

        kwargs: Dict[str, Any] = dict(
            device_map=config.device_map,
            torch_dtype=torch_dtype,
            token=token,
        )

        bnb_config = _build_bnb_config(config.quantization)
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config

        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        self._model = AutoModelForImageTextToText.from_pretrained(config.model_id, **kwargs)
        try:
            self._model.generation_config.do_sample = True
        except Exception:
            pass

        self._loaded_config = ModelLoadConfig(
            model_id=config.model_id,
            quantization=config.quantization,
            device_map=config.device_map,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        return self._loaded_config

    def describe(self) -> Dict[str, Any]:
        cfg = self._loaded_config
        if cfg is None:
            return {"loaded": False}

        try:
            dev = str(next(self.model.parameters()).device)
            dt = str(next(self.model.parameters()).dtype)
        except Exception:
            dev = "unknown"
            dt = "unknown"

        return {
            "loaded": True,
            "model_id": cfg.model_id,
            "quantization": cfg.quantization,
            "dtype": cfg.dtype,
            "device_map": cfg.device_map,
            "attn_implementation": cfg.attn_implementation,
            "param_device": dev,
            "param_dtype": dt,
        }


def _content_with_temp_images(images: List[Image.Image]) -> Tuple[List[Dict[str, Any]], List[str]]:
    temp_paths: List[str] = []
    content: List[Dict[str, Any]] = []
    for img in images:
        tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tf.name, format="PNG")
        tf.close()
        temp_paths.append(tf.name)
        content.append({"type": "image", "url": tf.name})
    return content, temp_paths


def _content_with_image_paths(paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for p in paths:
        content.append({"type": "image", "url": p})
    return content


def _cleanup_temp_files(paths: List[str]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except Exception:
            pass


def _apply_chat_template(
    *,
    processor,
    content: List[Dict[str, Any]],
    model_device,
    model_dtype: torch.dtype,
):
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Move tensores para o device do modelo; e ajusta dtype em tensores float.
    inputs = inputs.to(device=model_device)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(dtype=model_dtype)
    return inputs


def generate_images_to_text(
    *,
    manager: ModelManager,
    images: List[Image.Image],
    image_paths: Optional[List[str]] = None,
    prompt: str,
    gen: GenerationParams,
    empty_cuda_cache: bool = True,
) -> Tuple[str, int]:
    """
    Uma chamada única (imagens + texto do prompt).
    Retorna (text, input_token_len).
    """
    model = manager.model
    processor = manager.processor

    temp_paths: List[str] = []
    if image_paths is not None:
        content = _content_with_image_paths(image_paths)
    else:
        content, temp_paths = _content_with_temp_images(images)
    content.append({"type": "text", "text": prompt})

    try:
        inputs = _apply_chat_template(
            processor=processor,
            content=content,
            model_device=model.device,
            model_dtype=next(model.parameters()).dtype,
        )
        input_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            if gen.do_sample and gen.temperature > 0:
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(gen.max_new_tokens),
                    do_sample=True,
                    temperature=float(gen.temperature),
                    top_p=float(gen.top_p),
                    top_k=int(gen.top_k),
                )
            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(gen.max_new_tokens),
                    do_sample=False,
                )

        out = out[0][input_len:]
        text = processor.decode(out, skip_special_tokens=True)

        return text, input_len
    finally:
        _cleanup_temp_files(temp_paths)
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_text_only(
    *,
    manager: ModelManager,
    prompt: str,
    gen: GenerationParams,
) -> Tuple[str, int]:
    """
    Geração somente texto (sem imagens). Útil para refinamento/finalização.
    Retorna (text, input_token_len).
    """
    model = manager.model
    processor = manager.processor

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    inputs = _apply_chat_template(
        processor=processor,
        content=content,
        model_device=model.device,
        model_dtype=next(model.parameters()).dtype,
    )
    input_len = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        if gen.do_sample and gen.temperature > 0:
            out = model.generate(
                **inputs,
                max_new_tokens=int(gen.max_new_tokens),
                do_sample=True,
                temperature=float(gen.temperature),
                top_p=float(gen.top_p),
                top_k=int(gen.top_k),
            )
        else:
            out = model.generate(
                **inputs,
                max_new_tokens=int(gen.max_new_tokens),
                do_sample=False,
            )

    out = out[0][input_len:]
    text = processor.decode(out, skip_special_tokens=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text, input_len


@dataclass
class PipelineResult:
    report_text: str
    qc_issues: List[str]
    debug: Dict[str, Any]
    extracted_findings: Optional[List[Dict[str, Any]]] = None


def generate_report_pipeline(
    *,
    manager: ModelManager,
    images: List[Image.Image],
    modality: str,
    study_info: Dict[str, Any],
    language: Language,
    template: ReportTemplate,
    clinical_history: str,
    additional_instructions: str,
    prompt_override: str,
    pipeline: PipelineMode,
    chunk_size: int,
    gen: GenerationParams,
) -> PipelineResult:
    """
    API principal (alto nível) para geração do laudo.

    prompt_override: se não estiver vazio, vira o prompt usado nas chamadas com imagem.
    additional_instructions: é anexado ao prompt padrão (ignorado quando houver override).
    """
    debug: Dict[str, Any] = {
        "pipeline": pipeline,
        "chunks": 1,
        "images_used": len(images),
        "input_token_len": None,
    }

    # Single-pass prompt
    if prompt_override.strip():
        base_prompt = prompt_override.strip()
    else:
        base_prompt = build_default_prompt(
            modality=modality,
            study_info=study_info,
            language=language,
            template=template,
            clinical_history=clinical_history,
            additional_instructions=additional_instructions,
        )

    # Para chunked, salvar as imagens uma única vez em disco costuma deixar o pipeline bem mais "liso":
    # evita ficar escrevendo/deletando PNG a cada chunk (muito I/O no loop).
    image_paths: Optional[List[str]] = None
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    use_shared_paths = pipeline in ("chunked", "chunked_refine") and len(images) > 0

    try:
        if use_shared_paths:
            temp_dir = tempfile.TemporaryDirectory(prefix="medgemma_imgs_")
            image_paths = []
            for i, img in enumerate(images):
                p = os.path.join(temp_dir.name, f"img_{i:05d}.png")
                img.save(p, format="PNG")
                image_paths.append(p)

        if pipeline == "single":
            draft, in_len = generate_images_to_text(manager=manager, images=images, prompt=base_prompt, gen=gen)
            report = normalize_report_text(draft, language=language)
            issues = qc_report(report, language=language)
            debug["input_token_len"] = in_len
            return PipelineResult(report_text=report, qc_issues=issues, debug=debug)

        if pipeline == "single_refine":
            draft, in_len = generate_images_to_text(manager=manager, images=images, prompt=base_prompt, gen=gen)
            refine_prompt = build_refine_prompt(language=language, template=template, draft_report=draft)
            refined, refine_in_len = generate_text_only(manager=manager, prompt=refine_prompt, gen=gen)
            report = normalize_report_text(refined, language=language)
            issues = qc_report(report, language=language)
            debug["input_token_len"] = {"draft": in_len, "refine": refine_in_len}
            return PipelineResult(report_text=report, qc_issues=issues, debug=debug)

        # Chunked map-reduce
        if chunk_size <= 0:
            chunk_size = 16
        if image_paths is not None:
            chunks_paths: List[List[str]] = [image_paths[i : i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
            debug["chunks"] = len(chunks_paths)
        else:
            chunks: List[List[Image.Image]] = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]
            debug["chunks"] = len(chunks)

        extracted: List[Dict[str, Any]] = []
        chunk_warnings: List[str] = []
        total_in_lens: List[int] = []

        # Para extração em JSON, um chunk determinístico costuma:
        # - gerar JSON mais consistente (menos fallback),
        # - reduzir variação,
        # - ficar mais previsível de depurar.
        chunk_gen = GenerationParams(
            max_new_tokens=min(int(gen.max_new_tokens), 220),
            temperature=0.0,
            top_p=float(gen.top_p),
            top_k=int(gen.top_k),
            do_sample=False,
        )

        if image_paths is not None:
            total_chunks = len(chunks_paths)
            for idx, chunk_p in enumerate(chunks_paths, start=1):
                chunk_prompt = build_chunk_prompt(
                    modality=modality,
                    language=language,
                    template=template,
                    chunk_index=idx,
                    chunk_count=total_chunks,
                    clinical_history=clinical_history,
                )

                out, in_len = generate_images_to_text(
                    manager=manager,
                    images=[],
                    image_paths=chunk_p,
                    prompt=chunk_prompt,
                    gen=chunk_gen,
                    empty_cuda_cache=False,
                )
                total_in_lens.append(in_len)

                f_json, warns = parse_findings_json(out)
                if f_json:
                    extracted.extend(f_json)
                    chunk_warnings.extend(warns)
                    continue

                # Fallback
                extracted.extend(parse_findings_bullets(out))
                chunk_warnings.extend(warns)
        else:
            total_chunks = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                chunk_prompt = build_chunk_prompt(
                    modality=modality,
                    language=language,
                    template=template,
                    chunk_index=idx,
                    chunk_count=total_chunks,
                    clinical_history=clinical_history,
                )

                out, in_len = generate_images_to_text(
                    manager=manager,
                    images=chunk,
                    prompt=chunk_prompt,
                    gen=chunk_gen,
                    empty_cuda_cache=False,
                )
                total_in_lens.append(in_len)

                f_json, warns = parse_findings_json(out)
                if f_json:
                    extracted.extend(f_json)
                    chunk_warnings.extend(warns)
                    continue

                # Fallback
                extracted.extend(parse_findings_bullets(out))
                chunk_warnings.extend(warns)

        extracted = dedupe_findings(extracted)

        final_prompt = build_final_prompt_from_findings(
            modality=modality,
            study_info=study_info,
            language=language,
            template=template,
            clinical_history=clinical_history,
            findings=extracted,
            additional_instructions=additional_instructions,
        )

        final_text, final_in_len = generate_text_only(manager=manager, prompt=final_prompt, gen=gen)

        if pipeline == "chunked_refine":
            refine_prompt = build_refine_prompt(language=language, template=template, draft_report=final_text)
            final_text, refine_in_len = generate_text_only(manager=manager, prompt=refine_prompt, gen=gen)
            debug["input_token_len"] = {"chunks": total_in_lens, "final": final_in_len, "refine": refine_in_len}
        else:
            debug["input_token_len"] = {"chunks": total_in_lens, "final": final_in_len}

        report = normalize_report_text(final_text, language=language)
        issues = qc_report(report, language=language)

        # Surface parsing warnings as QC issues (non-fatal)
        for w in chunk_warnings:
            if w not in issues:
                issues.append(w)

        return PipelineResult(
            report_text=report,
            qc_issues=issues,
            debug=debug,
            extracted_findings=extracted,
        )
    finally:
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
