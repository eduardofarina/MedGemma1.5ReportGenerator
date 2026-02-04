"""Helpers do pipeline de laudo: prompts, extração estruturada, pós-processamento e QC.

Este módulo é agnóstico ao backend do modelo: NÃO importa torch/transformers.
A ideia é manter aqui apenas lógica de:
- montagem de prompt (template + idioma + regras),
- schema/JSON para "achados estruturados",
- saneamento e checagens básicas de qualidade,
- rodapé de debug para reprodutibilidade.
"""


from __future__ import annotations

# Aqui fica só a lógica de texto: prompts, parsing e QC.
# Mantemos este arquivo livre de torch para facilitar testes e manutenção.

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import re
import textwrap
from datetime import datetime


Language = Literal["English", "Português (Brasil)", "Español"]
ReportTemplate = Literal["General", "Chest X-ray", "CT (general)", "MRI (general)"]


_LANGUAGE_RULES: Dict[Language, str] = {
    "English": "Write in English.",
    "Português (Brasil)": "Escreva em português do Brasil.",
    "Español": "Escribe en español.",
}

_SECTION_HEADERS: Dict[Language, Dict[str, str]] = {
    "English": {
        "clinical_history": "CLINICAL HISTORY",
        "technique": "TECHNIQUE",
        "findings": "FINDINGS",
        "impression": "IMPRESSION",
        "fallback_impression": "Please refer to the findings above.",
    },
    "Português (Brasil)": {
        "clinical_history": "HISTÓRIA CLÍNICA",
        "technique": "TÉCNICA",
        "findings": "ACHADOS",
        "impression": "IMPRESSÃO",
        "fallback_impression": "Ver achados acima.",
    },
    "Español": {
        "clinical_history": "HISTORIA CLÍNICA",
        "technique": "TÉCNICA",
        "findings": "HALLAZGOS",
        "impression": "IMPRESIÓN",
        "fallback_impression": "Ver hallazgos arriba.",
    },
}

_COMMON_GUARDRAILS: Dict[Language, str] = {
    "English": """
- Do NOT include or invent patient-identifying data.
- If something is not visible / uncertain, say so and avoid guessing.
- Do NOT claim measurements unless you are confident they are visible.
- Prefer descriptive imaging findings over definitive diagnoses when uncertain.
- Keep IMPRESSION short and clinically actionable.
""".strip(),
    "Português (Brasil)": """
- NÃO inclua nem invente dados identificáveis do paciente.
- Se algo não estiver visível / for incerto, deixe isso claro e evite chutar.
- NÃO traga medidas se você não tiver certeza de que estão visíveis.
- Se estiver em dúvida, prefira descrever os achados em vez de fechar diagnóstico.
- Mantenha a IMPRESSÃO curta e clinicamente útil.
""".strip(),
    "Español": """
- NO incluyas ni inventes datos identificables del paciente.
- Si algo no es visible / es incierto, dilo claramente y evita adivinar.
- NO indiques medidas si no estás seguro de que son visibles.
- En caso de duda, prioriza describir hallazgos en lugar de cerrar diagnósticos.
- Mantén la IMPRESIÓN corta y clínicamente útil.
""".strip(),
}


def _template_rules(template: ReportTemplate, language: Language) -> str:
    h = _SECTION_HEADERS[language]
    ch = h["clinical_history"]
    tec = h["technique"]
    fin = h["findings"]
    imp = h["impression"]

    if template == "General":
        if language == "Português (Brasil)":
            return f"Produza um laudo radiológico estruturado com as seções: {ch} (se fornecida), {tec}, {fin}, {imp}. Use linguagem profissional e objetiva."
        if language == "Español":
            return f"Produzca un informe radiológico estructurado con las secciones: {ch} (si se proporciona), {tec}, {fin}, {imp}. Use un lenguaje profesional y conciso."
        return f"Produce a structured radiology report with the sections: {ch} (if provided), {tec}, {fin}, {imp}. Use concise professional language."

    if template == "Chest X-ray":
        if language == "Português (Brasil)":
            return (
                f"Produza um laudo de radiografia de tórax com as seções: {ch} (se fornecida), {tec}, "
                f"{fin} (Vias aéreas, Pulmões/Pleuras, Coração/Mediastino, Ossos/Partes moles) e {imp}. "
                "Se houver dispositivos (tubo/sonda/cateter), inclua uma subseção de Dispositivos em ACHADOS."
            )
        if language == "Español":
            return (
                f"Produzca un informe de radiografía de tórax con secciones: {ch} (si se proporciona), {tec}, "
                f"{fin} (Vías aéreas, Pulmones/Pleura, Corazón/Mediastino, Huesos/Tejidos blandos) e {imp}. "
                "Si hay dispositivos (tubo/sonda/catéter), incluya una subsección de Dispositivos en HALLAZGOS."
            )
        return (
            f"Produce a chest radiograph report with sections: {ch} (if provided), {tec}, "
            f"{fin} (Airways, Lungs/Pleura, Heart/Mediastinum, Bones/Soft Tissues), {imp}. "
            "If lines/tubes are present, include a dedicated Devices section under FINDINGS."
        )

    if template == "CT (general)":
        if language == "Português (Brasil)":
            return (
                f"Produza um laudo de TC com as seções: {ch} (se fornecida), {tec}, {fin} (organizado por sistema/região) e {imp} (numerada). "
                "Só mencione contraste se isso estiver explícito; caso contrário, deixe como desconhecido."
            )
        if language == "Español":
            return (
                f"Produzca un informe de TC con secciones: {ch} (si se proporciona), {tec}, {fin} (organizado por región/sistema) e {imp} (numerada). "
                "Mencione el contraste solo si está explícitamente indicado; de lo contrario, indique si es desconocido."
            )
        return (
            f"Produce a CT report with sections: {ch} (if provided), {tec}, {fin} (organized by region/system), {imp} (numbered). "
            "Mention contrast use only if explicitly known; otherwise state if unknown."
        )

    # MRI (general)
    if language == "Português (Brasil)":
        return (
            f"Produza um laudo de RM com as seções: {ch} (se fornecida), {tec}, {fin} e {imp} (numerada). "
            "Evite superestimar achados sutis; seja explícito sobre limitações."
        )
    if language == "Español":
        return (
            f"Produzca un informe de RM con secciones: {ch} (si se proporciona), {tec}, {fin} e {imp} (numerada). "
            "Evite sobrediagnosticar hallazgos sutiles; sea explícito sobre limitaciones."
        )
    return (
        f"Produce an MRI report with sections: {ch} (if provided), {tec}, {fin}, {imp} (numbered). "
        "Avoid over-calling subtle findings; be explicit about limitations."
    )


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_default_prompt(
    *,
    modality: str,
    study_info: Dict[str, Any],
    language: Language,
    template: ReportTemplate,
    clinical_history: str,
    additional_instructions: str,
) -> str:
    """Monta prompt base (template + idioma + regras) para geracao direta do laudo."""
    if language == "Português (Brasil)":
        role_line = "Você é um médico radiologista com certificação."
        modality_label = "Modalidade"
        study_desc_label = "Descrição do estudo"
        body_part_label = "Região examinada"
        series_summary_label = "Resumo das séries"
        clinical_history_label = "História clínica / indicação:"
        guardrails_label = "Regras:"
        additional_label = "Instruções adicionais do usuário:"
    elif language == "Español":
        role_line = "Eres un radiólogo certificado."
        modality_label = "Modalidad"
        study_desc_label = "Descripción del estudio"
        body_part_label = "Región examinada"
        series_summary_label = "Resumen de series"
        clinical_history_label = "Historia clínica / indicación:"
        guardrails_label = "Reglas:"
        additional_label = "Instrucciones adicionales del usuario:"
    else:
        role_line = "You are a board-certified radiologist."
        modality_label = "Modality"
        study_desc_label = "Study description"
        body_part_label = "Body part examined"
        series_summary_label = "Series summary"
        clinical_history_label = "Clinical history / indication:"
        guardrails_label = "Guardrails:"
        additional_label = "Additional user instructions:"

    study_desc = str(study_info.get("StudyDescription") or "").strip()
    body_part = str(study_info.get("BodyPartExamined") or "").strip()

    lines = [
        role_line,
        _LANGUAGE_RULES[language],
        _template_rules(template, language),
        "",
        f"{modality_label}: {modality}",
    ]
    if study_desc:
        lines.append(f"{study_desc_label}: {study_desc}")
    if body_part:
        lines.append(f"{body_part_label}: {body_part}")

    series_summary = study_info.get("SeriesSummary")
    if series_summary:
        # Keep it short to avoid blowing the prompt.
        lines.append(f"{series_summary_label}: {series_summary}")

    if clinical_history.strip():
        lines.append("")
        lines.append(clinical_history_label)
        lines.append(clinical_history.strip())

    lines.append("")
    lines.append(guardrails_label)
    lines.append(_COMMON_GUARDRAILS[language])

    if additional_instructions.strip():
        lines.append("")
        lines.append(additional_label)
        lines.append(additional_instructions.strip())

    return _clean_text("\n".join(lines))


def build_chunk_prompt(
    *,
    modality: str,
    language: Language,
    template: ReportTemplate,
    chunk_index: int,
    chunk_count: int,
    clinical_history: str,
) -> str:
    """Monta prompt para cada chunk: objetivo e extrair achados estruturados daquele subconjunto de imagens."""
    # Chunk prompt: force "evidence-first" extraction.
    lang = _LANGUAGE_RULES[language]
    template_hint = _template_rules(template, language)

    schema = {
        "findings": [
            {
                "anatomy": "string",
                "finding": "string",
                "laterality": "left|right|bilateral|midline|unknown",
                "confidence": "low|medium|high",
            }
        ]
    }
    schema_text = json.dumps(schema, indent=2)

    if language == "Português (Brasil)":
        lines = [
            "Você é um radiologista extraindo evidências dos achados.",
            lang,
            "",
            f"Estas imagens são o chunk {chunk_index}/{chunk_count} de um estudo {modality}.",
            "Tarefa: extraia SOMENTE achados observáveis nestas imagens.",
            "A saída DEVE ser JSON válido e DEVE seguir este schema (sem chaves extras):",
            schema_text,
            "",
            "Regras:",
            "- Não escreva nada fora do JSON.",
            "- Não adicione diagnósticos, a menos que sejam diretamente evidentes (ex.: pneumotórax).",
            "- Se não houver achados visíveis neste chunk, retorne {\"findings\": []}.",
        ]
    elif language == "Español":
        lines = [
            "Eres un radiólogo extrayendo evidencia de los hallazgos.",
            lang,
            "",
            f"Estas imágenes son el chunk {chunk_index}/{chunk_count} de un estudio {modality}.",
            "Tarea: extrae SOLO hallazgos observables en estas imágenes.",
            "La salida DEBE ser JSON válido y DEBE seguir este esquema (sin claves extra):",
            schema_text,
            "",
            "Reglas:",
            "- No escribas nada fuera del JSON.",
            "- No agregues diagnósticos salvo que sean directamente evidentes (p. ej., neumotórax).",
            "- Si no hay hallazgos visibles en este chunk, devuelve {\"findings\": []}.",
        ]
    else:
        lines = [
            "You are a radiologist performing evidence extraction.",
            lang,
            "",
            f"These images are chunk {chunk_index}/{chunk_count} from a {modality} study.",
            "Task: Extract ONLY observable imaging findings from these images.",
            "Output MUST be valid JSON and MUST match this schema (no extra keys):",
            schema_text,
            "",
            "Rules:",
            "- Do not output any text outside JSON.",
            "- Do not add diagnoses unless directly evident (e.g., pneumothorax).",
            "- If no findings are visible in this chunk, return {\"findings\": []}.",
        ]
    if clinical_history.strip():
        lines.append("")
        if language == "Português (Brasil)":
            lines.append("História clínica / indicação (apenas contexto):")
        elif language == "Español":
            lines.append("Historia clínica / indicación (solo contexto):")
        else:
            lines.append("Clinical history / indication (context only):")
        lines.append(clinical_history.strip())

    # Keep a small template hint to nudge relevant organ-system coverage.
    lines.append("")
    if language == "Português (Brasil)":
        lines.append("Dica de cobertura (seja conciso):")
    elif language == "Español":
        lines.append("Pista de cobertura (sé conciso):")
    else:
        lines.append("Coverage hint (keep concise):")
    lines.append(template_hint)

    return _clean_text("\n".join(lines))


def build_final_prompt_from_findings(
    *,
    modality: str,
    study_info: Dict[str, Any],
    language: Language,
    template: ReportTemplate,
    clinical_history: str,
    findings: List[Dict[str, Any]],
    additional_instructions: str,
) -> str:
    """Monta prompt final para gerar o laudo a partir de achados consolidados (sem imagens)."""
    lang = _LANGUAGE_RULES[language]
    template_hint = _template_rules(template, language)

    # Compact the findings list to keep prompt size under control.
    compact = []
    for f in findings[:200]:
        anatomy = str(f.get("anatomy", "")).strip()
        finding = str(f.get("finding", "")).strip()
        laterality = str(f.get("laterality", "")).strip()
        if not finding:
            continue
        item = {
            "anatomy": anatomy[:80],
            "finding": finding[:200],
            "laterality": laterality[:20] if laterality else "unknown",
        }
        compact.append(item)

    findings_json = json.dumps({"findings": compact}, ensure_ascii=False, indent=2)

    study_desc = str(study_info.get("StudyDescription") or "").strip()
    body_part = str(study_info.get("BodyPartExamined") or "").strip()

    lines = [
        ("Você é um médico radiologista com certificação."
         if language == "Português (Brasil)"
         else "Eres un radiólogo certificado."
         if language == "Español"
         else "You are a board-certified radiologist."),
        lang,
        template_hint,
        "",
        (f"Modalidade: {modality}"
         if language == "Português (Brasil)"
         else f"Modalidad: {modality}"
         if language == "Español"
         else f"Modality: {modality}"),
    ]
    if study_desc:
        if language == "Português (Brasil)":
            lines.append(f"Descrição do estudo: {study_desc}")
        elif language == "Español":
            lines.append(f"Descripción del estudio: {study_desc}")
        else:
            lines.append(f"Study description: {study_desc}")
    if body_part:
        if language == "Português (Brasil)":
            lines.append(f"Região examinada: {body_part}")
        elif language == "Español":
            lines.append(f"Región examinada: {body_part}")
        else:
            lines.append(f"Body part examined: {body_part}")

    if clinical_history.strip():
        lines.append("")
        if language == "Português (Brasil)":
            lines.append("História clínica / indicação:")
        elif language == "Español":
            lines.append("Historia clínica / indicación:")
        else:
            lines.append("Clinical history / indication:")
        lines.append(clinical_history.strip())

    lines.append("")
    if language == "Português (Brasil)":
        lines.append("Achados extraídos (consolidados em JSON):")
    elif language == "Español":
        lines.append("Hallazgos extraídos (consolidados en JSON):")
    else:
        lines.append("Aggregated extracted findings (JSON):")
    lines.append(findings_json)

    lines.append("")
    if language == "Português (Brasil)":
        lines.append("Instruções:")
        lines.append("- Escreva o laudo final estruturado.")
        lines.append("- Use SOMENTE informações presentes nos achados consolidados acima.")
        lines.append("- NÃO invente achados adicionais.")
        lines.append("- Se a lista de achados estiver vazia, descreva que não há alterações agudas identificáveis (quando fizer sentido) e mantenha o texto conciso.")
    elif language == "Español":
        lines.append("Instrucciones:")
        lines.append("- Escribe el informe final estructurado.")
        lines.append("- Usa SOLO la información presente en los hallazgos agregados arriba.")
        lines.append("- NO inventes hallazgos adicionales.")
        lines.append("- Si la lista está vacía, indica que no se identifican hallazgos agudos (si corresponde) y mantén el informe conciso.")
    else:
        lines.append("Instructions:")
        lines.append("- Write the final structured report.")
        lines.append("- ONLY use information present in the aggregated findings above.")
        lines.append("- Do NOT invent additional findings.")
        lines.append("- If the findings list is empty, state that no acute abnormality is identified (as appropriate), and keep report concise.")

    lines.append("")
    if language == "Português (Brasil)":
        lines.append("Regras:")
    elif language == "Español":
        lines.append("Reglas:")
    else:
        lines.append("Guardrails:")
    lines.append(_COMMON_GUARDRAILS[language])

    if additional_instructions.strip():
        lines.append("")
        if language == "Português (Brasil)":
            lines.append("Instruções adicionais do usuário:")
        elif language == "Español":
            lines.append("Instrucciones adicionales del usuario:")
        else:
            lines.append("Additional user instructions:")
        lines.append(additional_instructions.strip())

    return _clean_text("\n".join(lines))


def build_refine_prompt(
    *,
    language: Language,
    template: ReportTemplate,
    draft_report: str,
) -> str:
    """Monta prompt de refinamento: melhora clareza/estrutura sem adicionar novos achados."""
    lang = _LANGUAGE_RULES[language]
    template_hint = _template_rules(template, language)

    if language == "Português (Brasil)":
        lines = [
            "Você é um radiologista sênior revisando e melhorando o texto.",
            lang,
            template_hint,
            "",
            "Reescreva o laudo abaixo para melhorar clareza, estrutura e consistência.",
            "CRÍTICO: NÃO adicione novos achados. NÃO mude o significado. Só reestruture/refraseie.",
            "",
            "Laudo para refinar:",
            draft_report.strip(),
        ]
    elif language == "Español":
        lines = [
            "Eres un radiólogo senior revisando y mejorando el texto.",
            lang,
            template_hint,
            "",
            "Reescribe el informe a continuación para mejorar claridad, estructura y consistencia.",
            "CRÍTICO: NO agregues hallazgos nuevos. NO cambies el significado. Solo reestructura/reformula.",
            "",
            "Informe para refinar:",
            draft_report.strip(),
        ]
    else:
        lines = [
            "You are a senior radiologist editor.",
            lang,
            template_hint,
            "",
            "Rewrite the report below to improve clarity, structure, and consistency.",
            "CRITICAL: Do NOT add new findings. Do NOT change the meaning of the content. Only rephrase/restructure.",
            "",
            "Report to refine:",
            draft_report.strip(),
        ]
    return _clean_text("\n".join(lines))


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extrai o primeiro trecho que parece um objeto JSON.

    Funciona mesmo quando o modelo coloca texto antes/depois do JSON.
    """
    if not text:
        return None

    text = text.strip()
    # Fast path: já parece um JSON "puro".
    if text.startswith("{") and text.endswith("}"):
        return text

    # Procura o primeiro objeto JSON com contagem de chaves (respeitando strings).
    def _find_json_from(start_idx: int) -> Optional[str]:
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
                if depth < 0:
                    return None
        return None

    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return None
        candidate = _find_json_from(start)
        if candidate:
            # Só aceita se for parseável.
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
        idx = start + 1


def parse_findings_json(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Retorna (findings, warnings).
    """
    warnings: List[str] = []
    raw = _extract_first_json_object(text)
    if raw is None:
        return [], ["Não encontrei JSON no output do chunk; vou tentar interpretar como lista (bullets)."]

    try:
        obj = json.loads(raw)
    except Exception:
        return [], ["Falha ao fazer parse do JSON do chunk; vou tentar interpretar como lista (bullets)."]

    findings = obj.get("findings")
    if not isinstance(findings, list):
        return [], ["O JSON não tinha um array 'findings'; vou tentar interpretar como lista (bullets)."]

    cleaned: List[Dict[str, Any]] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        anatomy = str(f.get("anatomy", "")).strip()
        finding = str(f.get("finding", "")).strip()
        laterality = str(f.get("laterality", "unknown")).strip() or "unknown"
        confidence = str(f.get("confidence", "medium")).strip() or "medium"
        if not finding:
            continue
        cleaned.append(
            {
                "anatomy": anatomy,
                "finding": finding,
                "laterality": laterality,
                "confidence": confidence,
            }
        )

    return cleaned, warnings


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def parse_findings_bullets(text: str) -> List[Dict[str, Any]]:
    """
    Fallback bem tolerante quando o JSON falha.
    """
    lines = _clean_text(text).split("\n")
    items: List[str] = []
    for ln in lines:
        if _BULLET_RE.match(ln):
            items.append(_BULLET_RE.sub("", ln).strip())
    if not items:
        # Tenta separar por ponto e vírgula.
        for part in re.split(r"[;\n]+", _clean_text(text)):
            part = part.strip()
            if len(part) >= 5:
                items.append(part)
    findings: List[Dict[str, Any]] = []
    for it in items[:200]:
        findings.append(
            {
                "anatomy": "",
                "finding": it,
                "laterality": "unknown",
                "confidence": "low",
            }
        )
    return findings


def dedupe_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicatas e normaliza listas de achados para reduzir repeticao entre chunks."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for f in findings:
        key = re.sub(r"[^a-z0-9]+", " ", (f.get("anatomy", "") + " " + f.get("finding", "")).lower()).strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def normalize_report_text(report: str, *, language: Language = "English") -> str:
    """
    Limpeza básica + normalização de seções.

    Não tenta “consertar” o conteúdo, só organiza a formatação.
    """
    report = _clean_text(report)

    # Remove leading assistant-like prefixes.
    report = re.sub(r"^(assistant:|response:)\s*", "", report, flags=re.IGNORECASE)

    h = _SECTION_HEADERS.get(language) or _SECTION_HEADERS["English"]
    technique_h = h["technique"]
    findings_h = h["findings"]
    impression_h = h["impression"]
    fallback_impression = h["fallback_impression"]

    def _header_variants() -> Dict[str, List[str]]:
        # Variantes aceitas (inclui versões sem acento e equivalentes em inglês).
        if language == "Português (Brasil)":
            return {
                "clinical_history": ["HISTÓRIA CLÍNICA", "HISTORIA CLINICA", "CLINICAL HISTORY"],
                "technique": ["TÉCNICA", "TECNICA", "TECHNIQUE"],
                "findings": ["ACHADOS", "FINDINGS"],
                "impression": ["IMPRESSÃO", "IMPRESSAO", "IMPRESSION"],
            }
        if language == "Español":
            return {
                "clinical_history": ["HISTORIA CLÍNICA", "HISTORIA CLINICA", "CLINICAL HISTORY"],
                "technique": ["TÉCNICA", "TECNICA", "TECHNIQUE"],
                "findings": ["HALLAZGOS", "FINDINGS"],
                "impression": ["IMPRESIÓN", "IMPRESION", "IMPRESSION"],
            }
        return {
            "clinical_history": ["CLINICAL HISTORY"],
            "technique": ["TECHNIQUE"],
            "findings": ["FINDINGS"],
            "impression": ["IMPRESSION"],
        }

    variants = _header_variants()

    def _canonical(section: str) -> str:
        if section == "technique":
            return technique_h
        if section == "findings":
            return findings_h
        if section == "impression":
            return impression_h
        return h["clinical_history"]

    # Normaliza cabeçalhos (troca "FINDINGS" -> "ACHADOS", por exemplo), mantendo o resto da linha.
    new_lines: List[str] = []
    for ln in report.split("\n"):
        replaced = False
        for section in ("clinical_history", "technique", "findings", "impression"):
            for v in variants[section]:
                m = re.match(rf"^(\s*){re.escape(v)}(\s*[:\\-]?\s*)(.*)$", ln, flags=re.IGNORECASE)
                if not m:
                    continue
                indent, sep, rest = m.group(1), m.group(2), m.group(3)
                # Só trata como cabeçalho se estiver “sozinho” na linha ou tiver separador explícito.
                if (rest.strip() == "") or (":" in sep or "-" in sep):
                    canon = _canonical(section)
                    if rest.strip():
                        ln = f"{indent}{canon}: {rest.strip()}"
                    else:
                        ln = f"{indent}{canon}:"
                    replaced = True
                    break
            if replaced:
                break
        new_lines.append(ln)
    report = _clean_text("\n".join(new_lines))

    def has_header(h: str) -> bool:
        return re.search(rf"^\s*{re.escape(h)}\s*[:\-]?\s*$", report, flags=re.IGNORECASE | re.MULTILINE) is not None

    missing = [x for x in (technique_h, findings_h, impression_h) if not has_header(x)]

    if missing:
        # Se o modelo não respeitou estrutura, envolve tudo em ACHADOS + IMPRESSÃO (ou equivalente).
        # Mantém conservador: não inventa conteúdo.
        wrapped = []
        wrapped.append(f"{findings_h}:")
        wrapped.append(report)
        wrapped.append("")
        wrapped.append(f"{impression_h}:")
        wrapped.append(fallback_impression)
        report = _clean_text("\n".join(wrapped))

    # Numera a seção de impressão quando houver vários itens.
    m = re.search(rf"^\s*{re.escape(impression_h)}\s*[:\-]?\s*$", report, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        # Split at impression header position.
        start = m.end()
        pre = report[: m.start()].rstrip()
        post = report[start:].lstrip()
        post_lines = [ln.strip() for ln in post.split("\n") if ln.strip()]
        if len(post_lines) >= 2 and not any(re.match(r"^\d+\.", ln) for ln in post_lines[:2]):
            post_lines = [f"{i+1}. {ln}" for i, ln in enumerate(post_lines)]
            report = _clean_text(pre + f"\n\n{impression_h}:\n" + "\n".join(post_lines))

    return report


def qc_report(report: str, *, language: Language = "English") -> List[str]:
    """
    Checagens leves de QC para avisos ao usuário (não é validação médica).
    """
    issues: List[str] = []
    report = report.strip()
    h = _SECTION_HEADERS.get(language) or _SECTION_HEADERS["English"]
    if len(report) < 80:
        if language == "Español":
            issues.append("El informe quedó muy corto; prueba aumentar los tokens o usar el paso de refinamiento.")
        elif language == "Português (Brasil)":
            issues.append("O laudo ficou muito curto; tente aumentar os tokens ou usar o passo de refinamento.")
        else:
            issues.append("Report is very short; consider increasing max tokens or using refine step.")

    # Required headers
    for section_name in (h["technique"], h["findings"], h["impression"]):
        if re.search(rf"^\s*{re.escape(section_name)}\s*[:\-]?\s*$", report, flags=re.IGNORECASE | re.MULTILINE) is None:
            if language == "Español":
                issues.append(f"Falta la sección: {section_name}")
            elif language == "Português (Brasil)":
                issues.append(f"Seção ausente: {section_name}")
            else:
                issues.append(f"Missing section header: {section_name}")

    # Obvious hallucination markers
    if re.search(r"\b(I (am|cannot|can't)|sou|soy)\b.*\b(AI|IA)\b", report, flags=re.IGNORECASE):
        if language == "Español":
            issues.append("El modelo mencionó ser IA; considera refinar con un prompt más estricto.")
        elif language == "Português (Brasil)":
            issues.append("O modelo mencionou ser IA; considere refinar com um prompt mais estrito.")
        else:
            issues.append("Model mentioned being an AI; consider refining with stricter prompt.")

    # Patient-identifying patterns (very rough)
    if re.search(r"\bMRN\b|\bDOB\b|\bSSN\b", report, flags=re.IGNORECASE):
        if language == "Español":
            issues.append("Posibles datos identificables detectados (MRN/DOB/SSN). Revisa antes de usar.")
        elif language == "Português (Brasil)":
            issues.append("Possível PHI detectada (MRN/DOB/SSN). Revise antes de usar.")
        else:
            issues.append("Potential PHI-like tokens detected (MRN/DOB/SSN). Review before use.")

    return issues


def build_debug_footer(meta: Dict[str, Any]) -> str:
    """Gera um rodape com metadados (pipeline, parametros, timestamp) para reprodutibilidade."""
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return _clean_text(
        "\n".join(
            [
                "",
                "---",
                "DEBUG (não-clínico):",
                f"- generated_at_utc: {ts}",
                f"- pipeline: {meta.get('pipeline')}",
                f"- chunks: {meta.get('chunks')}",
                f"- images_used: {meta.get('images_used')}",
                f"- input_token_len: {meta.get('input_token_len')}",
            ]
        )
    )


_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\+?\d{1,3}\s*)?(?:\(?\d{2,3}\)?\s*)?(?:\d[\s\-]*){7,}\d(?:(?=\s)|$)"
)
_UID_LIKE_RE = re.compile(r"\b\d+(?:\.\d+){2,}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")

_PHI_LABEL_RE = re.compile(
    r"(?im)^\s*((?:"
    r"patient\s*name|name|patient\s*id|patient\s*identifier|mrn|dob|ssn|accession(?:\s*number)?|"
    r"nome|paciente|id\s*do\s*paciente|prontu[aá]rio|cpf|rg|cns|data\s*de\s*nascimento|"
    r"nombre|paciente|id\s*del\s*paciente|historia\s*cl[ií]nica|dni"
    r"))\s*[:\-]\s*(.+?)\s*$"
)


def sanitize_phi_text(text: str, *, language: Language = "Português (Brasil)") -> Tuple[str, List[str]]:
    """
    Sanitiza possíveis identificadores (PHI/PII) em texto livre.

    É propositalmente heurístico: a ideia é reduzir risco de vazar dados, não “entender” contexto.
    Retorna (texto_sanitizado, avisos).
    """
    if not text:
        return text, []

    warnings: List[str] = []
    s = str(text)

    # Redação por linha quando houver rótulos explícitos.
    def _redact_label_line(m: re.Match) -> str:
        # Não usar m.group(0), porque pode conter o valor quando o separador é "-".
        label = str(m.group(1)).strip()
        return f"{label}: [REDACTED]"

    before = s
    s = _PHI_LABEL_RE.sub(_redact_label_line, s)
    if s != before:
        if language == "Español":
            warnings.append("Sanitización: redacté líneas que parecen contener datos del paciente (p. ej., nombre/ID).")
        elif language == "English":
            warnings.append("Sanitizer: redacted likely patient-identifying fields (e.g., name/ID).")
        else:
            warnings.append("Sanitização: removi valores em linhas que parecem conter dados do paciente (ex.: nome/ID).")

    # Email
    if _EMAIL_RE.search(s):
        s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
        if language == "Español":
            warnings.append("Sanitización: redacté un posible e-mail.")
        elif language == "English":
            warnings.append("Sanitizer: redacted a possible email.")
        else:
            warnings.append("Sanitização: removi possível e-mail.")

    # Telefone (bem aproximado)
    if _PHONE_RE.search(s):
        s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
        if language == "Español":
            warnings.append("Sanitización: redacté un posible teléfono.")
        elif language == "English":
            warnings.append("Sanitizer: redacted a possible phone number.")
        else:
            warnings.append("Sanitização: removi possível telefone.")

    # UIDs/IDs longos (prontuário, accession, etc.)
    if _UID_LIKE_RE.search(s):
        s = _UID_LIKE_RE.sub("[REDACTED_UID]", s)
        if language == "Español":
            warnings.append("Sanitización: redacté un posible UID/identificador técnico.")
        elif language == "English":
            warnings.append("Sanitizer: redacted a UID-like identifier.")
        else:
            warnings.append("Sanitização: removi possível UID/identificador técnico.")

    if _LONG_DIGITS_RE.search(s):
        s = _LONG_DIGITS_RE.sub("[REDACTED_ID]", s)
        if language == "Español":
            warnings.append("Sanitización: redacté un identificador numérico largo (posible ID).")
        elif language == "English":
            warnings.append("Sanitizer: redacted a long numeric identifier.")
        else:
            warnings.append("Sanitização: removi sequência numérica longa (possível ID).")

    warnings = list(dict.fromkeys(warnings))
    return _clean_text(s), warnings


def sanitize_findings(
    findings: List[Dict[str, Any]],
    *,
    language: Language = "Português (Brasil)",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Sanitiza campos textuais de uma lista de achados (para evitar PHI no .json)."""
    out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        anatomy = f.get("anatomy", "")
        finding = f.get("finding", "")
        anatomy_s, w1 = sanitize_phi_text(str(anatomy), language=language)
        finding_s, w2 = sanitize_phi_text(str(finding), language=language)
        warnings.extend(w1)
        warnings.extend(w2)
        nf = dict(f)
        nf["anatomy"] = anatomy_s
        nf["finding"] = finding_s
        out.append(nf)
    warnings = list(dict.fromkeys(warnings))
    return out, warnings
