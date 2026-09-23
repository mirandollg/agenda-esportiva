#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coletor de agenda esportiva.

Fontes:
  1. Esportes na TV (Bluesky) -> imagem da agenda do dia, lida por OCR celula a celula
  2. Doentes por Futebol      -> HTML em texto, usado como fonte e como conferente do OCR
  3. Tomada de Tempo          -> automobilismo, via API REST do WordPress

Saida: docs/app.json  (publicado pelo GitHub Pages)
"""

import io
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

# ----------------------------------------------------------------------------
# CONFIGURACAO
# ----------------------------------------------------------------------------

# Conta do Esportes na TV no Bluesky. Usamos o DID, nao o handle:
# o handle pode mudar se eles registrarem dominio proprio, o DID nunca muda.
DID = "did:plc:lngl4ki52wbunv2xzq74bqbb"
BSKY_API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
BSKY_LIMITE = 30  # quantos posts puxar do feed

DPF_URL = "https://doentesporfutebol.com.br/guiadejogos/"
TDT_API = "https://www.tomadadetempo.com.br/wp-json/wp/v2/posts"

SAIDA = "docs/app.json"
DEBUG_DIR = "debug"

# Filtro de escopo. Lista vazia = manter tudo.
# Exemplo: ESCOPO = ["brasileiro", "libertadores", "copa do brasil", "f1", "motogp"]
ESCOPO = []

# Quantos dias manter no JSON final (hoje + proximos)
DIAS_A_MANTER = 3

TZ_BR = timezone(timedelta(hours=-3))
UA = {"User-Agent": "agenda-esportiva-bot/1.0 (uso pessoal)"}

# ----------------------------------------------------------------------------
# UTILITARIOS
# ----------------------------------------------------------------------------


def log(msg):
    print(f"[{datetime.now(TZ_BR):%H:%M:%S}] {msg}", flush=True)


def normalizar(txt):
    """Minusculas, sem acento, sem espaco duplo. Usado so para comparar."""
    txt = unicodedata.normalize("NFKD", txt or "")
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", txt).strip().lower()


def limpar(txt):
    """Limpeza leve para texto que vai aparecer no app."""
    txt = re.sub(r"\s+", " ", txt or "").strip()
    return txt.strip(" -|.,;")


def normalizar_hora(bruto):
    """
    Converte '20h00', '20:00', '2Oh0O' etc para 'HH:MM'.
    Devolve None se nao parecer hora.
    """
    if not bruto:
        return None
    t = bruto.strip()
    # confusoes classicas de OCR em digito
    t = t.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    t = t.replace("S", "5").replace("B", "8")
    m = re.search(r"(\d{1,2})\s*[h:.]\s*(\d{2})", t)
    if not m:
        return None
    hora, minuto = int(m.group(1)), int(m.group(2))
    if hora > 23 or minuto > 59:
        return None
    return f"{hora:02d}:{minuto:02d}"


def parecido(a, b, corte=0.72):
    return SequenceMatcher(None, normalizar(a), normalizar(b)).ratio() >= corte


def dentro_do_escopo(evento):
    if not ESCOPO:
        return True
    alvo = normalizar(f"{evento.get('competicao','')} {evento.get('evento','')}")
    return any(normalizar(termo) in alvo for termo in ESCOPO)


def salvar_debug(nome, conteudo):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    caminho = os.path.join(DEBUG_DIR, nome)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    log(f"debug gravado em {caminho}")


# ----------------------------------------------------------------------------
# FONTE 1 — ESPORTES NA TV (BLUESKY + OCR)
# ----------------------------------------------------------------------------


def bluesky_posts_de_agenda():
    """
    Devolve [(data_iso, [url_imagem, ...]), ...] dos posts de agenda diaria.
    Nos fins de semana o post traz 3 imagens; todas precisam ser lidas.
    """
    url = f"{BSKY_API}?actor={DID}&limit={BSKY_LIMITE}"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    feed = r.json().get("feed", [])

    resultado = []
    for item in feed:
        post = item.get("post", {})
        record = post.get("record", {})
        texto = record.get("text", "")

        if "agenda esportiva" not in normalizar(texto):
            continue

        m = re.search(r"\((\d{2})/(\d{2})/(\d{4})\)", texto)
        if not m:
            continue
        data_iso = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

        imagens = (post.get("embed") or {}).get("images") or []
        urls = [img["fullsize"] for img in imagens if img.get("fullsize")]
        if not urls:
            continue

        if not any(d == data_iso for d, _ in resultado):
            resultado.append((data_iso, urls))

    log(f"Bluesky: {len(resultado)} dia(s) de agenda encontrados")
    return resultado


def _faixas_claras(perfil, limiar, minimo=2):
    """
    Recebe um perfil de brilho (media por linha ou por coluna) e devolve
    as faixas continuas mais claras que o limiar — os separadores brancos
    da tabela. Ignora faixas com menos de `minimo` pixels.
    """
    faixas = []
    inicio = None
    for i, valor in enumerate(perfil):
        if valor >= limiar:
            if inicio is None:
                inicio = i
        else:
            if inicio is not None and i - inicio >= minimo:
                faixas.append((inicio, i))
            inicio = None
    if inicio is not None and len(perfil) - inicio >= minimo:
        faixas.append((inicio, len(perfil)))
    return faixas


def _blocos_entre(faixas, tamanho, minimo=12):
    """Converte separadores em blocos de conteudo (o que sobra entre eles)."""
    blocos = []
    cursor = 0
    for ini, fim in faixas:
        if ini - cursor >= minimo:
            blocos.append((cursor, ini))
        cursor = fim
    if tamanho - cursor >= minimo:
        blocos.append((cursor, tamanho))
    return blocos


def ler_celula(img, psm=7):
    """
    OCR de uma celula isolada. Amplia 3x antes de ler — texto pequeno
    ampliado eh onde o Tesseract mais ganha precisao.
    """
    if img.width < 5 or img.height < 5:
        return ""
    grande = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    cfg = f"--oem 3 --psm {psm}"
    try:
        txt = pytesseract.image_to_string(grande, lang="por", config=cfg)
    except pytesseract.TesseractError:
        txt = pytesseract.image_to_string(grande, config=cfg)
    return limpar(txt)


def ler_agenda_imagem(url, data_iso, indice=0):
    """
    Baixa a imagem da agenda e le celula a celula.
    A tabela tem separadores brancos entre linhas e entre colunas; achamos
    esses separadores pelo brilho e recortamos o que sobra. Assim o OCR
    nunca embaralha colunas, que eh o erro classico em tabela.
    """
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cinza = np.array(img.convert("L"), dtype=np.float32)
    altura, largura = cinza.shape

    # --- linhas -------------------------------------------------------------
    perfil_linhas = cinza.mean(axis=1)
    limiar_linha = max(235.0, float(np.percentile(perfil_linhas, 90)))
    seps_linha = _faixas_claras(perfil_linhas, limiar_linha, minimo=2)
    linhas = _blocos_entre(seps_linha, altura, minimo=14)

    if len(linhas) < 2:
        # sem separador detectavel: divide em faixas de altura constante
        estimativa = 26
        n = max(1, altura // estimativa)
        passo = altura / n
        linhas = [(int(i * passo), int((i + 1) * passo)) for i in range(n)]

    # a primeira faixa eh o cabecalho escuro ("AGENDA ESPORTIVA - ...")
    cabecalho = cinza[linhas[0][0]:linhas[0][1]].mean() < 120
    if cabecalho:
        linhas = linhas[1:]

    # --- colunas ------------------------------------------------------------
    corpo_ini = linhas[0][0] if linhas else 0
    corpo = cinza[corpo_ini:, :]
    perfil_colunas = corpo.mean(axis=0)
    limiar_coluna = max(235.0, float(np.percentile(perfil_colunas, 88)))
    seps_coluna = _faixas_claras(perfil_colunas, limiar_coluna, minimo=2)
    colunas = _blocos_entre(seps_coluna, largura, minimo=40)

    if len(colunas) != 4:
        # proporcoes medidas nas imagens do perfil; a largura varia
        # (615, 635, 655 px), por isso o corte eh proporcional
        cortes = [0.0, 0.155, 0.470, 0.815, 1.0]
        colunas = [
            (int(cortes[i] * largura), int(cortes[i + 1] * largura))
            for i in range(4)
        ]
        log(f"  imagem {indice}: colunas por proporcao (detectadas: {len(seps_coluna)+1})")

    eventos = []
    for (y0, y1) in linhas:
        margem = 1
        celulas = []
        for (x0, x1) in colunas[:4]:
            recorte = img.crop((x0 + margem, y0 + margem, x1 - margem, y1 - margem))
            celulas.append(recorte)

        hora = normalizar_hora(ler_celula(celulas[0], psm=7))
        if not hora:
            continue  # linha sem hora valida nao eh linha de evento

        competicao = limpar(ler_celula(celulas[1], psm=7))
        evento = limpar(ler_celula(celulas[2], psm=7))
        canais = limpar(ler_celula(celulas[3], psm=7))

        # a coluna 2 traz um icone antes do nome; sobra sujeira no inicio
        competicao = re.sub(r"^[^A-Za-z0-9ÀÁÂÃÉÊÍÓÔÕÚÇ]+", "", competicao)

        eventos.append(
            {
                "hora": hora,
                "competicao": competicao,
                "evento": evento,
                "canais": canais,
                "fonte": "esportesnatv",
                "confianca": "ocr",
            }
        )

    log(f"  imagem {indice} ({data_iso}): {len(eventos)} evento(s) lidos")
    return eventos


def coletar_esportesnatv():
    dias = {}
    for data_iso, urls in bluesky_posts_de_agenda():
        eventos = []
        for i, url in enumerate(urls):
            try:
                eventos.extend(ler_agenda_imagem(url, data_iso, i))
            except Exception as e:
                log(f"  ERRO ao ler imagem {i} de {data_iso}: {e}")
        if eventos:
            dias[data_iso] = eventos
    return dias


# ----------------------------------------------------------------------------
# FONTE 2 — DOENTES POR FUTEBOL (HTML, so futebol)
# ----------------------------------------------------------------------------


def coletar_dpf():
    """
    Estrutura da pagina:
        TERCA-FEIRA - 22/09/2026
        (relogio) 20:00 Campeonato Brasileiro Serie A Sub-17
        Sao Paulo x Palmeiras
        (tv) SPORTV
    """
    r = requests.get(DPF_URL, headers=UA, timeout=45)
    r.raise_for_status()
    sopa = BeautifulSoup(r.text, "html.parser")
    for tag in sopa(["script", "style", "nav", "footer"]):
        tag.decompose()

    linhas = [limpar(l) for l in sopa.get_text("\n").split("\n")]
    linhas = [l for l in linhas if l]

    dias = {}
    data_atual = None
    i = 0
    while i < len(linhas):
        linha = linhas[i]

        m_data = re.search(r"(\d{2})/(\d{2})/(\d{4})", linha)
        if m_data and re.search(r"(SEGUNDA|TER[CÇ]A|QUARTA|QUINTA|SEXTA|S[ÁA]BADO|DOMINGO)",
                                linha, re.IGNORECASE):
            data_atual = f"{m_data.group(3)}-{m_data.group(2)}-{m_data.group(1)}"
            dias.setdefault(data_atual, [])
            i += 1
            continue

        m_hora = re.match(r"^[^\d]{0,4}(\d{1,2}[:h]\d{2})\s+(.*)$", linha)
        if m_hora and data_atual:
            hora = normalizar_hora(m_hora.group(1))
            competicao = limpar(m_hora.group(2))
            evento, canais = "", ""
            for j in range(i + 1, min(i + 4, len(linhas))):
                seguinte = linhas[j]
                if "📺" in seguinte or normalizar(seguinte).startswith("tv"):
                    canais = limpar(seguinte.replace("📺", ""))
                    i = j
                    break
                if not evento:
                    evento = limpar(seguinte)
            if hora and evento:
                dias[data_atual].append(
                    {
                        "hora": hora,
                        "competicao": competicao,
                        "evento": evento,
                        "canais": canais,
                        "fonte": "dpf",
                        "confianca": "texto",
                    }
                )
        i += 1

    total = sum(len(v) for v in dias.values())
    log(f"DPF: {total} evento(s) em {len(dias)} dia(s)")
    return dias


# ----------------------------------------------------------------------------
# FONTE 3 — TOMADA DE TEMPO (automobilismo, WordPress REST)
# ----------------------------------------------------------------------------


def coletar_tomada_de_tempo():
    """
    O portal eh WordPress, entao a API REST responde sem chave.
    Pegamos os posts de programacao e varremos o texto atras de
    linhas com horario. O formato interno do post ainda nao foi
    conferido: o primeiro post vira debug/tdt_exemplo.txt para ajuste.
    """
    params = {
        "per_page": 15,
        "search": "Programação, horários e transmissão",
        "orderby": "date",
        "order": "desc",
    }
    try:
        r = requests.get(TDT_API, params=params, headers=UA, timeout=45)
        r.raise_for_status()
        posts = r.json()
    except Exception as e:
        log(f"Tomada de Tempo: falhou ({e})")
        return {}

    dias = {}
    for n, post in enumerate(posts):
        titulo = limpar(BeautifulSoup(post["title"]["rendered"], "html.parser").get_text())
        html = post["content"]["rendered"]
        texto = BeautifulSoup(html, "html.parser").get_text("\n")
        linhas = [limpar(l) for l in texto.split("\n")]
        linhas = [l for l in linhas if l]

        if n == 0:
            salvar_debug("tdt_exemplo.txt", f"{titulo}\n\n" + "\n".join(linhas))

        # categoria = primeira parte do titulo, antes do travessao
        categoria = limpar(re.split(r"[–-]", titulo)[0]) or "Automobilismo"
        data_post = post["date"][:10]
        data_atual = data_post

        for linha in linhas:
            m_data = re.search(r"(\d{2})/(\d{2})/(\d{4})", linha)
            if m_data and len(linha) < 80:
                data_atual = f"{m_data.group(3)}-{m_data.group(2)}-{m_data.group(1)}"

            m = re.match(r"^(\d{1,2}[h:]\d{2})\s*[-–—:]?\s*(.+)$", linha)
            if not m:
                continue
            hora = normalizar_hora(m.group(1))
            descricao = limpar(m.group(2))
            if not hora or len(descricao) < 3:
                continue

            canais = ""
            m_canal = re.search(r"\(([^)]*(?:tv|sportv|band|espn|youtube|globo)[^)]*)\)",
                                descricao, re.IGNORECASE)
            if m_canal:
                canais = limpar(m_canal.group(1))
                descricao = limpar(descricao.replace(m_canal.group(0), ""))

            dias.setdefault(data_atual, []).append(
                {
                    "hora": hora,
                    "competicao": categoria,
                    "evento": descricao,
                    "canais": canais,
                    "fonte": "tomadadetempo",
                    "confianca": "texto",
                }
            )

    total = sum(len(v) for v in dias.values())
    log(f"Tomada de Tempo: {total} evento(s) em {len(dias)} dia(s)")
    return dias


# ----------------------------------------------------------------------------
# CRUZAMENTO E MONTAGEM
# ----------------------------------------------------------------------------


def corrigir_ocr_com_texto(eventos_ocr, eventos_texto):
    """
    Onde o mesmo jogo aparece nas duas fontes, o texto manda.
    Se o OCR leu 'S4O PAULO' as 20:00 e o DPF diz 'Sao Paulo' as 20:00,
    o registro corrigido entra no lugar do lido por OCR.
    """
    corrigidos = 0
    saida = []
    for ev in eventos_ocr:
        melhor = None
        for ref in eventos_texto:
            if ref["hora"] != ev["hora"]:
                continue
            if parecido(ref["evento"], ev["evento"]):
                melhor = ref
                break
        if melhor:
            novo = dict(melhor)
            novo["fonte"] = "dpf+esportesnatv"
            novo["confianca"] = "conferido"
            saida.append(novo)
            corrigidos += 1
        else:
            saida.append(ev)
    if corrigidos:
        log(f"  {corrigidos} evento(s) de OCR conferidos pelo texto")
    return saida


def deduplicar(eventos):
    vistos = []
    saida = []
    ordem = {"conferido": 0, "texto": 1, "ocr": 2}
    for ev in sorted(eventos, key=lambda e: ordem.get(e["confianca"], 3)):
        chave = (ev["hora"], normalizar(ev["evento"])[:40])
        duplicado = any(
            k[0] == chave[0] and parecido(k[1], chave[1], 0.8) for k in vistos
        )
        if duplicado:
            continue
        vistos.append(chave)
        saida.append(ev)
    return saida


def montar():
    agora = datetime.now(TZ_BR)
    hoje = agora.date().isoformat()

    por_dia = {}

    def juntar(bloco):
        for data_iso, eventos in bloco.items():
            por_dia.setdefault(data_iso, []).extend(eventos)

    try:
        dpf = coletar_dpf()
    except Exception as e:
        log(f"DPF falhou: {e}")
        dpf = {}

    try:
        entv = coletar_esportesnatv()
    except Exception as e:
        log(f"Esportes na TV falhou: {e}")
        entv = {}

    # cruzamento antes de juntar
    for data_iso in list(entv.keys()):
        entv[data_iso] = corrigir_ocr_com_texto(entv[data_iso], dpf.get(data_iso, []))

    juntar(dpf)
    juntar(entv)

    try:
        juntar(coletar_tomada_de_tempo())
    except Exception as e:
        log(f"Tomada de Tempo falhou: {e}")

    dias = []
    for data_iso in sorted(por_dia):
        if data_iso < hoje:
            continue
        eventos = [e for e in deduplicar(por_dia[data_iso]) if dentro_do_escopo(e)]
        eventos.sort(key=lambda e: (e["hora"], normalizar(e["competicao"])))
        if eventos:
            dias.append({"data": data_iso, "eventos": eventos})

    dias = dias[:DIAS_A_MANTER]

    return {
        "gerado_em": agora.isoformat(timespec="seconds"),
        "fontes": [
            "Esportes na TV (Bluesky)",
            "Doentes por Futebol",
            "Tomada de Tempo",
        ],
        "total": sum(len(d["eventos"]) for d in dias),
        "dias": dias,
    }


def main():
    dados = montar()
    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    log(f"OK: {dados['total']} evento(s) em {len(dados['dias'])} dia(s) -> {SAIDA}")
    if dados["total"] == 0:
        log("AVISO: nenhum evento coletado")
        sys.exit(1)


if __name__ == "__main__":
    main()
