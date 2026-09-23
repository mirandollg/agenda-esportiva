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
BSKY_LIMITE = 20  # quantos posts puxar do feed (a agenda eh diaria)

DPF_URL = "https://doentesporfutebol.com.br/guiadejogos/"
TDT_API = "https://www.tomadadetempo.com.br/wp-json/wp/v2/posts"

SAIDA = "docs/app.json"
DEBUG_DIR = "debug"

# A PARTIR DE QUANDO COLETAR
#   0 = de hoje em diante
#   1 = so de amanha em diante
# O corte eh aplicado ANTES de baixar a imagem.
PRIMEIRO_DIA = 0

# Quantos dias manter no JSON final, contados a partir do primeiro dia
DIAS_A_MANTER = 3

# Filtro de escopo. Lista vazia = manter tudo.
# Exemplo: ESCOPO = ["brasileiro", "libertadores", "copa do brasil", "f1", "motogp"]
ESCOPO = []

# Proporcoes de largura das 4 colunas, usadas so se a deteccao falhar
CORTES_PADRAO = [0.0, 0.125, 0.445, 0.805, 1.0]

TZ_BR = timezone(timedelta(hours=-3))
UA = {"User-Agent": "agenda-esportiva-bot/1.0 (uso pessoal)"}

# ----------------------------------------------------------------------------
# UTILITARIOS
# ----------------------------------------------------------------------------


def log(msg):
    print(f"[{datetime.now(TZ_BR):%H:%M:%S}] {msg}", flush=True)


def data_corte():
    """Primeira data que interessa, em ISO."""
    return (datetime.now(TZ_BR).date() + timedelta(days=PRIMEIRO_DIA)).isoformat()


def normalizar(txt):
    """Minusculas, sem acento, sem espaco duplo. Usado so para comparar."""
    txt = unicodedata.normalize("NFKD", txt or "")
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", txt).strip().lower()


def limpar(txt):
    """Limpeza leve para texto que vai aparecer no app."""
    txt = re.sub(r"\s+", " ", txt or "").strip()
    return txt.strip(" -|.,;")


def limpar_competicao(txt):
    """
    A coluna 2 traz um icone antes do nome. O OCR le o desenho como lixo
    ('5', '6)', '69'). Corta tudo antes da primeira palavra de verdade.
    """
    txt = limpar(txt)
    m = re.search(r"[A-Za-zÀ-ÿ]{3,}", txt)
    return txt[m.start():].strip() if m else txt


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
    Devolve [(data_iso, [url_imagem, ...]), ...] SO dos dias que interessam.

    O feed vem do mais novo para o mais antigo. Assim que aparece um post de
    agenda anterior ao corte, para de varrer: o resto eh historico.
    Nos fins de semana o post traz 3 imagens; todas entram.
    """
    corte = data_corte()
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

        if data_iso < corte:
            break  # daqui para tras eh tudo mais antigo

        if any(d == data_iso for d, _ in resultado):
            continue  # o feed repete o post quando ele tem respostas

        imagens = (post.get("embed") or {}).get("images") or []
        urls = [img["fullsize"] for img in imagens if img.get("fullsize")]
        if urls:
            resultado.append((data_iso, urls))

    total_imgs = sum(len(u) for _, u in resultado)
    log(f"Bluesky: {len(resultado)} dia(s) a partir de {corte}, {total_imgs} imagem(ns)")
    return resultado


def _faixas_claras(perfil, limiar, minimo=1):
    """
    Recebe um perfil de brilho e devolve as faixas continuas acima do
    limiar — os separadores brancos da tabela.
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


def detectar_colunas(corpo, largura):
    """
    Acha as 4 colunas pelo brilho MINIMO de cada coluna de pixels.

    Um separador branco eh branco em TODAS as linhas, entao seu minimo
    fica alto. Qualquer coluna que cruze texto tem minimo baixo, mesmo
    que a media seja clara. Por isso o minimo separa muito melhor que a
    media — a media se dilui entre as celulas coloridas.
    """
    perfil_min = corpo.min(axis=0)

    for limiar in (215, 200, 185, 170):
        seps = _faixas_claras(perfil_min, limiar, minimo=1)
        # descarta separadores grudados na borda
        seps = [(a, b) for (a, b) in seps if a > largura * 0.04 and b < largura * 0.97]
        blocos = _blocos_entre(seps, largura, minimo=int(largura * 0.05))
        if len(blocos) == 4:
            return blocos, f"minimo>={limiar}"

    cortes = CORTES_PADRAO
    blocos = [
        (int(cortes[i] * largura), int(cortes[i + 1] * largura)) for i in range(4)
    ]
    return blocos, "proporcao padrao"


def detectar_linhas(cinza, altura, largura):
    """Mesma ideia das colunas, aplicada na horizontal."""
    perfil_min = cinza.min(axis=1)
    for limiar in (215, 200, 185):
        seps = _faixas_claras(perfil_min, limiar, minimo=1)
        linhas = _blocos_entre(seps, altura, minimo=14)
        if len(linhas) >= 3:
            return linhas
    # ultimo recurso: media
    perfil_media = cinza.mean(axis=1)
    seps = _faixas_claras(perfil_media, max(235.0, float(np.percentile(perfil_media, 90))), 2)
    return _blocos_entre(seps, altura, minimo=14)


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
    Achamos os separadores brancos e recortamos o que sobra entre eles.
    Assim o OCR nunca embaralha colunas, que eh o erro classico em tabela.
    """
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cinza = np.array(img.convert("L"), dtype=np.float32)
    altura, largura = cinza.shape

    linhas = detectar_linhas(cinza, altura, largura)
    if not linhas:
        log(f"  imagem {indice}: nenhuma linha detectada")
        return []

    # a primeira faixa eh o cabecalho escuro ("AGENDA ESPORTIVA - ...")
    if cinza[linhas[0][0]:linhas[0][1]].mean() < 120:
        linhas = linhas[1:]
    if not linhas:
        return []

    corpo = cinza[linhas[0][0]:, :]
    colunas, metodo = detectar_colunas(corpo, largura)
    log(
        f"  imagem {indice} ({largura}x{altura}): {len(linhas)} linha(s), "
        f"colunas por {metodo} -> {[c for c in colunas]}"
    )

    eventos = []
    for (y0, y1) in linhas:
        celulas = []
        for (x0, x1) in colunas[:4]:
            # 1px de folga para nao encostar no separador
            celulas.append(img.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1)))

        hora = normalizar_hora(ler_celula(celulas[0], psm=7))
        if not hora:
            continue  # linha sem hora valida nao eh linha de evento

        eventos.append(
            {
                "hora": hora,
                "competicao": limpar_competicao(ler_celula(celulas[1], psm=7)),
                "evento": limpar(ler_celula(celulas[2], psm=7)),
                "canais": limpar(ler_celula(celulas[3], psm=7)),
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


def _quebrar_evento_dpf(linha):
    """
    A pagina usa <br> dentro do mesmo bloco, entao a entrada costuma vir
    numa linha so:
        '13:45 UEFA Champions League Feminina Bayern x Man City ESPN4'
    Separa pelos simbolos de relogio e de TV.
    """
    linha = linha.replace("🕗", "").strip()
    partes = [limpar(p) for p in linha.split("📺")]
    principal = partes[0]
    canais = limpar(partes[1]) if len(partes) > 1 else ""

    m = re.match(r"^\s*(\d{1,2}[:h]\d{2})\s+(.*)$", principal)
    if not m:
        return None
    hora = normalizar_hora(m.group(1))
    resto = limpar(m.group(2))
    return hora, resto, canais


def coletar_dpf():
    """
    Estrutura da pagina:
        TERCA-FEIRA - 22/09/2026
        (relogio) 20:00 Campeonato Brasileiro Serie A Sub-17
        Sao Paulo x Palmeiras
        (tv) SPORTV
    """
    corte = data_corte()
    r = requests.get(DPF_URL, headers=UA, timeout=45)
    r.raise_for_status()
    sopa = BeautifulSoup(r.text, "html.parser")
    for tag in sopa(["script", "style", "nav", "footer"]):
        tag.decompose()

    # <br> nao produz texto: sem isso, hora, jogo e canal grudam numa linha so
    for br in sopa.find_all("br"):
        br.replace_with("\n")

    linhas = [limpar(l) for l in sopa.get_text("\n").split("\n")]
    linhas = [l for l in linhas if l]
    salvar_debug("dpf_texto.txt", "\n".join(linhas))

    dias = {}
    data_atual = None
    i = 0
    while i < len(linhas):
        linha = linhas[i]

        m_data = re.search(r"(\d{2})/(\d{2})/(\d{4})", linha)
        if m_data and re.search(
            r"(SEGUNDA|TER[CÇ]A|QUARTA|QUINTA|SEXTA|S[ÁA]BADO|DOMINGO)", linha, re.I
        ):
            data_atual = f"{m_data.group(3)}-{m_data.group(2)}-{m_data.group(1)}"
            if data_atual >= corte:
                dias.setdefault(data_atual, [])
            i += 1
            continue

        tem_hora = re.match(r"^[^\d]{0,4}(\d{1,2}[:h]\d{2})\s+", linha)
        if tem_hora and data_atual and data_atual >= corte:
            hora, competicao, canais = (None, "", "")
            quebrado = _quebrar_evento_dpf(linha)
            if quebrado:
                hora, competicao, canais = quebrado

            evento = ""
            if not canais:
                # caso o <br> tenha virado quebra de verdade: le as proximas
                for j in range(i + 1, min(i + 4, len(linhas))):
                    seguinte = linhas[j]
                    if "📺" in seguinte:
                        canais = limpar(seguinte.replace("📺", ""))
                        i = j
                        break
                    if not evento:
                        evento = limpar(seguinte)
            else:
                # tudo veio junto: o jogo eh o final do texto, depois da
                # competicao. Separa no ultimo ' x ' encontrado.
                m_jogo = re.search(r"^(.*?)\s+([^,;]+\s+x\s+[^,;]+)$", competicao, re.I)
                if m_jogo:
                    competicao, evento = limpar(m_jogo.group(1)), limpar(m_jogo.group(2))

            if hora and (evento or competicao):
                dias.setdefault(data_atual, []).append(
                    {
                        "hora": hora,
                        "competicao": competicao,
                        "evento": evento or competicao,
                        "canais": canais,
                        "fonte": "dpf",
                        "confianca": "texto",
                    }
                )
        i += 1

    total = sum(len(v) for v in dias.values())
    log(f"DPF: {total} evento(s) em {len(dias)} dia(s) a partir de {corte}")
    return dias


# ----------------------------------------------------------------------------
# FONTE 3 — TOMADA DE TEMPO (automobilismo, WordPress REST)
# ----------------------------------------------------------------------------


def coletar_tomada_de_tempo():
    """
    O portal eh WordPress, entao a API REST responde sem chave.
    O formato interno do post ainda nao foi conferido: o primeiro post
    vira debug/tdt_exemplo.txt para ajuste.
    """
    corte = data_corte()
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
        titulo = limpar(
            BeautifulSoup(post["title"]["rendered"], "html.parser").get_text()
        )
        sopa = BeautifulSoup(post["content"]["rendered"], "html.parser")
        for br in sopa.find_all("br"):
            br.replace_with("\n")
        linhas = [limpar(l) for l in sopa.get_text("\n").split("\n")]
        linhas = [l for l in linhas if l]

        if n == 0:
            salvar_debug("tdt_exemplo.txt", f"{titulo}\n\n" + "\n".join(linhas))

        # categoria = primeira parte do titulo, antes do travessao
        categoria = limpar(re.split(r"[–-]", titulo)[0]) or "Automobilismo"
        data_atual = post["date"][:10]

        for linha in linhas:
            m_data = re.search(r"(\d{2})/(\d{2})/(\d{4})", linha)
            if m_data and len(linha) < 80:
                data_atual = f"{m_data.group(3)}-{m_data.group(2)}-{m_data.group(1)}"

            if data_atual < corte:
                continue

            m = re.match(r"^(\d{1,2}[h:]\d{2})\s*[-–—:]?\s*(.+)$", linha)
            if not m:
                continue
            hora = normalizar_hora(m.group(1))
            descricao = limpar(m.group(2))
            if not hora or len(descricao) < 3:
                continue

            canais = ""
            m_canal = re.search(
                r"\(([^)]*(?:tv|sportv|band|espn|youtube|globo)[^)]*)\)",
                descricao,
                re.IGNORECASE,
            )
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
    log(f"Tomada de Tempo: {total} evento(s) em {len(dias)} dia(s) a partir de {corte}")
    return dias


# ----------------------------------------------------------------------------
# CRUZAMENTO E MONTAGEM
# ----------------------------------------------------------------------------


def corrigir_ocr_com_texto(eventos_ocr, eventos_texto):
    """
    Onde o mesmo jogo aparece nas duas fontes, o texto manda.
    Compara por hora e por semelhanca do confronto; o OCR erra letras,
    mas raramente a ponto de derrubar a semelhanca.
    """
    corrigidos = 0
    saida = []
    for ev in eventos_ocr:
        melhor = None
        for ref in eventos_texto:
            if ref["hora"] != ev["hora"]:
                continue
            alvo_ocr = f"{ev['competicao']} {ev['evento']}"
            alvo_ref = f"{ref['competicao']} {ref['evento']}"
            if parecido(ref["evento"], ev["evento"]) or parecido(alvo_ref, alvo_ocr, 0.6):
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
    corte = data_corte()
    log(f"Coletando a partir de {corte} (PRIMEIRO_DIA={PRIMEIRO_DIA})")

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
        if data_iso < corte:
            continue
        eventos = [e for e in deduplicar(por_dia[data_iso]) if dentro_do_escopo(e)]
        eventos.sort(key=lambda e: (e["hora"], normalizar(e["competicao"])))
        if eventos:
            dias.append({"data": data_iso, "eventos": eventos})

    dias = dias[:DIAS_A_MANTER]

    return {
        "gerado_em": agora.isoformat(timespec="seconds"),
        "primeiro_dia": corte,
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
