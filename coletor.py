#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coletor de agenda esportiva.

Fontes (as duas por OCR de imagem):
  1. Esportes na TV  -> Bluesky, imagem em 4 colunas
  2. Tomada de Tempo -> canal do Telegram, imagem em 5 colunas

O site do Tomada de Tempo devolve 403 para robo, entao a imagem do
Telegram eh o caminho: ela ja traz a data no cabecalho e a grade inteira.

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
from PIL import Image, ImageOps
import pytesseract

# ----------------------------------------------------------------------------
# CONFIGURACAO
# ----------------------------------------------------------------------------

# Esportes na TV. Usamos o DID, nao o handle: o handle pode mudar se eles
# registrarem dominio proprio, o DID nunca muda.
DID = "did:plc:lngl4ki52wbunv2xzq74bqbb"
BSKY_API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
BSKY_LIMITE = 20

# Tomada de Tempo: versao web do canal publico, sem login e sem bloqueio
TDT_CANAL = "https://t.me/s/tomadadetempo"
TDT_DIAS_DE_FEED = 8     # so olha mensagens desta janela
TDT_MAX_IMAGENS = 15     # teto de seguranca por execucao

SAIDA = "docs/app.json"
DEBUG_DIR = "debug"

# A PARTIR DE QUANDO COLETAR
#   0 = normal, de hoje em diante
#   1 = so de amanha em diante
#  -7 = modo de teste: aceita a ultima semana, util para conferir a
#       leitura de imagens ja publicadas
PRIMEIRO_DIA = 0

# Quantos dias manter no JSON final
DIAS_A_MANTER = 3

# Filtro de escopo. Lista vazia = manter tudo.
# Exemplo: ESCOPO = ["brasileirao", "wnba", "moto gp", "nascar"]
ESCOPO = []

# Proporcoes de largura, usadas so quando a deteccao falha.
# Esportes na TV: hora | competicao | evento | canal
CORTES_ENTV = [(0.000, 0.125), (0.125, 0.445), (0.445, 0.805), (0.805, 1.000)]
# Tomada de Tempo: hora | categoria | etapa | sessao | transmissao
# (a coluna estreita do "B" de piloto brasileiro fica fora de proposito)
CORTES_TDT = [
    (0.004, 0.069),
    (0.069, 0.335),
    (0.335, 0.532),
    (0.532, 0.727),
    (0.750, 0.995),
]

# Erros recorrentes do OCR nos nomes de canal
CORRECOES_CANAL = {
    "voutube": "YOUTUBE",
    "vutube": "YOUTUBE",
    "youtub": "YOUTUBE",
    "youtube": "YOUTUBE",
    "espna": "ESPN4",
    "espna4": "ESPN4",
    "disnev+": "DISNEY+",
    "disnev+ pr": "DISNEY+ PR",
    "disney + pr": "DISNEY+ PR",
    "disney +": "DISNEY+",
    "phiztyv": "PHIZTV",
    "phizty": "PHIZTV",
    "bandsports": "BANDSPORTS",
    "xsports": "XSPORTS",
}

TZ_BR = timezone(timedelta(hours=-3))
# Alguns sites recusam requisicao sem cara de navegador
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
}

# ----------------------------------------------------------------------------
# UTILITARIOS
# ----------------------------------------------------------------------------


def log(msg):
    print(f"[{datetime.now(TZ_BR):%H:%M:%S}] {msg}", flush=True)


def data_corte():
    return (datetime.now(TZ_BR).date() + timedelta(days=PRIMEIRO_DIA)).isoformat()


def normalizar(txt):
    txt = unicodedata.normalize("NFKD", txt or "")
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", txt).strip().lower()


def limpar(txt):
    txt = re.sub(r"\s+", " ", txt or "").strip()
    return txt.strip(" -|.,;")


def limpar_competicao(txt):
    """Corta o lixo que o OCR gera ao tentar ler o icone antes do nome."""
    txt = limpar(txt)
    m = re.search(r"[A-Za-zÀ-ÿ]{3,}", txt)
    return txt[m.start():].strip() if m else txt


def limpar_confronto(txt):
    """
    'Bayern Fx Manchester City F' -> 'Bayern F x Manchester City F'.
    So age quando ha UMA maiuscula isolada antes do x, entao nao estraga
    nomes terminados em x, como 'Red Sox x Guardians'.
    """
    txt = limpar(txt)
    return re.sub(r"\b([A-Z])x(?=\s)", r"\1 x", txt)


def limpar_canais(txt):
    txt = limpar(txt)
    partes = [p.strip() for p in re.split(r"[,;|]", txt) if p.strip()]
    saida = []
    for parte in partes:
        saida.append(CORRECOES_CANAL.get(normalizar(parte), parte))
    return ", ".join(saida)


def normalizar_hora(bruto):
    """'20h00', '20:00', '2Oh0O' -> 'HH:MM'. None se nao parecer hora."""
    if not bruto:
        return None
    t = bruto.strip()
    t = t.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    t = t.replace("S", "5").replace("B", "8")
    m = re.search(r"(\d{1,2})\s*[h:.]\s*(\d{2})", t)
    if not m:
        return None
    hora, minuto = int(m.group(1)), int(m.group(2))
    if hora > 23 or minuto > 59:
        return None
    return f"{hora:02d}:{minuto:02d}"


def data_de_texto(txt):
    m = re.search(r"(\d{2})\s*/\s*(\d{2})\s*/\s*(\d{4})", txt or "")
    if not m:
        return None
    dia, mes, ano = m.group(1), m.group(2), m.group(3)
    try:
        datetime(int(ano), int(mes), int(dia))
    except ValueError:
        return None
    return f"{ano}-{mes}-{dia}"


def parecido(a, b, corte=0.8):
    return SequenceMatcher(None, normalizar(a), normalizar(b)).ratio() >= corte


def dentro_do_escopo(evento):
    if not ESCOPO:
        return True
    alvo = normalizar(
        f"{evento.get('competicao','')} {evento.get('evento','')}"
    )
    return any(normalizar(termo) in alvo for termo in ESCOPO)


def salvar_debug(nome, conteudo):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with open(os.path.join(DEBUG_DIR, nome), "w", encoding="utf-8") as f:
        f.write(conteudo)


def baixar_imagem(url):
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# ----------------------------------------------------------------------------
# OCR — BLOCOS COMUNS
# ----------------------------------------------------------------------------


def ler_celula(img, psm=7, inverter=False):
    """Amplia 3x antes de ler: texto pequeno ampliado eh onde o Tesseract ganha."""
    if img.width < 5 or img.height < 5:
        return ""
    if inverter:
        img = ImageOps.invert(img.convert("L"))
    grande = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    cfg = f"--oem 3 --psm {psm}"
    try:
        txt = pytesseract.image_to_string(grande, lang="por", config=cfg)
    except pytesseract.TesseractError:
        txt = pytesseract.image_to_string(grande, config=cfg)
    return limpar(txt)


def _agrupar(contagem, gap_min, tam_min):
    """
    Recebe um vetor de contagem de tinta e devolve as faixas com texto,
    unindo trechos separados por intervalos menores que gap_min (espacos
    entre palavras nao devem virar separador de coluna).
    """
    faixas = []
    inicio = None
    vazio = 0
    for i, valor in enumerate(contagem):
        if valor > 0:
            if inicio is None:
                inicio = i
            vazio = 0
        else:
            if inicio is not None:
                vazio += 1
                if vazio >= gap_min:
                    fim = i - vazio + 1
                    if fim - inicio >= tam_min:
                        faixas.append((inicio, fim))
                    inicio = None
                    vazio = 0
    if inicio is not None and len(contagem) - inicio >= tam_min:
        faixas.append((inicio, len(contagem)))
    return faixas


def mascara_de_tinta(cinza, limiar=140):
    return (cinza < limiar).astype(np.int32)


def _primeiro_bloco(linhas):
    """
    Mantem so as linhas coladas umas nas outras, cortando no primeiro
    intervalo muito maior que o normal. Serve para separar a tabela do
    rodape, que fica isolado la embaixo depois de uma area branca.
    """
    if len(linhas) < 3:
        return linhas
    alturas = sorted(b - a for a, b in linhas)
    tipica = alturas[len(alturas) // 2]
    saida = [linhas[0]]
    for anterior, atual in zip(linhas, linhas[1:]):
        if atual[0] - anterior[1] > tipica * 3:
            break
        saida.append(atual)
    return saida


def faixas_por_proporcao(largura, cortes):
    return [(int(a * largura), int(b * largura)) for a, b in cortes]


# ----------------------------------------------------------------------------
# FONTE 1 — ESPORTES NA TV (BLUESKY)
# ----------------------------------------------------------------------------


def bluesky_posts_de_agenda():
    """
    [(data_iso, [url, ...])] so dos dias que interessam. O feed vem do mais
    novo para o mais antigo; ao achar post anterior ao corte, para de varrer.
    """
    corte = data_corte()
    r = requests.get(
        f"{BSKY_API}?actor={DID}&limit={BSKY_LIMITE}", headers=UA, timeout=30
    )
    r.raise_for_status()
    feed = r.json().get("feed", [])

    resultado = []
    for item in feed:
        post = item.get("post", {})
        texto = (post.get("record") or {}).get("text", "")
        if "agenda esportiva" not in normalizar(texto):
            continue
        data_iso = data_de_texto(texto)
        if not data_iso:
            continue
        if data_iso < corte:
            break
        if any(d == data_iso for d, _ in resultado):
            continue  # o feed repete o post quando ele tem respostas
        imagens = (post.get("embed") or {}).get("images") or []
        urls = [i["fullsize"] for i in imagens if i.get("fullsize")]
        if urls:
            resultado.append((data_iso, urls))

    log(
        f"Esportes na TV: {len(resultado)} dia(s) a partir de {corte}, "
        f"{sum(len(u) for _, u in resultado)} imagem(ns)"
    )
    return resultado


def _faixas_claras(perfil, limiar, minimo=1):
    faixas, inicio = [], None
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
    blocos, cursor = [], 0
    for ini, fim in faixas:
        if ini - cursor >= minimo:
            blocos.append((cursor, ini))
        cursor = fim
    if tamanho - cursor >= minimo:
        blocos.append((cursor, tamanho))
    return blocos


def assinatura_icone(celula):
    """
    Devolve uma assinatura do desenho da modalidade, ou "" se nao achar.

    O icone eh a unica coisa colorida da celula — o nome da competicao eh
    texto preto. Entao filtramos por saturacao, pegamos o bloco mais a
    esquerda (o desenho vem antes do nome), reduzimos a 6x6 e gravamos
    cada canal em um digito hexadecimal. Icones iguais geram assinaturas
    iguais, e eh isso que permite agrupar modalidades sem saber o nome.
    """
    arr = np.asarray(celula.convert("RGB"), dtype=np.int16)
    if arr.size == 0:
        return ""
    maximo = arr.max(axis=2)
    minimo = arr.min(axis=2)
    colorido = ((maximo - minimo) > 38) & (maximo > 55)
    if colorido.sum() < 10:
        return ""

    ys, xs = np.nonzero(colorido)
    lado = celula.height
    # o icone eh quadrado e fica na esquerda; ignora cor que apareca depois
    limite = xs.min() + lado + 2
    dentro = xs <= limite
    ys, xs = ys[dentro], xs[dentro]
    if len(xs) < 10:
        return ""

    recorte = celula.crop((int(xs.min()), int(ys.min()),
                           int(xs.max()) + 1, int(ys.max()) + 1))
    mini = recorte.resize((6, 6), Image.LANCZOS).convert("RGB")
    dados = np.asarray(mini, dtype=np.int16) // 16  # 16 niveis por canal
    return "".join(f"{v:x}" for v in dados.flatten())


def ler_agenda_entv(url, data_iso, indice=0):
    """
    Grade do Esportes na TV: separadores BRANCOS entre celulas.
    Achamos os separadores pelo brilho minimo — um separador eh branco em
    todas as linhas, entao seu minimo fica alto; qualquer coluna que cruze
    texto tem minimo baixo, mesmo que a media seja clara.
    """
    img = baixar_imagem(url)
    cinza = np.array(img.convert("L"), dtype=np.float32)
    altura, largura = cinza.shape

    linhas = []
    for limiar in (215, 200, 185):
        seps = _faixas_claras(cinza.min(axis=1), limiar, 1)
        linhas = _blocos_entre(seps, altura, minimo=14)
        if len(linhas) >= 3:
            break
    if not linhas:
        log(f"  [entv {indice}] nenhuma linha detectada")
        return []

    # a primeira faixa eh o cabecalho escuro
    if cinza[linhas[0][0]:linhas[0][1]].mean() < 120:
        linhas = linhas[1:]
    if not linhas:
        return []

    corpo = cinza[linhas[0][0]:, :]
    colunas, metodo = None, "proporcao"
    for limiar in (215, 200, 185, 170):
        seps = _faixas_claras(corpo.min(axis=0), limiar, 1)
        seps = [(a, b) for a, b in seps if a > largura * 0.04 and b < largura * 0.97]
        blocos = _blocos_entre(seps, largura, minimo=int(largura * 0.05))
        if len(blocos) == 4:
            colunas, metodo = blocos, f"minimo>={limiar}"
            break
    if colunas is None:
        colunas = faixas_por_proporcao(largura, CORTES_ENTV)

    log(f"  [entv {indice}] {largura}x{altura}, {len(linhas)} linha(s), colunas por {metodo}")

    eventos = []
    marcas = []
    for y0, y1 in linhas:
        cel = [img.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1)) for x0, x1 in colunas[:4]]
        hora = normalizar_hora(ler_celula(cel[0]))
        if not hora:
            continue
        competicao = limpar_competicao(ler_celula(cel[1]))
        marca = assinatura_icone(cel[1])
        if marca:
            marcas.append(f"{marca}  {competicao}")
        eventos.append(
            {
                "hora": hora,
                "competicao": competicao,
                "evento": limpar_confronto(ler_celula(cel[2])),
                "canais": limpar_canais(ler_celula(cel[3])),
                "icone": marca,
                "fonte": "esportesnatv",
            }
        )
    if indice == 0 and marcas:
        salvar_debug("icones.txt", "\n".join(marcas))
    log(f"  [entv {indice}] {data_iso}: {len(eventos)} evento(s), "
        f"{sum(1 for e in eventos if e['icone'])} com icone")
    return eventos


def coletar_esportesnatv():
    dias = {}
    for data_iso, urls in bluesky_posts_de_agenda():
        eventos = []
        for i, url in enumerate(urls):
            try:
                eventos.extend(ler_agenda_entv(url, data_iso, i))
            except Exception as e:
                log(f"  ERRO [entv {i}] {data_iso}: {e}")
        if eventos:
            dias[data_iso] = eventos
    return dias


# ----------------------------------------------------------------------------
# FONTE 2 — TOMADA DE TEMPO (TELEGRAM)
# ----------------------------------------------------------------------------


def telegram_imagens():
    """
    Le a versao web do canal e devolve [(datahora_post, url_imagem), ...]
    das mensagens recentes. O Telegram guarda a foto como background-image
    no HTML da previa, entao o endereco sai por expressao regular.
    """
    r = requests.get(TDT_CANAL, headers=UA, timeout=45)
    r.raise_for_status()
    sopa = BeautifulSoup(r.text, "html.parser")

    limite = datetime.now(timezone.utc) - timedelta(days=TDT_DIAS_DE_FEED)
    achados = []

    for msg in sopa.select("div.tgme_widget_message"):
        marca = msg.select_one("time[datetime]")
        try:
            quando = datetime.fromisoformat(marca["datetime"].replace("Z", "+00:00"))
        except Exception:
            continue
        if quando < limite:
            continue
        for foto in msg.select("a.tgme_widget_message_photo_wrap"):
            estilo = foto.get("style", "")
            m = re.search(r"background-image\s*:\s*url\(['\"]?([^'\")]+)", estilo)
            if m:
                achados.append((quando, m.group(1)))

    achados.sort(key=lambda x: x[0], reverse=True)
    achados = achados[:TDT_MAX_IMAGENS]
    log(f"Tomada de Tempo: {len(achados)} imagem(ns) nas mensagens recentes")
    return achados


def _data_do_cabecalho(img):
    """
    A tarja vermelha do topo traz 'SEXTA-FEIRA - 18/09/2026 PARTE 1'.
    Eh texto claro sobre fundo escuro, o inverso do que o OCR espera,
    entao tenta normal e invertido.
    """
    faixa = img.crop((0, 0, img.width, int(img.height * 0.18)))
    for inverter in (True, False):
        texto = ler_celula(faixa, psm=6, inverter=inverter)
        data_iso = data_de_texto(texto)
        if data_iso:
            return data_iso, texto
    return None, ""


def ler_agenda_tdt(url, indice=0):
    """
    Grade do Tomada de Tempo: 5 colunas, faixas alternando branco e cinza
    SEM separador branco entre elas. Por isso aqui a deteccao eh por tinta:
    onde ha pixel escuro ha texto, onde nao ha eh intervalo. Funciona
    independente da cor de fundo.
    """
    img = baixar_imagem(url)
    cinza = np.array(img.convert("L"), dtype=np.float32)
    altura, largura = cinza.shape

    data_iso, texto_cab = _data_do_cabecalho(img)
    if indice == 0:
        salvar_debug("tdt_cabecalho.txt", f"{url}\n\n{texto_cab}\n\ndata lida: {data_iso}")
    if not data_iso:
        log(f"  [tdt {indice}] data do cabecalho nao reconhecida, imagem ignorada")
        return None, []
    if data_iso < data_corte():
        return data_iso, []

    tinta = mascara_de_tinta(cinza)

    # linhas: intervalos verticais sem tinta nenhuma
    linhas = _agrupar(tinta.sum(axis=1), gap_min=3, tam_min=9)
    # descarta o cabecalho (logo e tarja ocupam o topo)
    linhas = [(a, b) for a, b in linhas if a > altura * 0.14]
    if not linhas:
        log(f"  [tdt {indice}] nenhuma linha de dados")
        return data_iso, []

    # Fica so com o primeiro bloco continuo de linhas. Entre a tabela e o
    # rodape ha uma area branca enorme; cortar ali tira o rodape, que
    # atravessa a largura toda e taparia os corredores entre as colunas.
    linhas = _primeiro_bloco(linhas)

    corpo = tinta[linhas[0][0]: linhas[-1][1], :]
    gap = max(4, int(largura * 0.008))
    blocos = _agrupar(corpo.sum(axis=0), gap_min=gap, tam_min=int(largura * 0.025))
    # a coluna do "B" (piloto brasileiro) eh estreita e quase sempre vazia
    blocos = [b for b in blocos if (b[1] - b[0]) >= largura * 0.035]

    # a primeira coluna tem so o horario: se vier larga, a deteccao colou
    # o horario na categoria e nao da para confiar
    primeira_ok = bool(blocos) and (blocos[0][1] - blocos[0][0]) <= largura * 0.12

    if len(blocos) == 5 and primeira_ok:
        colunas, metodo = blocos, "tinta"
    else:
        colunas = faixas_por_proporcao(largura, CORTES_TDT)
        metodo = f"proporcao (detectou {len(blocos)})"

    log(
        f"  [tdt {indice}] {largura}x{altura}, {data_iso}, "
        f"{len(linhas)} linha(s), colunas por {metodo}"
    )

    eventos = []
    dump = [f"{url}", f"data: {data_iso}", f"colunas: {colunas}", ""]

    for y0, y1 in linhas:
        folga = 2
        cel = [
            img.crop((max(0, x0 - folga), y0 - 1, min(largura, x1 + folga), y1 + 1))
            for x0, x1 in colunas[:5]
        ]
        brutos = [ler_celula(c) for c in cel]
        dump.append(f"y={y0}-{y1} | " + " || ".join(brutos))

        hora = normalizar_hora(brutos[0])
        if not hora:
            continue  # cabecalho da tabela e rodape caem aqui

        categoria = limpar(brutos[1])
        etapa = limpar(brutos[2])
        sessao = limpar(brutos[3])
        canais = limpar_canais(brutos[4])

        descricao = " - ".join([p for p in (etapa, sessao) if p])
        eventos.append(
            {
                "hora": hora,
                "competicao": categoria,
                "evento": descricao or categoria,
                "canais": canais,
                "etapa": etapa,
                "sessao": sessao,
                "fonte": "tomadadetempo",
            }
        )

    if indice == 0:
        salvar_debug(f"tdt_linhas_{indice}.txt", "\n".join(dump))

    log(f"  [tdt {indice}] {data_iso}: {len(eventos)} evento(s)")
    return data_iso, eventos


def coletar_tomada_de_tempo():
    dias = {}
    try:
        imagens = telegram_imagens()
    except Exception as e:
        log(f"Tomada de Tempo: canal do Telegram falhou ({e})")
        return {}

    for i, (_, url) in enumerate(imagens):
        try:
            data_iso, eventos = ler_agenda_tdt(url, i)
        except Exception as e:
            log(f"  ERRO [tdt {i}]: {e}")
            continue
        if data_iso and eventos:
            dias.setdefault(data_iso, []).extend(eventos)

    total = sum(len(v) for v in dias.values())
    log(f"Tomada de Tempo: {total} evento(s) em {len(dias)} dia(s)")
    return dias


# ----------------------------------------------------------------------------
# MONTAGEM
# ----------------------------------------------------------------------------


def deduplicar(eventos):
    """Mesma hora e descricao parecida = repetido (imagem PARTE 1 e PARTE 2)."""
    vistos, saida = [], []
    for ev in eventos:
        chave = (ev["hora"], normalizar(f"{ev['competicao']} {ev['evento']}")[:50])
        if any(k[0] == chave[0] and parecido(k[1], chave[1]) for k in vistos):
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
        juntar(coletar_esportesnatv())
    except Exception as e:
        log(f"Esportes na TV falhou: {e}")

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
        "fontes": ["Esportes na TV", "Tomada de Tempo"],
        "total": sum(len(d["eventos"]) for d in dias),
        "dias": dias,
    }


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    dados = montar()
    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    salvar_debug("ultima_execucao.txt", f"{dados['gerado_em']}\n{dados['total']} evento(s)")
    log(f"OK: {dados['total']} evento(s) em {len(dados['dias'])} dia(s) -> {SAIDA}")
    if dados["total"] == 0:
        log("AVISO: nenhum evento coletado")
        sys.exit(1)


if __name__ == "__main__":
    main()
