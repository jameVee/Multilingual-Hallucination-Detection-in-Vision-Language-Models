#!/usr/bin/env python3
"""SHROOM-visions Sample Browser — opens in your browser automatically.
Run:  python sample_browser_app.py
"""

import json, sys, subprocess
from pathlib import Path

# ── Auto-install dependencies ─────────────────────────────────────────────────
for pkg, install_name in [('gradio', 'gradio'), ('PIL', 'Pillow')]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {install_name}…")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_name])

import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / 'data'
IMAGE_DIR = _HERE / 'shroom-vis-images'

LABEL_COLORS = {
    'ocr':              ('#3b82f6', '#dbeafe'),
    'mischaracterization': ('#f97316', '#ffedd5'),
    'invention':        ('#ef4444', '#fee2e2'),
    'miscounting':      ('#8b5cf6', '#ede9fe'),
    'other':            ('#16a34a', '#dcfce7'),
}
_UNK_COLOR = ('#6b7280', '#f3f4f6')

# ── Load data ─────────────────────────────────────────────────────────────────
ALL_SAMPLES: list[dict] = []
for _split in ['train', 'val', 'test']:
    _fp = DATA_DIR / f'{_split}.jsonl'
    if _fp.exists():
        with open(_fp, encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    s = json.loads(_line)
                    s['_file_split'] = _split   # tag with the actual file source
                    for lb in (s.get('labels') or []):
                        lb['label'] = lb['label'].lower()  # normalize: 'OCR' → 'ocr'
                    ALL_SAMPLES.append(s)

ID_INDEX = {s['id']: i for i, s in enumerate(ALL_SAMPLES) if 'id' in s}
print(f"Loaded {len(ALL_SAMPLES):,} samples  "
      f"({sum(1 for s in ALL_SAMPLES if s.get('labels'))} with labels)")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _esc(t: str) -> str:
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def highlight_response(response: str, labels: list) -> str:
    if not labels:
        return (
            '<div style="white-space:pre-wrap;font-family:sans-serif;'
            'line-height:1.7;font-size:14px;color:#111827;">'
            + _esc(response) + '</div>'
        )
    spans = sorted(
        [lb for lb in labels if lb['end'] > lb['start']],
        key=lambda x: (x['start'], -x['end']),
    )
    parts, pos = [], 0
    for sp in spans:
        s, e = sp['start'], sp['end']
        lbl  = sp.get('label', 'other')
        prob = sp.get('prob', 0.5)
        bc, bg = LABEL_COLORS.get(lbl, _UNK_COLOR)
        if s > pos:
            parts.append(_esc(response[pos:s]))
        if s < e <= len(response):
            parts.append(
                f'<mark style="background:{bg};border-bottom:2px solid {bc};color:#111827;'
                f'border-radius:3px;padding:0 2px;cursor:default;" '
                f'title="{lbl}  prob={prob:.3f}">{_esc(response[s:e])}</mark>'
            )
        pos = max(pos, e)
    parts.append(_esc(response[pos:]))
    return (
        '<div style="white-space:pre-wrap;font-family:sans-serif;'
        'line-height:1.7;font-size:14px;color:#111827;">' + ''.join(parts) + '</div>'
    )


def get_filtered(split: str, lang: str, cat: str, label_filter: str) -> list:
    result = []
    for s in ALL_SAMPLES:
        if split != 'all' and s.get('_file_split') != split:
            continue
        if lang != 'all' and s.get('language') != lang:
            continue
        labels = s.get('labels') or []
        if label_filter == 'Has labels' and not labels:
            continue
        if label_filter == 'Clean only' and labels:
            continue
        if cat != 'all' and cat not in {lb['label'] for lb in labels}:
            continue
        result.append(s)
    return result


def render_sample(sample: dict) -> tuple:
    labels   = sample.get('labels') or []
    response = sample.get('response', '')
    img_name = sample.get('image_name', '')

    # Image
    img_path = None
    if img_name:
        p = IMAGE_DIR / img_name
        if p.exists():
            img_path = str(p)

    # Meta
    label_set = {lb['label'] for lb in labels}
    badges = ''.join(
        f'<span style="background:{LABEL_COLORS.get(lb, _UNK_COLOR)[1]};'
        f'border:1px solid {LABEL_COLORS.get(lb, _UNK_COLOR)[0]};'
        f'color:{LABEL_COLORS.get(lb, _UNK_COLOR)[0]};'
        f'border-radius:4px;padding:2px 8px;font-size:12px;margin:0 3px;">{lb}</span>'
        for lb in sorted(label_set)
    )
    meta_html = (
        f'<div style="padding:6px 0;font-family:sans-serif;color:#111827;">'
        f'<b style="font-family:monospace;font-size:15px;">{_esc(sample.get("id","?"))}</b> '
        f'<span style="background:#e5e7eb;color:#374151;border-radius:4px;padding:2px 8px;'
        f'font-size:12px;margin-right:4px;">{sample.get("_file_split","?")}</span>'
        f'<span style="background:#e5e7eb;color:#374151;border-radius:4px;padding:2px 8px;'
        f'font-size:12px;">{sample.get("language","?")}</span>'
        f' {badges}<br>'
        f'<small style="color:#6b7280;font-family:monospace;">{_esc(img_name)}</small>'
        f'</div>'
    )

    # Question
    q_html = (
        '<div style="font-size:12px;font-weight:600;color:#374151;'
        'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">Question</div>'
        f'<div style="background:#f3f4f6;color:#111827;border-left:3px solid #6b7280;'
        f'padding:8px 12px;border-radius:0 6px 6px 0;font-family:sans-serif;'
        f'font-size:14px;line-height:1.6;">{_esc(sample.get("prompt",""))}</div>'
    )

    # Response (with legend)
    if labels:
        legend_items = ''.join(
            f'<span style="background:{bg};border-bottom:2px solid {bc};color:{bc};'
            f'border-radius:3px;padding:1px 6px;margin-right:5px;font-size:12px;font-family:sans-serif;">{lbl}</span>'
            for lbl, (bc, bg) in LABEL_COLORS.items()
        )
        legend = f'<div style="margin-bottom:6px;">{legend_items}</div>'
    else:
        legend = ''

    resp_html = (
        '<div style="font-size:12px;font-weight:600;color:#374151;'
        'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">Response</div>'
        + legend
        + '<div style="background:#ffffff;color:#111827;border:1px solid #d1d5db;'
        'border-radius:6px;padding:10px 14px;max-height:460px;overflow-y:auto;">'
        + highlight_response(response, labels) + '</div>'
    )

    # Spans table
    if labels:
        rows = []
        for sp in sorted(labels, key=lambda x: x['start']):
            s, e   = sp['start'], sp['end']
            lbl    = sp.get('label', '?')
            prob   = sp.get('prob', 0)
            text   = response[s:e] if e <= len(response) else '(out of range)'
            bc, bg = LABEL_COLORS.get(lbl, _UNK_COLOR)
            bar_w  = int(prob * 60)
            rows.append(
                f'<tr>'
                f'<td style="padding:3px 8px;font-family:monospace;font-size:12px;color:#111827;">[{s}:{e}]</td>'
                f'<td style="padding:3px 8px;">'
                f'<span style="background:{bg};color:{bc};border-radius:3px;padding:1px 5px;font-size:12px;">'
                f'{lbl}</span></td>'
                f'<td style="padding:3px 8px;font-size:12px;white-space:nowrap;color:#111827;">'
                f'<div style="display:inline-block;width:{bar_w}px;height:8px;background:{bc};'
                f'border-radius:4px;vertical-align:middle;"></div> {prob:.3f}</td>'
                f'<td style="padding:3px 8px;font-family:monospace;font-size:12px;color:#111827;'
                f'max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                f'{_esc(text)}</td>'
                f'</tr>'
            )
        th = ''.join(
            f'<th style="padding:3px 8px;text-align:left;font-size:12px;color:#374151;">{h}</th>'
            for h in ['Span', 'Label', 'Prob', 'Text']
        )
        span_html = (
            '<div style="font-size:12px;font-weight:600;color:#374151;'
            'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">'
            f'Spans ({len(labels)})</div>'
            f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:6px;padding:4px;">'
            f'<table style="border-collapse:collapse;width:100%;">'
            f'<thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )
    else:
        span_html = '<p style="color:#6b7280;font-size:13px;font-family:sans-serif;">No labels — clean sample</p>'

    return img_path, meta_html, q_html, resp_html, span_html


# ── Gradio event functions ────────────────────────────────────────────────────
def main_render(split, lang, cat, has_labels, idx):
    filtered = get_filtered(split, lang, cat, has_labels)
    n   = len(filtered)
    idx = max(0, min(int(idx or 0), n - 1)) if n else 0
    counter     = f"{idx + 1} / {n:,}" if n else "0 / 0"
    slider_upd  = gr.update(maximum=max(0, n - 1), value=idx)
    if not n:
        no_match = '<p style="color:#6b7280;font-family:sans-serif;">No matching samples.</p>'
        return None, counter, slider_upd, '', no_match, '', ''
    img, meta, q, r, sp = render_sample(filtered[idx])
    return img, counter, slider_upd, meta, q, r, sp


def on_filter(split, lang, cat, label_filter):
    return main_render(split, lang, cat, label_filter, 0)

def on_slide(split, lang, cat, label_filter, idx):
    return main_render(split, lang, cat, label_filter, idx)

def on_prev(split, lang, cat, label_filter, idx):
    return main_render(split, lang, cat, label_filter, max(0, int(idx) - 1))

def on_next(split, lang, cat, label_filter, idx):
    filtered = get_filtered(split, lang, cat, label_filter)
    return main_render(split, lang, cat, label_filter,
                       min(len(filtered) - 1, int(idx) + 1))

def on_search(split, lang, cat, label_filter, idx, target):
    target = (target or '').strip()
    if not target:
        return main_render(split, lang, cat, label_filter, idx)
    filtered = get_filtered(split, lang, cat, label_filter)
    for i, s in enumerate(filtered):
        if s.get('id') == target:
            return main_render(split, lang, cat, label_filter, i)
    global_idx = ID_INDEX.get(target)
    if global_idx is not None:
        return main_render('all', 'all', 'all', 'All', global_idx)
    img, counter, slider_upd, meta, q, r, sp = main_render(split, lang, cat, label_filter, idx)
    r = f'<p style="color:red;font-family:sans-serif;">ID not found: <b>{_esc(target)}</b></p>' + r
    return img, counter, slider_upd, meta, q, r, sp


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title='SHROOM-visions Browser') as demo:
    gr.Markdown('## SHROOM-visions Sample Browser')

    with gr.Row():
        split_dd  = gr.Dropdown(['all', 'train', 'val', 'test'],
                                value='all', label='Split', scale=1)
        lang_dd   = gr.Dropdown(['all', 'en', 'fr', 'it', 'zh'],
                                value='all', label='Language', scale=1)
        cat_dd    = gr.Dropdown(
                        ['all', 'ocr', 'mischaracterization',
                         'invention', 'miscounting', 'other'],
                        value='all', label='Category', scale=2)
        labels_dd = gr.Dropdown(['All', 'Has labels', 'Clean only'],
                                value='All', label='Samples', scale=1)

    with gr.Row():
        prev_btn   = gr.Button('◀ Prev', min_width=80)
        counter_tb = gr.Textbox(value='— / —', show_label=False, interactive=False,
                                min_width=110, container=False)
        idx_sl     = gr.Slider(0, max(0, len(ALL_SAMPLES) - 1), step=1,
                               value=0, label='Index', scale=4)
        next_btn   = gr.Button('Next ▶', min_width=80)
        search_tb  = gr.Textbox(placeholder='Jump to ID (e.g. train-en-9757)',
                                show_label=False, scale=3, container=False)
        search_btn = gr.Button('Go', min_width=50)

    with gr.Row():
        with gr.Column(scale=2):
            img_out  = gr.Image(label='Image', type='filepath', height=380)
            meta_out = gr.HTML()
        with gr.Column(scale=3):
            q_out    = gr.HTML()
            resp_out = gr.HTML()
            span_out = gr.HTML()

    OUTPUTS    = [img_out, counter_tb, idx_sl, meta_out, q_out, resp_out, span_out]
    FILTER_INP = [split_dd, lang_dd, cat_dd, labels_dd]
    NAV_INP    = FILTER_INP + [idx_sl]

    for w in FILTER_INP:
        w.change(on_filter, inputs=FILTER_INP, outputs=OUTPUTS)

    idx_sl.release(on_slide,   inputs=NAV_INP,             outputs=OUTPUTS)
    prev_btn.click (on_prev,   inputs=NAV_INP,             outputs=OUTPUTS)
    next_btn.click (on_next,   inputs=NAV_INP,             outputs=OUTPUTS)
    search_btn.click(on_search, inputs=NAV_INP + [search_tb], outputs=OUTPUTS)
    search_tb.submit(on_search, inputs=NAV_INP + [search_tb], outputs=OUTPUTS)

    demo.load(on_filter, inputs=FILTER_INP, outputs=OUTPUTS)


if __name__ == '__main__':
    demo.launch()
