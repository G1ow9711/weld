"""Project-local adapter for the observed September 27 ChatGPT picker DOM.

Does not change the installed skill or relax response-model/sentinel checks.
Only accepts the live model menu linked to the visible composer model button.
"""
import copy
import json
import sys
import time
from pathlib import Path

SKILL = Path(r'C:\Users\Administrator\.codex\skills\gpt6-pro-consult\scripts')
sys.path.insert(0, str(SKILL))
import run_gpt6_pro_consult as runner
from model_picker import picker_kind as original_kind


def project_state_root():
    """Use the supported state-root override to keep new runtime writes local.

    Read the installed skill's default first; never create a second mapping if
    a conversation is already bound there. This project adapter always reuses
    its own thread-keyed state on subsequent calls.
    """
    default=original_state_root()
    if runner.read_state(runner.resolve_thread_id(),default).get('found'):
        return default
    return Path.cwd()/'.codex-local/gpt6-pro-state'

SNAPSHOT = r'''(() => {
 const visible = e => e && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden';
 const norm = s => (s || '').replace(/\s+/g,' ').trim();
 const buttons = [...document.querySelectorAll('button[aria-label="选择 ChatGPT 模型"]')].filter(visible);
 const menus = [...document.querySelectorAll('[role="menu"]')].filter(visible);
 return {composer:buttons.map(e=>({text:norm(e.innerText),id:e.id,expanded:e.getAttribute('aria-expanded')})),
 menus:menus.map(e=>({
  text:norm(e.innerText), picker:!!e.querySelector('[data-testid="composer-intelligence-picker-content"]'),
  linkedModelMenu:e.className.includes('ModelPickerDropdownContent-') && buttons.some(b=>b.id===e.getAttribute('aria-labelledby') && b.getAttribute('aria-expanded')==='true'),
  expanders:[...e.querySelectorAll('[aria-label]')].filter(visible).map(x=>x.getAttribute('aria-label')),
  family:norm(e.querySelector('[aria-label="选择模型"],[aria-label="Choose model"],[aria-label="Select model"]')?.innerText),
  radios:[...e.querySelectorAll('[role="menuitemradio"],[role="radio"]')].filter(visible).map(r=>({text:norm(r.innerText),checked:r.getAttribute('aria-checked'),testid:r.getAttribute('data-testid')})),
  sliders:[...e.querySelectorAll('[role="slider"]')].filter(visible).map(s=>({min:s.getAttribute('aria-valuemin'),max:s.getAttribute('aria-valuemax'),value:s.getAttribute('aria-valuenow'),text:s.getAttribute('aria-valuetext')}))
 }))};
})()'''


def observed_kind(snapshot):
    checked=copy.deepcopy(snapshot)
    for m in checked.get('menus',[]):
        if m.get('linkedModelMenu'):m['picker']=True
    return original_kind(checked)


def validate_adapter():
    positive={'menus':[{'picker':False,'linkedModelMenu':True,'family':'6 Pro',
        'text':'6 Pro Pro，第 5 项，共 5 项。','radios':[{'text':'最新','checked':'true'},{'text':'GPT-5.6 Sol','checked':'false'}],
        'sliders':[{'min':'0','max':'4','value':'4'}]}]}
    assert observed_kind(positive)=='gpt6-slider-v202609'
    for field,value in [('linkedModelMenu',False),('family','5.6 Pro'),('text','6 Pro High')]:
        negative=copy.deepcopy(positive);negative['menus'][0][field]=value
        assert observed_kind(negative) is None
    for value in ['0','1','2','3']:
        negative=copy.deepcopy(positive);negative['menus'][0]['sliders'][0]['value']=value
        assert observed_kind(negative) is None
    negative=copy.deepcopy(positive);negative['menus'][0]['radios'][1]['checked']='true'
    assert observed_kind(negative) is None
    negative=copy.deepcopy(positive);negative['menus'][0]['radios'][0]['checked']='false'
    assert observed_kind(negative) is None


def composer_ref(state):
    return runner.find_ref(state,lambda line:'contenteditable=true' in line and 'role=textbox' in line and 'aria-label=询问 ChatGPT' in line)


def ensure_current(session):
    # Navigation hydration is asynchronous. No Send while selection is unknown.
    for _ in range(8):
        snap=runner.get_picker_snapshot(session)
        if snap.get('composer'):break
        time.sleep(1)
    result=original_ensure(session)
    log=Path('weld_pipeline/reports/consult_picker_evidence.jsonl')
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps({'verified_closed':runner.get_picker_snapshot(session),'time':time.time()},ensure_ascii=False)+'\n')
    return result


def wait_current(session,sentinel,timeout_seconds,poll_seconds):
    deadline=time.time()+timeout_seconds
    while time.time()<deadline:
        time.sleep(poll_seconds)
        raw=runner.run_opencli(['browser',session,'eval',r'''(() => {
         const a=[...document.querySelectorAll('[data-message-author-role="assistant"]')].at(-1);
         const visible=e=>e.getClientRects().length>0;
         const stops=[...document.querySelectorAll('button')].filter(visible).filter(e=>/^(停止|停止回答|停止生成|Stop generating|Stop streaming)$/.test(e.getAttribute('aria-label')||'') || e.getAttribute('data-testid')==='stop-button');
         return {url:location.href,slug:a?.getAttribute('data-message-model-slug'),text:a?.innerText,generating:stops.length>0};
        })()'''])
        v=json.loads(raw)
        Path('weld_pipeline/reports/consult_live_status.json').write_text(json.dumps({k:v.get(k) for k in ['url','slug','generating']},indent=2),encoding='utf-8')
        if v.get('slug') and v['slug']!='gpt-6-pro':
            raise runner.ConsultError('Response execution model mismatch: '+str(v['slug']))
        reply=(v.get('text') or '').replace('\\_','_')
        if reply.startswith(sentinel) and v.get('slug')=='gpt-6-pro' and not v.get('generating'):
            ensure_current(session)
            return reply
    raise runner.ConsultError('Pending: completed assistant-side sentinel and exact gpt-6-pro execution slug not jointly verified. Do not resend.')


validate_adapter()
original_state_root=runner.state_root
runner.state_root=project_state_root
runner.PICKER_SNAPSHOT_JS=SNAPSHOT
runner.picker_kind=observed_kind
runner.composer_ref=composer_ref
runner.SEND_HINTS=(*runner.SEND_HINTS,'aria-label=发送 ')
original_ensure=runner.ensure_gpt6_pro
runner.ensure_gpt6_pro=ensure_current
runner.wait_for_reply=wait_current

if __name__=='__main__':
    if '--check-picker' in sys.argv:
        snap=runner.get_picker_snapshot('gpt6-pro-1942f22055111baa')
        print(json.dumps(snap,ensure_ascii=False,indent=2));print('verified_kind:',observed_kind(snap))
    else:raise SystemExit(runner.main())
