const pageSize=25;let paused=false;let requestOffset=0;let eventOffset=0;let requestFilter={};let eventFilter={};
const el=id=>document.getElementById(id);
const cell=(row,value,cls)=>{const td=document.createElement('td');if(cls)td.className=cls;td.textContent=value??'';row.append(td);return td};
const badge=(text,tone)=>{const span=document.createElement('span');span.className=`badge badge-${tone}`;span.textContent=text;return span};
const I18N={
  en:{state:'State',model:'Model',requests:'Requests',success:'Success',latAvg:'Latency avg',latP95:'Latency p95',started:'Started',ready:'Ready',notReady:'Not ready',ok:'ok',allTime:'all time',last100:'last 100',prev:'← Prev',next:'Next →',of:'of',pause:'Pause',resume:'Resume',updated:'updated',dim:'dim',connecting:'Connecting',unavailable:'Unavailable'},
  zh:{state:'状态',model:'模型',requests:'请求数',success:'成功率',latAvg:'平均延迟',latP95:'P95 延迟',started:'启动时间',ready:'就绪',notReady:'未就绪',ok:'成功',allTime:'全程',last100:'最近 100 次',prev:'← 上一页',next:'下一页 →',of:'共',pause:'暂停',resume:'恢复',updated:'更新于',dim:'维度',connecting:'连接中',unavailable:'不可用'}
};
let lang=localStorage.getItem('lang')||(navigator.language&&navigator.language.startsWith('zh')?'zh':'en');
const t=key=>I18N[lang][key]||I18N.en[key];
const fmtMs=ms=>ms==null?'—':ms>=1000?`${(ms/1000).toFixed(2)} s`:`${Math.round(ms)} ms`;
const fmtTime=iso=>{if(!iso)return'';const d=new Date(iso);return isNaN(d)?iso:d.toLocaleString([],{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'})};
async function get(url){const response=await fetch(url);if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json()}
function applyI18n(){
  document.documentElement.lang=lang==='zh'?'zh-CN':'en';
  for(const node of document.querySelectorAll('[data-i18n]'))node.textContent=t(node.dataset.i18n)||node.textContent;
  for(const node of document.querySelectorAll('[data-i18n-ph]'))node.placeholder=t(node.dataset.i18nPh)||node.placeholder;
  const pause=el('pause');
  if(!paused)pause.textContent=t('pause');
  el('lang').value=lang;
}
function pager(container,offset,total,move){container.replaceChildren();const prev=document.createElement('button');prev.textContent=t('prev');prev.disabled=offset===0;const next=document.createElement('button');next.textContent=t('next');next.disabled=offset+pageSize>=total;prev.onclick=()=>{move(Math.max(0,offset-pageSize));refresh()};next.onclick=()=>{move(offset+pageSize);refresh()};const label=document.createElement('span');label.textContent=`${total?offset+1:0}–${Math.min(offset+pageSize,total)} ${t('of')} ${total}`;container.append(prev,next,label)}
function renderStats(status){
  const box=el('stats');box.replaceChildren();
  const items=[
    {label:t('state'),value:status.ready?t('ready'):t('notReady'),sub:status.device,tone:status.ready?'ok':'err'},
    {label:t('model'),value:status.model,sub:`${t('dim')} ${status.dimension??'—'}`},
    {label:t('requests'),value:status.requests??'—',sub:`${status.successes??0} ${t('ok')}`},
    {label:t('success'),value:status.success_rate==null?'—':`${(status.success_rate*100).toFixed(1)}%`,tone:status.success_rate!=null&&status.success_rate<1?'warn':'ok'},
    {label:t('latAvg'),value:fmtMs(status.average_ms),sub:t('allTime')},
    {label:t('latP95'),value:fmtMs(status.p95_ms),sub:t('last100')},
    {label:t('started'),value:fmtTime(status.started_at),sub:''},
  ];
  for(const item of items){
    const card=document.createElement('div');card.className='stat';
    const label=document.createElement('div');label.className='label';label.textContent=item.label;
    const value=document.createElement('div');value.className='value';
    if(item.tone){const toneClass=item.tone==='ok'?'badge-ok':item.tone==='warn'?'badge-warn':'badge-err';const b=badge(String(item.value),toneClass);value.append(b)}else value.textContent=item.value;
    card.append(label,value);
    if(item.sub){const sub=document.createElement('div');sub.className='sub';sub.textContent=item.sub;card.append(sub)}
    box.append(card);
  }
}
async function refresh(){
  if(paused)return;
  try{
    const [status,requests,events]=await Promise.all([
      get('/api/status'),
      get(`/api/requests?limit=${pageSize}&offset=${requestOffset}&${new URLSearchParams(requestFilter)}`),
      get(`/api/events?limit=${pageSize}&offset=${eventOffset}&${new URLSearchParams(eventFilter)}`),
    ]);
    const state=el('state');
    state.textContent=status.ready?t('ready'):t('notReady');
    state.className=`badge ${status.ready?'badge-ok':'badge-err'}`;
    el('updated').textContent=`${t('updated')} ${new Date().toLocaleTimeString()}`;
    renderStats(status);
    const requestBody=el('requests');requestBody.replaceChildren();
    for(const item of requests.data){
      const row=document.createElement('tr');
      cell(row,fmtTime(item.timestamp));
      cell(row,item.request_id);
      cell(row,item.input_count);
      cell(row,item.token_count);
      cell(row,fmtMs(item.duration_ms));
      const st=cell(row,'');st.append(badge(item.status==='ok'?t('ok'):item.status,item.status==='ok'?'ok':'err'));
      cell(row,item.error);
      requestBody.append(row);
    }
    pager(el('request-pages'),requestOffset,requests.total,value=>requestOffset=value);
    const eventBody=el('events');eventBody.replaceChildren();
    for(const item of events.data){
      const row=document.createElement('tr');
      cell(row,fmtTime(item.timestamp));
      const lv=cell(row,'');lv.append(badge(item.level,item.level==='ERROR'?'err':item.level==='WARNING'?'warn':'muted'));
      cell(row,item.event);
      cell(row,item.request_id);
      cell(row,item.detail);
      eventBody.append(row);
    }
    pager(el('event-pages'),eventOffset,events.total,value=>eventOffset=value);
  }catch(error){
    const state=el('state');state.textContent=`${t('unavailable')}: ${error.message}`;state.className='badge badge-err';
  }
}
el('lang').onchange=event=>{lang=event.target.value;localStorage.setItem('lang',lang);applyI18n();refresh()};
el('pause').onclick=event=>{paused=!paused;event.currentTarget.setAttribute('aria-pressed',String(paused));event.currentTarget.textContent=paused?t('resume'):t('pause');if(!paused)refresh()};
el('request-filter').onsubmit=event=>{event.preventDefault();requestFilter=Object.fromEntries(new FormData(event.currentTarget));requestOffset=0;refresh()};
el('event-filter').onsubmit=event=>{event.preventDefault();eventFilter=Object.fromEntries(new FormData(event.currentTarget));eventOffset=0;refresh()};
applyI18n();
refresh();setInterval(refresh,5000);
