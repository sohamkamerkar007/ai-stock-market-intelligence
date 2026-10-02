import {api} from './api.js';
import {esc,date} from './analysis-components.js';
import {num,pct} from './charts.js';
import {setSafeHTML} from './security.js';

const node=id=>document.getElementById(id);
const put=(id,html)=>setSafeHTML(node(id),html);
const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'—';
const rupees=value=>Number.isFinite(value)?`₹${value.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2})}`:'—';
const model=name=>name==='xgboost'?'XGBoost':name==='logistic_regression'?'Logistic Regression':name||'Unavailable';
const metric=(label,value)=>`<div class="prediction-metric"><span>${esc(label)}</span><strong>${value}</strong></div>`;
const confidence=value=>value>=.75?'Strong model preference':value>=.6?'Moderate model preference':'Mixed signals';
const confidenceText=(value,direction)=>value>=.75?`The model strongly leans ${direction.toLowerCase()} based on the latest available signals.`:value>=.6?`The model leans ${direction.toLowerCase()}, though the signal is not decisive.`:'Current market signals are mixed, so the model has limited confidence.';
const volatility=value=>value<.01?'Lower expected movement':value<.02?'Moderate expected movement':'Elevated expected movement';
const reasonLine=r=>`${r.label} ${r.effect==='supports'?'supports':'tempers'} this outlook`;

export async function predictions(){
  const data=await api.finalOptions();
  const selected=new URLSearchParams(location.hash.split('?')[1]||'').get('symbol')||localStorage.getItem('selectedStock')||'RELIANCE';
  const stocks=data.stocks.map(a=>`<option value="${esc(a.symbol)}" ${a.symbol===selected?'selected':''}>${esc(a.symbol)} · ${esc(a.name)}</option>`).join('');
  return `<div class="page-head"><div><span class="eyebrow">Real market analysis</span><h2>AI Predictions</h2><p>A clear outlook built from the latest available market observations and saved models.</p></div></div>
    <section class="card prediction-controls" aria-label="Analyze a stock"><div class="prediction-field"><label for="predictAsset">Select stock</label><select id="predictAsset">${stocks}</select></div><div class="prediction-field"><label for="predictHorizon">Prediction horizon</label><select id="predictHorizon">${[1,3,5].map(h=>`<option value="${h}" ${h===data.recommended_horizon?'selected':''}>${h} trading session${h===1?'':'s'}</option>`).join('')}</select></div><button id="predictRun" class="primary-action">Analyze stock <span aria-hidden="true">→</span></button></section>
    <section id="predictResult" class="prediction-results" aria-live="polite"><div class="card empty"><div><b>Ready when you are</b><span>Choose a stock and select Analyze stock to see its latest outlook.</span></div></div></section>`;
}

function details(direction,real,research){
  const t=direction.test_metrics,m=real.metrics||{},selected=real.selected||{};
  const regress=Object.entries(m).map(([name,value])=>{const score=Object.values(value)[0]||{};return `<tr><td>${esc(name.replaceAll('_',' '))}</td><td>${num(score.mae,4)}</td><td>${num(score.rmse,4)}</td></tr>`}).join('');
  const researchModel=research?.selected?.model||research?.selected_model||'logistic_regression';
  const researchScore=research?.selected?.test||research?.selected_test||research?.models?.logistic_regression?.test;
  return `<details class="card prediction-details"><summary>View technical model details</summary><div class="technical-body"><div class="technical-grid">
    ${metric('Direction model',model(direction.model))}${metric('Horizon',`${direction.horizon} session${direction.horizon===1?'':'s'}`)}${metric('Probability of UP',percent(direction.probability_up))}${metric('Feature count',num(direction.feature_count,0))}
    ${metric('Training ended',esc(direction.train_end))}${metric('Validation ended',esc(direction.validation_end))}${metric('Held-out test',`${esc(direction.test_start)} to ${esc(direction.test_end)}`)}${metric('Data source','Real daily market history')}
  </div><h3>Real-market held-out validation</h3><div class="technical-grid">${metric('Accuracy',percent(t.accuracy))}${metric('Balanced accuracy',percent(t.balanced_accuracy))}${metric('Precision',percent(t.precision))}${metric('Recall',percent(t.recall))}${metric('F1 score',percent(t.f1))}${metric('ROC-AUC',num(t.roc_auc,3))}${metric('Training-majority baseline',percent(direction.baseline_test_accuracy))}</div>
  <p class="technical-note">Chronological training, validation, and final test periods. Overlapping labels are purged at split boundaries. Model choice uses validation balanced accuracy and F1. The final test period is evaluated once.</p>
  <h3>Price and volatility model checks</h3><p>Price: ${esc(selected.future_return?.model||'unavailable')} · Volatility: ${esc(selected.future_volatility?.model||'unavailable')}. Selection uses validation MAE.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Test MAE</th><th>Test RMSE</th></tr></thead><tbody>${regress}</tbody></table></div>
  <h3>SHAP feature contributions</h3><div class="table-scroll"><table><thead><tr><th>Market signal</th><th>Observed value</th><th>Effect on outlook</th></tr></thead><tbody>${direction.reasons.map(x=>`<tr><td>${esc(x.label)}</td><td>${num(x.feature_value,4)}</td><td>${esc(x.effect)}</td></tr>`).join('')}</tbody></table></div>
  ${researchScore?`<h3>Controlled Synthetic Research Evaluation</h3><p>This separate simulated dataset does not measure real-market forecasting accuracy.</p><div class="technical-grid">${metric('Accuracy',percent(researchScore.accuracy))}${metric('Balanced accuracy',percent(researchScore.balanced_accuracy))}${metric('Model',model(researchModel))}</div>`:''}
  </div></details>`;
}

function renderResult(r,research,warning){
  const a=r.real_stock,d=r.direction,p=a.selected?.future_return,v=a.selected?.future_volatility;
  if(!d)return `<div class="card empty"><div><b>Direction model unavailable</b><span>Train the saved real-market direction models to view an outlook.</span></div></div>`;
  const up=d.direction==='UP',reasons=d.reasons||[],support=reasons.filter(x=>x.effect==='supports'),risks=reasons.filter(x=>x.effect!=='supports');
  const change=p?.estimate/a.current_close-1;
  const supporting=support.slice(0,2).map(x=>x.label.toLowerCase()).join(' and ');
  const summary=`${r.symbol} currently leans ${d.direction.toLowerCase()} over the next ${d.horizon} trading session${d.horizon===1?'':'s'}. ${supporting?`${supporting[0].toUpperCase()+supporting.slice(1)} ${support.length>1?'support':'supports'} the outlook.`:'The available signals offer limited support.'} ${risks.length?`${risks[0].label} tempers it.`:''}`;
  return `${warning?`<div class="notice warning">${esc(warning)}</div>`:''}<article class="card outlook-hero ${up?'up':'down'}"><div><span class="eyebrow">AI market outlook · ${esc(r.symbol)}</span><p class="outlook-kicker">Expected direction</p><div class="outlook-direction"><span aria-hidden="true">${up?'↑':'↓'}</span> ${esc(d.direction)}</div><p class="outlook-meta">${model(d.model)} · ${d.horizon}-session outlook</p><p class="outlook-freshness">Latest available daily close: ${date(a.as_of)} · Analysis checked ${new Date().toLocaleString('en-IN')}</p></div><div class="confidence-panel"><span>AI confidence</span><strong>${percent(d.confidence)}</strong><b>${confidence(d.confidence)}</b><p>${esc(confidenceText(d.confidence,d.direction))}</p></div></article>
  <div class="prediction-cards"><article class="card"><span class="eyebrow">Estimated price</span><div class="price-comparison"><div><small>Current price</small><strong>${rupees(a.current_close)}</strong></div><span aria-hidden="true">→</span><div><small>Estimated price</small><strong>${p?rupees(p.estimate):'Unavailable'}</strong></div></div><div class="prediction-foot"><span>Estimated change</span><strong class="${change>=0?'positive':'negative'}">${p?pct(change):'—'}</strong><span>Over ${d.horizon} trading session${d.horizon===1?'':'s'}</span></div></article>
  <article class="card"><span class="eyebrow">Expected volatility</span><div class="volatility-value">${v?percent(v.estimate):'—'}</div><p>${v?volatility(v.estimate):'Estimate unavailable'}</p><small>Estimated daily movement over the selected horizon</small></article></div>
  <article class="card reasons-card"><span class="eyebrow">Why the AI thinks this</span><h3>Signals shaping this outlook</h3>${reasons.length?`<div class="reason-grid">${reasons.map(x=>`<div class="reason ${x.effect==='supports'?'support':'risk'}"><span aria-hidden="true">${x.effect==='supports'?'✓':'!'}</span><p>${esc(reasonLine(x))}.</p></div>`).join('')}</div>`:'<p>Feature explanation is unavailable for this prediction.</p>'}</article>
  <article class="card ai-summary"><span class="eyebrow">AI summary</span><p>${esc(summary)}</p><small>Experimental real-market direction model. Held-out accuracy ${percent(d.test_metrics.accuracy)}; predictive advantage has not been demonstrated.</small></article>
  ${details(d,a,research)}`;
}

export async function hydratePredictions(){
  const run=async()=>{const button=node('predictRun');button.disabled=true;button.textContent='Analyzing…';put('predictResult','<div class="card empty"><div><b>Analyzing latest observations</b><span>Loading saved models and market signals.</span></div></div>');
    try{const symbol=node('predictAsset').value,horizon=+node('predictHorizon').value;localStorage.setItem('selectedStock',symbol);let warning='';try{const refresh=await api.refresh(symbol);if(Object.keys(refresh.failures||{}).length)warning='Provider refresh was unavailable. Showing the latest stored observations.'}catch{warning='Provider refresh was unavailable. Showing the latest stored observations.'}const [data,options]=await Promise.all([api.finalPrediction({symbol,horizon}),api.finalOptions()]);put('predictResult',renderResult(data,options.research,warning))}catch(e){put('predictResult',`<div class="card empty"><div><b>Analysis unavailable</b><span>${esc(e.message||String(e))}</span></div></div>`)}finally{button.disabled=false;button.textContent='Analyze stock →'}};
  node('predictRun').addEventListener('click',run);
}
