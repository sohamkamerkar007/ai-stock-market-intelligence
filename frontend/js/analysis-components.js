import {num,prob,pct} from './charts.js';
export {num,prob,pct};
export const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const date=value=>value?new Date(value).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric'}):'Unavailable';
export const panel=(title,description,body)=>`<section class="card mt"><header><div><h3>${title}</h3><p>${description}</p></div></header>${body}</section>`;
export const empty=message=>`<div class="empty"><div><b>Analysis unavailable</b><span>${esc(message)}</span></div></div>`;
export const failure=error=>empty('This analysis could not be loaded. Please try again.')+`<details class="details"><summary>Technical details</summary><pre>${esc(error.message)}</pre></details>`;
export const table=(headers,rows)=>`<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
export const metricKeys=['accuracy','balanced_accuracy','precision','recall','f1','roc_auc','mcc'];
export const labels={accuracy:'Accuracy',balanced_accuracy:'Class-balanced accuracy',precision:'Correctness of UP calls',recall:'UP moves detected',f1:'Precision / recall balance',roc_auc:'Up / down separation',mcc:'Matthews correlation'};
export const help={accuracy:'Fraction of correct direction classifications.',balanced_accuracy:'Average recall of each class; a constant-class predictor scores 50%.',precision:'Of all predicted UP moves, the share that occurred.',recall:'Of all actual UP moves, the share the model identified.',f1:'Harmonic mean of precision and recall.',roc_auc:'Ranking ability across all probability thresholds. 0.5 indicates chance separation.',mcc:'Correlation between predicted and actual classes; 0 indicates no correlation.'};
export const metricHeaders=()=>metricKeys.map(k=>`<abbr tabindex="0" title="${help[k]}">${labels[k]}</abbr>`);
export const metricCells=m=>metricKeys.map(k=>m[k]==null?'Unavailable':k==='mcc'?num(m[k],3):prob(m[k]));
export const selectedAsset=()=>new URLSearchParams(location.hash.split('?')[1]||'').get('symbol');
export const modelName=name=>({extra_trees:'ExtraTrees',hist_gradient_boosting:'Histogram Gradient Boosting',logistic_regression:'Logistic Regression',random_forest:'Random Forest',xgboost:'XGBoost'}[name]||esc(name));
export async function get(path,options){const response=await fetch('/api/v1'+path,{headers:{'Content-Type':'application/json'},...options});if(!response.ok){let body;try{body=await response.json()}catch{}throw Error(typeof body?.detail==='string'?body.detail:`Request returned ${response.status}`)}return response.json()}
