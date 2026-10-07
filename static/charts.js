/* tiny dependency-free bar-chart helper with hover tooltips (works offline) */
function drawBars(id, data, color, line){
  const cv=document.getElementById(id), dpr=window.devicePixelRatio||1, W=cv.parentElement.clientWidth-32, H=230;
  cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=W*dpr; cv.height=H*dpr;
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const L=38,R=8,T=10,B=30, n=data.labels.length, vals=data.values.map(v=>v===null?0:v);
  const mx=Math.max(1,...vals)*1.1, mn=Math.min(0,...vals), rng=mx-mn;
  const y=v=>T+(H-T-B)*(1-(v-mn)/rng), bw=(W-L-R)/n;
  g.font='11px system-ui'; g.fillStyle='#6b7580'; g.strokeStyle='#e6e9ec'; g.textAlign='right';
  for(let i=0;i<=4;i++){const v=mn+rng*i/4, yy=y(v); g.beginPath(); g.moveTo(L,yy); g.lineTo(W-R,yy); g.stroke(); g.fillText(v.toFixed(1),L-5,yy+4);}
  g.textAlign='center'; const step=Math.ceil(n/12);
  const pts=[];
  data.labels.forEach((lb,i)=>{const x=L+bw*i+bw/2, v=vals[i]; pts.push({x,y:y(v),lb,v:data.values[i]});
    if(i%step===0){g.fillStyle='#6b7580'; g.fillText(lb,x,H-10);} });
  if(line){ g.beginPath(); pts.forEach((p,i)=>i?g.lineTo(p.x,p.y):g.moveTo(p.x,p.y)); g.strokeStyle=color; g.lineWidth=2.5; g.stroke();
    g.lineTo(pts[n-1].x,y(0)); g.lineTo(pts[0].x,y(0)); g.fillStyle=color+'22'; g.fill();
    pts.forEach(p=>{g.beginPath(); g.arc(p.x,p.y,3,0,7); g.fillStyle=color; g.fill();}); }
  else { pts.forEach(p=>{ g.fillStyle=color; const h=y(0)-p.y; g.fillRect(p.x-bw*.36,Math.min(p.y,y(0)),bw*.72,Math.abs(h)); }); }
  const tip=document.getElementById('tip');
  cv.onmousemove=e=>{const r=cv.getBoundingClientRect(), x=e.clientX-r.left; const i=Math.max(0,Math.min(n-1,Math.floor((x-L)/bw)));
    if(x<L||x>W-R){tip.style.display='none';return;} const p=pts[i];
    tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+window.scrollY-34)+'px'; tip.textContent=p.lb+': '+(p.v===null?'no data':p.v+' min');};
  cv.onmouseleave=()=>{tip.style.display='none'};
}
