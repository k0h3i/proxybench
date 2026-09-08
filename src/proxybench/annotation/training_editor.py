"""Readable form controls for structured training labels."""

SCRIPT = r'''
function trainingName(key){return key.replaceAll('_',' ').replace(/^./,s=>s.toUpperCase());}
function trainingDisplay(text,property){
 if(property!=='value'&&property!=='raw_text')return text||'(blank)';
 if(text==='')return '(unreviewed)';
 function describe(value){
  if(value===null)return 'No value';
  if(typeof value==='string')return value||'(empty text)';
  if(Array.isArray(value))return value.length?value.map(describe).join('\n'):'No components';
  if(Object.hasOwn(value,'availability'))return value.availability==='PRESENT'?describe(value.value):trainingName(value.availability.toLowerCase());
  return Object.entries(value).map(([key,item])=>trainingName(key)+': '+describe(item)).join('\n');
 }
 try{return describe(JSON.parse(text));}catch{return 'Invalid saved value';}
}
function trainingControl(field,property){
 const host=document.createElement('div');host.className='training-value';
 const schema=property==='raw_text'?{type:'text'}:trainingSchema[field];
 let value=null,untouched=true;
 const notify=()=>{untouched=false;host.dispatchEvent(new Event('input',{bubbles:true}));};
 const make=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;};
 const emptyField=()=>({value:null,raw_text:null,availability:'ABSENT_IN_CONTEXT',origin:null});
 function initial(spec){
  if(spec.type==='object')return Object.fromEntries(Object.keys(spec.fields).map(key=>[key,emptyField()]));
  if(spec.type==='list')return [];
  return spec.choices?spec.choices[0]:'';
 }
 function select(parent,title,choices,current,change){
  const label=make('label',title),control=make('select');
  for(const choice of choices){const option=make('option',choice?trainingName(choice.toLowerCase()):'No origin');option.value=choice;control.append(option);}
  control.value=current||'';control.oninput=event=>{event.stopPropagation();change(control.value);notify();};label.append(control);parent.append(label);
 }
 function nested(parent,title,spec,item){
  const group=make('fieldset');group.className='training-group';group.append(make('legend',title));parent.append(group);
  const body=make('div');group.append(body);
  function draw(){
   body.replaceChildren();
   editor(body,spec,()=>item.value,v=>{item.value=v;});
   select(body,'Availability',['PRESENT','ABSENT_IN_CONTEXT','AMBIGUOUS','UNREADABLE','CONFLICTING','NOT_APPLICABLE'],item.availability,v=>{
    item.availability=v;if(v!=='PRESENT'){item.value=null;item.origin=null;}else if(item.value===null){item.value=initial(spec);item.origin='EXTRACTED';}draw();
   });
   select(body,'Origin',['','EXTRACTED','DERIVED'],item.origin,v=>{item.origin=v||null;});
   const wording=make('details');wording.append(make('summary','Original wording'));body.append(wording);
   editor(wording,{type:'text'},()=>item.raw_text,v=>{item.raw_text=v;});
  }
  draw();
 }
 function editor(parent,spec,get,set){
  const box=make('div');parent.append(box);
  function draw(){
   box.replaceChildren();
   const missing=make('label'),toggle=make('input');toggle.type='checkbox';toggle.checked=get()===null;
   missing.append(toggle,make('span',' No value'));box.append(missing);
   toggle.oninput=event=>{event.stopPropagation();set(toggle.checked?null:initial(spec));draw();notify();};
   if(get()===null)return;
   if(spec.type==='object'){
    for(const [key,child] of Object.entries(spec.fields))nested(box,trainingName(key),child,get()[key]);
   }else if(spec.type==='list'){
    get().forEach((item,index)=>{
     const row=make('div');row.className='training-item';box.append(row);
     if(spec.item.type==='object'){
      const title=make('p','Item '+(index+1));row.append(title);
      for(const [key,child] of Object.entries(spec.item.fields))nested(row,trainingName(key),child,item[key]);
     }else nested(row,'Member '+(index+1),spec.item,item);
     const remove=make('button','Remove item');remove.type='button';remove.onclick=()=>{get().splice(index,1);draw();notify();};row.append(remove);
    });
    const add=make('button','Add item');add.type='button';add.onclick=()=>{get().push(spec.item.type==='object'?initial(spec.item):emptyField());draw();notify();};box.append(add);
   }else if(spec.choices){select(box,'Value',spec.choices,get(),v=>set(v));}
   else{
    const input=make('textarea');input.value=get();input.setAttribute('aria-label',trainingName(field)+' '+(property==='raw_text'?'original wording':'value'));
    input.oninput=event=>{event.stopPropagation();set(input.value);notify();};box.append(input);
   }
  }
  draw();
 }
 Object.defineProperty(host,'value',{get:()=>untouched?'':JSON.stringify(value),set:text=>{
  untouched=text==='';value=untouched?null:JSON.parse(text);host.replaceChildren();editor(host,schema,()=>value,v=>{value=v;});
 }});
 return host;
}
'''
