import os, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
from transformers import AutoTokenizer, AutoModel
import uvicorn
M="Qwen/Qwen3-Embedding-0.6B"
tok=AutoTokenizer.from_pretrained(M, padding_side="left"); mdl=AutoModel.from_pretrained(M, torch_dtype=torch.float32).eval()
torch.set_num_threads(8)
app=FastAPI()
class Req(BaseModel):
    input: Union[str, List[str]]
    model: str = M
    encoding_format: str = "float"
@app.get("/v1/models")
def models(): return {"object":"list","data":[{"id":M,"object":"model"}]}
@app.post("/v1/embeddings")
def emb(r: Req):
    xs=[r.input] if isinstance(r.input,str) else r.input
    out=[];n=0
    with torch.no_grad():
        for i in range(0,len(xs),8):
            b=tok(xs[i:i+8],padding=True,truncation=True,max_length=2048,return_tensors="pt"); n+=int(b["attention_mask"].sum())
            h=mdl(**b).last_hidden_state[:,-1]; h=torch.nn.functional.normalize(h,dim=-1); out+=h.tolist()
    return {"object":"list","model":r.model,"data":[{"object":"embedding","index":i,"embedding":e} for i,e in enumerate(out)],"usage":{"prompt_tokens":n,"total_tokens":n}}
uvicorn.run(app,host="127.0.0.1",port=8001)
