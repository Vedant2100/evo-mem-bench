import csv,sys,statistics as st,json
run,lo,hi=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
rows=list(csv.DictReader(open(f"{run}/analysis/category_and_overall.csv")))
ov=[float(r["success_rate"]) for r in rows if r["method"]=="none" and r["category"]=="__overall__"]
cats={}
for r in rows:
    if r["method"]=="none" and r["category"]!="__overall__": cats.setdefault(r["category"],[]).append(float(r["success_rate"]))
m=st.mean(ov); weak=[c for c,v in cats.items() if st.mean(v)<0.30]
print(json.dumps({"none_overall_mean":m,"weak_categories(warning)":weak}))
sys.exit(0 if lo<=m<=hi else 1)
