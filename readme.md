# Customer service MLOps MVP (production grade)

## 
End-to-end intent classification and retrieval system: SetFit fine-tuning + BentoML deployment + Seldon Canary + Evidently monitoring.

## start
1. `pip install -r requirements.txt`
2. `dvc init` 
3. `python src/data/day1_prep.py` 
4. `docker-compose up` 

## structure
- data/: input data (DVC track)
- src/: code
- models/: models artifact
- deployments/: K8s YAML
- monitoring/: Prometheus/Grafana
- docs/: report

## perfermance
- pin P95 <50ms
- drift PSI >0.25

deatils in docs/FINAL_REPORT.md