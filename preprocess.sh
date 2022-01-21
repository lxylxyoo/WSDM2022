#!/bin/bash

python dataprocess_for_tgn_datasetA.py
python dataprocess_for_tgn_datasetA_initial.py
python dataprocess_for_tgn_datasetA_input.py
python dataprocess_for_tgn_datasetA_final.py

python dataprocess_for_tgn_datasetB.py
python dataprocess_for_tgn_datasetB_initial.py
python dataprocess_for_tgn_datasetB_input.py
python dataprocess_for_tgn_datasetB_final.py

v_job_stat=$(expr ${v_job_stat} + $?)
echo "v_job_stat = ${v_job_stat}"
exit ${v_job_stat}