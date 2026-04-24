@echo off
echo =======================================================
echo OutoLLM Baslatiliyor...
echo Lutfen tarayicinizdan asagidaki adrese gidin:
echo http://localhost:8502
echo =======================================================
cd automl_app
streamlit run app.py --server.port 8502 --server.address localhost
